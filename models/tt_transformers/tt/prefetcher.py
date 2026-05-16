# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import Mode


# Model configurations for which DRAM prefetcher is supported
# TODO #38278: to be removed when model support matrix is unified in tt-transformers
VERIFIED_MODEL_CONFIGS = {
    "Llama-3.2-1B": {"dim": 2048, "hidden_dim": 8192, "n_heads": 32, "n_kv_heads": 8},
    "Llama-3.2-3B": {"dim": 3072, "hidden_dim": 8192, "n_heads": 24, "n_kv_heads": 8},
    "Llama-3.1-8B": {"dim": 4096, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8},
    "Llama-3.3-70B": {"dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8},
    "Qwen3-32B": {"dim": 5120, "hidden_dim": 22016, "n_heads": 40, "n_kv_heads": 8},
    "Qwen3-VL-7B": {"dim": 4096, "hidden_dim": 11008, "n_heads": 32, "n_kv_heads": 8},
    "Qwen3-VL-14B": {"dim": 5120, "hidden_dim": 13824, "n_heads": 40, "n_kv_heads": 8},
    "Qwen3-VL-72B": {"dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8},
    "Gemma3-4B": {"dim": 2560, "hidden_dim": 14336, "n_heads": 20, "n_kv_heads": 20},
    "Gemma3-27B": {"dim": 4608, "hidden_dim": 24576, "n_heads": 32, "n_kv_heads": 8},
    # Tenstorrent-p1: Qwen3 text-only sizes
    "Qwen3-1.7B": {"dim": 2048, "hidden_dim": 6144, "n_heads": 16, "n_kv_heads": 8},
    "Qwen3-8B": {"dim": 4096, "hidden_dim": 12288, "n_heads": 32, "n_kv_heads": 8},
}


def generate_sender_receiver_mapping(num_receivers_per_sender: int = 8) -> dict:
    """
    Generate custom sender->receiver mapping for Blackhole prefetcher.
    Args:
        num_receivers_per_sender (int): Number of receiver cores per sender (8 for 64 total, 10 for 80 total)
    Returns:
        dict: {(sender_x, sender_y): [(rx, ry), ...]} mapping
    """
    cfg = ARCH_CONFIG["blackhole"]
    left_y = cfg["bank_ordered_y_coords"]["left"]
    right_y = cfg["bank_ordered_y_coords"]["right"]
    left_sender_col = cfg["sender_cols"]["left"]
    right_sender_col = cfg["sender_cols"]["right"]
    left_senders = [(left_sender_col, r) for r in left_y]
    right_senders = [(right_sender_col, r) for r in right_y]
    mapping = {}
    for sx, sy in left_senders:
        mapping[(sx, sy)] = [(x, sy) for x in range(1, num_receivers_per_sender + 1)]
    for sx, sy in right_senders:
        cols = list(range(8, 11)) + [x for x in range(8) if x != right_sender_col]
        mapping[(sx, sy)] = [(x, sy) for x in cols[:num_receivers_per_sender]]
    return mapping


def generate_mux_safe_sender_receiver_mapping(
    num_receivers_per_sender: int = 8,
    max_row: int = 7,
) -> dict:
    """
    Generate sender->receiver mapping safe for MUX-clamped Blackhole grids.

    On P300_X2 with MUX dispatch, the worker grid is clamped to rows [0, max_row]
    (typically 12x8 = cols 0-11, rows 0-7). The default YAML config places left
    senders at row 9 which is outside this grid.

    This function remaps out-of-bounds sender Y coords to unused rows within the
    valid range, and excludes core (0,0) from receiver placement to avoid L1
    circular-buffer clash with norm ops.
    """
    cfg = ARCH_CONFIG["blackhole"]
    left_y = cfg["bank_ordered_y_coords"]["left"]
    right_y = cfg["bank_ordered_y_coords"]["right"]
    left_sender_col = cfg["sender_cols"]["left"]
    right_sender_col = cfg["sender_cols"]["right"]

    valid_left_y = [y for y in left_y if y <= max_row]
    valid_right_y = [y for y in right_y if y <= max_row]
    used_y = set(valid_left_y) | set(valid_right_y)
    available_y = sorted(set(range(max_row + 1)) - used_y)

    remapped_left_y = []
    for y in left_y:
        if y <= max_row:
            remapped_left_y.append(y)
        elif available_y:
            replacement = available_y.pop(0)
            logger.info(f"DRAM Prefetcher MUX: remapping left sender row {y} → {replacement} (grid max row {max_row})")
            remapped_left_y.append(replacement)
        else:
            logger.warning(f"DRAM Prefetcher MUX: dropping left sender at row {y} (no rows available within grid)")

    left_senders = [(left_sender_col, r) for r in remapped_left_y]
    right_senders = [(right_sender_col, r) for r in valid_right_y]

    mapping = {}
    for sx, sy in left_senders:
        mapping[(sx, sy)] = [(x, sy) for x in range(1, num_receivers_per_sender + 1)]
    for sx, sy in right_senders:
        cols = list(range(8, 11)) + [x for x in range(8) if x != right_sender_col]
        receivers = [(x, sy) for x in cols if not (x == 0 and sy == 0)]
        mapping[(sx, sy)] = receivers[:num_receivers_per_sender]
    return mapping


def is_prefetcher_supported(model_name: str, num_devices: int, ring_size: int = 16) -> bool:
    """
    Check if model can use DRAM prefetcher: CB pages <= 65535, L1 size fits, kv_heads % num_devices == 0.
    """
    verified_model_name = next((m for m in VERIFIED_MODEL_CONFIGS if m in model_name), None)
    if not is_blackhole() or verified_model_name is None:
        return False
    TILE_SIZE, MAX_CB_PAGES = 32, 65535
    BYTES_PER_TILE_BFP8 = 1088
    MAX_L1_PER_BANK = {4: 1000000, 8: 1000000}.get(num_devices, 850000)
    kv_heads_divisible = VERIFIED_MODEL_CONFIGS[verified_model_name]["n_kv_heads"] % num_devices == 0
    dim, hidden_dim = (
        VERIFIED_MODEL_CONFIGS[verified_model_name]["dim"],
        VERIFIED_MODEL_CONFIGS[verified_model_name]["hidden_dim"],
    )
    n_per_device = hidden_dim // num_devices
    n_per_core = math.ceil(n_per_device / ring_size)
    n_per_core_padded = ((n_per_core + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    n_padded = n_per_core_padded * ring_size
    h_tiles = math.ceil(dim / TILE_SIZE)
    w_tiles = n_padded // TILE_SIZE
    h_tiles_padded = ((h_tiles + ring_size - 1) // ring_size) * ring_size
    tiles_per_core = (h_tiles_padded * w_tiles) // ring_size
    pages_ok = tiles_per_core <= MAX_CB_PAGES
    bytes_per_core = tiles_per_core * BYTES_PER_TILE_BFP8
    l1_ok = bytes_per_core <= MAX_L1_PER_BANK
    logger.info(
        f"DRAM Prefetcher support check: tiles_per_core: {tiles_per_core} <= {MAX_CB_PAGES} is {pages_ok}, bytes_per_core: {bytes_per_core} <= {MAX_L1_PER_BANK} is {l1_ok}, kv_heads_divisible: {kv_heads_divisible}"
    )
    return pages_ok and l1_ok and kv_heads_divisible


@dataclass
class PrefetcherCoreConfig:
    """
    Defines the core locations of the sender and receiver cores for the prefetcher.

    Architecture layout:
    - Blackhole: Left sender column=0, Right sender column=7
                 Left receiver columns=1-6, Right receiver columns=8-10
    - Wormhole:  Left sender column=0, Right sender column=4
                 Left receiver columns=1-3, Right receiver columns=5-6
    """

    # DRAM banks ordered to match ring worker access pattern (by y-coordinate proximity)
    # The bank ids are ordered in column major order from left to right
    DRAM_BANKS = {
        "blackhole": [ttnn.CoreCoord(bank, 0) for bank in [1, 3, 2, 0, 5, 7, 6, 4]],  # 8 banks in total
        "wormhole": [ttnn.CoreCoord(bank, 0) for bank in [1, 2, 3, 0, 4, 6, 9, 10, 11, 8, 7, 5]],  # 12 banks in total
    }

    num_receiver_cores: int
    mesh_device: ttnn.MeshDevice

    # Sender core rows that are adjacent to DRAM banks (different for left and right sides)
    # Active rows are the ones adjacent to DRAM banks, inactive are the remaining rows
    SENDER_ROWS = {
        "blackhole": {
            "left": {"active": [0, 3, 7, 9], "inactive": [1, 2, 4, 5, 6, 8]},
            "right": {"active": [1, 4, 6, 9], "inactive": [0, 2, 3, 5, 7, 8]},
        },
        "wormhole": {
            "left": {"active": [0, 4, 5, 9], "inactive": [1, 2, 3, 6, 7, 8]},
            "right": {"active": [0, 1, 2, 4, 5, 6, 7, 9], "inactive": [3, 8]},
        },
    }

    # Sender core columns (left side near banks 0-3, right side near banks 4-7)
    SENDER_COLS = {
        "blackhole": {"left": 0, "right": 7},
        "wormhole": {"left": 0, "right": 4},
    }

    # Receiver core column ranges (start_col, end_col exclusive)
    RECEIVER_COLS = {
        "blackhole": {"left": (1, 7), "right": (8, 11)},
        "wormhole": {"left": (1, 4), "right": (5, 7)},
    }

    def __post_init__(self):
        arch = "blackhole" if is_blackhole() else "wormhole"
        self._sender_rows = self.SENDER_ROWS[arch]
        self._sender_cols = self.SENDER_COLS[arch]
        self._receiver_cols = self.RECEIVER_COLS[arch]
        self._dram_banks = self.DRAM_BANKS[arch]

    def _get_sender_rows(self, active: Optional[bool], side: str) -> List[int]:
        """Get sender rows based on active filter for a specific side."""
        if active is True:
            return self._sender_rows[side]["active"]
        elif active is False:
            return self._sender_rows[side]["inactive"]
        else:  # None - return all rows
            return self._sender_rows[side]["active"] + self._sender_rows[side]["inactive"]

    def _get_receiver_col_range(self, active: Optional[bool], side: str) -> tuple:
        """Get receiver column range (start, end) based on active filter."""
        start, end = self._receiver_cols[side]
        if active is True:
            return (start, start + self.num_receiver_cores)
        elif active is False:
            return (start + self.num_receiver_cores, end)
        else:  # None - return all columns
            return (start, end)

    def sender_cores(self, active: Optional[bool] = None) -> List[ttnn.CoreCoord]:
        """
        Get sender cores (cores adjacent to DRAM banks).

        Args:
            active: If True, return only active sender cores (one per DRAM bank).
                   If False, return only inactive sender cores.
                   If None, return all sender cores (active first, then inactive).

        The order is: left_active, right_active, left_inactive, right_inactive.
        This ensures the first num_dram_banks cores are the active ones, which
        matches the sub-device configuration.
        """
        left_col = self._sender_cols["left"]
        right_col = self._sender_cols["right"]

        if active is True:
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            return [ttnn.CoreCoord(left_col, r) for r in left_active] + [
                ttnn.CoreCoord(right_col, r) for r in right_active
            ]
        elif active is False:
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return [ttnn.CoreCoord(left_col, r) for r in left_inactive] + [
                ttnn.CoreCoord(right_col, r) for r in right_inactive
            ]
        else:  # None - return all: active first, then inactive
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return (
                [ttnn.CoreCoord(left_col, r) for r in left_active]
                + [ttnn.CoreCoord(right_col, r) for r in right_active]
                + [ttnn.CoreCoord(left_col, r) for r in left_inactive]
                + [ttnn.CoreCoord(right_col, r) for r in right_inactive]
            )

    def receiver_cores(
        self, sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None
    ) -> List[ttnn.CoreRange]:
        """
        Get receiver core ranges (worker cores adjacent to sender cores).

        Each sender core has a horizontal strip of receiver cores on the same row.

        Args:
            sender_active: Filter which sender rows to create receiver ranges for.
            receiver_active: Filter which receiver columns to include.

        The order matches sender_cores: left_active, right_active, left_inactive, right_inactive.
        """
        left_recv = self._get_receiver_col_range(receiver_active, "left")
        right_recv = self._get_receiver_col_range(receiver_active, "right")

        def make_range(col_start, col_end, row):
            return ttnn.CoreRange(ttnn.CoreCoord(col_start, row), ttnn.CoreCoord(col_end - 1, row))

        if sender_active is True:
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            return [make_range(*left_recv, r) for r in left_active] + [make_range(*right_recv, r) for r in right_active]
        elif sender_active is False:
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return [make_range(*left_recv, r) for r in left_inactive] + [
                make_range(*right_recv, r) for r in right_inactive
            ]
        else:  # None - return all: active first, then inactive
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return (
                [make_range(*left_recv, r) for r in left_active]
                + [make_range(*right_recv, r) for r in right_active]
                + [make_range(*left_recv, r) for r in left_inactive]
                + [make_range(*right_recv, r) for r in right_inactive]
            )

    def dram_banks(self) -> List[ttnn.CoreCoord]:
        return self._dram_banks


### Helper class to manage subdevices for the Prefetcher
# The class PrefetcherSubDevice provides an interface for creating subdevices is only managed by the prefetcher module
class PrefetcherSubDevice:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.num_sub_devices = 0
        self.sub_devices: List[ttnn.SubDevice] = []
        self.sub_devices_id: List[ttnn.SubDeviceId] = []

    def add_sub_device(self, core_range_set: ttnn.CoreRangeSet):
        self.sub_devices.append(ttnn.SubDevice([core_range_set]))
        self.sub_devices_id.append(ttnn.SubDeviceId(len(self.sub_devices_id)))

    def init_sub_device_manager(self):
        assert len(self.sub_devices) > 0, "No subdevices have been created. Cannot create sub device manager."
        self.manager_id = self.mesh_device.create_sub_device_manager(self.sub_devices, 0)
        self.mesh_device.load_sub_device_manager(self.manager_id)
        self.mesh_device.set_sub_device_stall_group(self.sub_devices_id)


class Prefetcher(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_tensors: int,
        num_layers: int,
        num_receiver_cores: int = None,
    ):
        """
        Prefetcher class that prefetches tensors from DRAM to
        """
        ### Device, Global CB, Parameters
        assert (
            num_receiver_cores is None or num_receiver_cores in self.legal_receiver_cores
        ), "num_receiver_cores must be in legal_receiver_cores"

        grid = self.mesh_device.compute_with_storage_grid_size()
        # compute_with_storage_grid_size() reports hardware capability (13×10
        # on Blackhole) not the MUX-restricted runtime grid (12×8). Detect
        # MUX from architecture: multi-device Blackhole always uses MUX
        # dispatch which clamps the worker grid to rows 0-7.
        self._mux_clamped = is_blackhole()
        _mux_max_row = 7 if self._mux_clamped else grid.y - 1
        if self._mux_clamped:
            logger.info(
                f"DRAM Prefetcher: Blackhole detected — using MUX-safe mapping "
                f"(max_row={_mux_max_row}, hw grid {grid.x}x{grid.y})"
            )

        def _make_mapping(n_recv):
            if self._mux_clamped:
                return generate_mux_safe_sender_receiver_mapping(n_recv, max_row=_mux_max_row)
            return generate_sender_receiver_mapping(n_recv) if n_recv > 3 else None

        if num_receiver_cores is not None:
            assert is_prefetcher_supported(
                self.model_name, self.mesh_device.get_num_devices(), num_receiver_cores * self.num_senders
            ), "num_receiver_cores is not supported"
            self.num_receiver_cores = num_receiver_cores
            self.receiver_mapping_override = _make_mapping(num_receiver_cores)
        else:
            # Prefer smallest valid ring: fewer receiver cores = more cores
            # available for compute in the 3-sub-device layout (receivers are
            # isolated from the worker grid to prevent L1 CB clashes).
            for num_receivers in self.legal_receiver_cores:
                if is_prefetcher_supported(
                    self.model_name, self.mesh_device.get_num_devices(), num_receivers * self.num_senders
                ):
                    self.num_receiver_cores = num_receivers
                    self.receiver_mapping_override = _make_mapping(num_receivers)
                    break

        ### Core Config
        self.core_config = PrefetcherCoreConfig(
            num_receiver_cores=self.num_receiver_cores, mesh_device=self.mesh_device
        )

        self.dram_banks = self.core_config.dram_banks

        ### Worker core ranges for the worker sub device
        # Use _mux_max_row for the actual runtime grid limit (MUX clamps
        # rows to 0-7 even though compute_with_storage_grid reports 10)
        _grid_max_x = grid.x - 1
        _grid_max_y = _mux_max_row
        full_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_grid_max_x, _grid_max_y))]
        )
        self.all_core_range_set = full_grid
        if self.receiver_mapping_override:
            sender_cores = [
                ttnn.CoreRange(ttnn.CoreCoord(s.x, s.y), ttnn.CoreCoord(s.x, s.y))
                for s in self.core_config.sender_cores(active=True)
            ]
            sender_set = ttnn.CoreRangeSet(sender_cores)
            self.all_worker_cores_range_set = full_grid.subtract(sender_set)
        else:
            left_range = self.core_config._receiver_cols["left"]
            right_range = self.core_config._receiver_cols["right"]
            self.all_worker_cores_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(left_range[0], 0), ttnn.CoreCoord(left_range[1] - 1, _grid_max_y))]
                + [ttnn.CoreRange(ttnn.CoreCoord(right_range[0], 0), ttnn.CoreCoord(right_range[1] - 1, _grid_max_y))]
            )

        ### Prefetched Tensors
        self.callbacks = []
        self.prefetched_tensors = []
        self.prefetched_tensor_addr = []
        self.prefetched_tt_addr_tensor = None

        ### Core Ranges
        self.sender_cores = None
        self.receiver_cores = None
        self.mode = Mode.PREFILL
        self.init_decode_done = False
        self.init_prefill_done = False
        self.prefetch_done = False

    # NOTE: DRAM prefetched weights are prefetched in the order of the construction of the module
    def register_callback(self, callback: Callable[[], None]):
        self.callbacks.append(callback)

    # Mapping from mesh shape (as tuple) to optimal number of receiver cores
    OPTIMAL_RECEIVER_CORES = {
        (1, 1): 2,
        (1, 2): 2,
        (1, 4): 2,
        (1, 8): 1,
    }

    def get_optimal_receiver_cores(self):
        mesh_shape = tuple(self.mesh_device.shape)
        if mesh_shape not in self.OPTIMAL_RECEIVER_CORES:
            supported = list(self.OPTIMAL_RECEIVER_CORES.keys())
            raise ValueError(f"Mesh shape {mesh_shape} is not supported. Supported shapes: {supported}")
        return self.OPTIMAL_RECEIVER_CORES[mesh_shape]

    def to_core_range_set(
        self, cores: Union[List[ttnn.CoreCoord], List[ttnn.CoreRange]], return_list: bool = False
    ) -> ttnn.CoreRangeSet:
        assert len(cores) > 0, "No cores provided to to_core_range_set"
        if isinstance(cores[0], ttnn.CoreCoord):
            return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        elif isinstance(cores[0], ttnn.CoreRange):
            if return_list:  # Return a list of CoreRangeSets (used for creating sender receiver mapping)
                return [ttnn.CoreRangeSet([core]) for core in cores]
            else:  # Return a single CoreRangeSet
                return ttnn.CoreRangeSet(cores)
        else:
            raise ValueError(f"Provided cores {cores} is not a list of CoreCoords or CoreRanges")

    def init(self, mode: Mode = Mode.DECODE) -> None:
        """
        Initializes the prefetcher sub devices
        Args:
            mode: The mode to run the prefetcher in, either "decode" or "prefill"
        NOTE: All DRAM prefetcher APIs can only be called after init() is called for the given mode
        NOTE: Calling init() again for the same mode is a no-op
        """
        # If the prefetcher has already been initialized for the given mode, we do not need to initialize it again
        if mode == Mode.DECODE and self.init_decode_done or mode == Mode.PREFILL and self.init_prefill_done:
            return

        self.mode = mode
        # Get the sender and receiver cores
        # Create a single config instance to ensure consistent state
        self.sender_cores = self.core_config.sender_cores
        self.receiver_cores = self.core_config.receiver_cores

        self.sender_receiver_mapping = list(
            zip(
                self.sender_cores(),
                self.to_core_range_set(self.receiver_cores(sender_active=None, receiver_active=True), return_list=True),
            )
        )
        match mode:
            case Mode.DECODE:
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                sender_set = self.to_core_range_set(self.sender_cores(active=True))
                self.prefetcher_sub_device.add_sub_device(sender_set)
                # 3-sub-device layout: separate receivers from workers so the
                # prefetcher CB on receiver cores doesn't clash with model op
                # L1 allocations (RMSNorm etc.) on compute cores.
                if self.receiver_mapping_override:
                    all_receivers = set()
                    for s in self.sender_cores(active=True):
                        for r_set in self.receiver_cores(sender_active=None, receiver_active=True):
                            for cr in r_set.ranges():
                                for x in range(cr.start_coord.x, cr.end_coord.x + 1):
                                    for y in range(cr.start_coord.y, cr.end_coord.y + 1):
                                        all_receivers.add((x, y))
                    receiver_ranges = [
                        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                        for x, y in sorted(all_receivers)
                    ]
                    if receiver_ranges:
                        receiver_set = ttnn.CoreRangeSet(receiver_ranges)
                        compute_only = self.all_worker_cores_range_set.subtract(receiver_set)
                        self.prefetcher_sub_device.add_sub_device(receiver_set)
                        self.prefetcher_sub_device.add_sub_device(compute_only)
                        self.all_worker_cores_range_set = compute_only
                        logger.info(
                            f"[Prefetcher] 3-sub-device layout: "
                            f"{len(all_receivers)} receiver cores isolated from worker grid"
                        )
                    else:
                        self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
                else:
                    self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()
            case Mode.PREFILL:
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                self.prefetcher_sub_device.add_sub_device(self.all_core_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()

        self.worker_sub_device_id = self.prefetcher_sub_device.sub_devices_id[-1]
        logger.info("=" * 50)
        logger.info("[Prefetcher Initialization]")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Sender cores: {self.sender_cores(active=True)}")
        logger.info(f"  Receiver cores: {self.receiver_cores(sender_active=None, receiver_active=True)}")
        logger.info(f"  Number of receiver cores: {self.num_receiver_cores}")
        logger.info(f"  Number of tensors to prefetch: {self.num_tensors}")
        logger.info(f"  Number of layers: {self.num_layers}")
        logger.warning(
            "Prefetcher has only been thoroughly tested on Llama3.1-8B BH QB 2 and BH LB 1. If using for other models and other device types, expect potential errors."
        )
        logger.info("=" * 50)
        self.init_decode_done = True if mode == Mode.DECODE else False
        self.init_prefill_done = True if mode == Mode.PREFILL else False

    def create_address_tensor(self):
        """
        Creates a ttnn tensor which holds the addresses of the tensors to be prefetched
        The addresses are replicated on each sender core
        """
        assert (
            len(self.prefetched_tensor_addr) == self.num_tensors * self.num_layers
        ), f"Number of tensor addresses have been inserted does not match the number of tensors to prefetch (num_tensors * num_layers), got {len(self.prefetched_tensor_addr)} != {self.num_tensors * self.num_layers}"

        tensor_addrs = torch.tensor(self.prefetched_tensor_addr)
        tensor_addrs = tensor_addrs.repeat(self.mesh_device.dram_grid_size().x, 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.to_core_range_set(self.sender_cores(active=True)),
                [tensor_addrs.shape[0] // self.mesh_device.dram_grid_size().x, tensor_addrs.shape[1]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_tensor_addrs = ttnn.as_tensor(
            tensor_addrs,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=tensor_addrs_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tt_tensor_addrs

    def insert_tensor(self, tensor: ttnn.Tensor):
        """
        Populates the tensor addresses that need to be prefetched
        Args:
            tensor: The tensor to insert into the prefetcher queue
        """
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot insert tensors"
        bytes_in_tile = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
        if tensor.volume() % self.ring_size != 0:
            raise ValueError(
                f"Tensor volume ({tensor.volume()}) must be divisible by ring_size ({self.ring_size}) for prefetcher."
            )
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(
                f"Tensor must be DRAM sharded for prefetcher. Got sharded={tensor.is_sharded()}, "
                f"buffer_type={tensor.memory_config().buffer_type}"
            )
        h, w = tensor.shape[-2], tensor.shape[-1]
        h_tiles, w_tiles = math.ceil(h / ttnn.TILE_SIZE), math.ceil(w / ttnn.TILE_SIZE)
        h_tiles_padded = math.ceil(h_tiles / self.ring_size) * self.ring_size
        max_tensor_tiles = (h_tiles_padded * w_tiles) // self.ring_size
        self.max_tensor_block_size = max(max_tensor_tiles * bytes_in_tile[tensor.dtype], self.max_tensor_block_size)
        self.prefetched_tensors.append(tensor)
        self.prefetched_tensor_addr.append(tensor.buffer_address())
        logger.info(
            f"[DRAM Prefetcher] Inserted tensor of shape {tensor.shape} into prefetcher, total number of tensors in prefetcher queue: {len(self.prefetched_tensor_addr)}"
        )

    def prefetch(self):
        """
        Inserts the tensors to be prefetched in a queue
        The tensors are prefetched in the order of the registration of the callbacks
        NOTE: This only needs to be called if a callback is registered for inserting tensors
        NOTE: prefetch() only needs to be called once and in decode mode, subsequent calls are no-ops
        """
        if self.mode == Mode.DECODE:
            assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot prefetch tensors"
            assert (
                len(self.callbacks) > 0
            ), "No tensors insertion callbacks have been inserted into the prefetcher queue. Cannot prefetch an empty queue"
            if not self.prefetch_done:
                for callback in self.callbacks:
                    callback()
                self.prefetch_done = True
        # NO-OP for prefill mode
        return

    def run(self):
        """
        Start prefetching weights into global CB with dram_prefetcher op
        """
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot run prefetcher"
        # Create global cb buffer if it was not yet created.
        if self.global_cb is None:
            self.global_cb_size = self.max_tensor_block_size
            logger.info(f"[DRAM Prefetcher] Creating global CB with size: {self.global_cb_size}")
            self.global_cb = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )

        # Create address tensor if it was not created yet
        if self.prefetched_tt_addr_tensor is None:
            self.prefetched_tt_addr_tensor = self.create_address_tensor()

        # Run prefetcher op (prefetcher op will start asynchronously prefetching weights until prefetcher.stop() is called)
        self.garbage = ttnn.dram_prefetcher(
            self.prefetched_tensors[: self.num_tensors] + [self.prefetched_tt_addr_tensor],
            num_layers=self.num_layers,
            global_cb=self.global_cb,
            enable_performance_mode=self.enable_performance_mode,
        )
        # Set worker sub device stall group
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device.sub_devices_id[-1]])
        return

    def stop(self):
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot stop prefetcher"
        assert self.garbage is not None, "Prefetcher has not been run. Cannot stop prefetcher"
        ttnn.deallocate(self.garbage)
        self.garbage = None
        return
