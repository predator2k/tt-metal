# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import yaml
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import Mode

# Prefetcher yaml config file describes the sender/receiver core placements
_CONFIG_PATH = Path(__file__).parent / "prefetcher/prefetcher_config.yaml"
with open(_CONFIG_PATH) as f:
    ARCH_CONFIG = yaml.safe_load(f)

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
    # Tenstorrent-p1: Qwen3 text-only sizes added for the SGLang/tt-sglang
    # path. Both have strictly smaller dim/hidden_dim than at least one
    # already-verified Llama in this table (Llama-3.2-1B has hidden_dim=8192
    # vs Qwen3-1.7B 6144; Llama-3.1-8B has hidden_dim=14336 vs Qwen3-8B
    # 12288), so the CB-pages / L1-size constraints in
    # is_prefetcher_supported() pass conservatively.
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
        # Receivers for right senders: columns 8-10, plus columns 0-6 excluding sender column
        cols = list(range(8, 11)) + [x for x in range(8) if x != right_sender_col]
        mapping[(sx, sy)] = [(x, sy) for x in cols[:num_receivers_per_sender]]
    return mapping


def generate_2subdev_safe_sender_receiver_mapping(
    num_receivers_per_sender: int = 4,
    max_row: int = 7,
) -> dict:
    """
    U6 — Galaxy-parity 2-sub-device receiver mapping for Blackhole MUX-clamped grid.

    Differs from `generate_mux_safe_sender_receiver_mapping` in one important way:
    RIGHT-side senders' receivers come exclusively from cols 8-11 (the
    yaml-declared `receiver_cols.right`) instead of borrowing col 0 to make up
    `num_receivers_per_sender`. This is the key requirement for the U1 Attack 1
    galaxy-style 2-sub-device + dummy_receivers fix because:

      - Worker sub-device is constructed as 2 RECTANGLES, one per side
        (e.g. `(1,0)-(6,7)` left, `(8,0)-(11,7)` right). The matmul factory
        bbox-rounds each rect's intersection with receivers (`a.shard_spec.grid`)
        and creates the GlobalCB-attached CB on that bbox; GlobalCB must
        contain those bbox cores (else `CreateCircularBuffer` TT_FATALs at
        circular_buffer.cpp:172).

      - If right-side receivers live in col 0 (the previous mapping's hack),
        they fall OUTSIDE BOTH worker rects, so the worker rect's bbox cannot
        include them — they can't be reached from either rect's matmul cores
        without expanding the worker into col 0, which would have to step
        around the col-0 LEFT-sender rows (i.e. break rectangularity).

    Active sender layout is unchanged (left col 0 / right col 7). Receivers
    for nrc=4: left {1,2,3,4} at sender rows, right {8,9,10,11} at sender rows.

    Args:
        num_receivers_per_sender: Number of receiver cores per sender (must be
            <= 4 for safe right-side col 8-11 placement). Default 4 matches
            Qwen3-8B / Qwen3-32B prefetcher ring config.
        max_row: Maximum y coord on the MUX-clamped runtime grid (default 7).

    Returns:
        dict {(sender_x, sender_y): [(rx, ry), ...]} matching the
        signature of `generate_mux_safe_sender_receiver_mapping`.

    Constraints (asserted):
        - num_receivers_per_sender <= 4 (cols 8-11 give exactly 4 right slots)
        - resulting receiver set must not overlap `dynamic_worker_core_grid`
          cols {5, 6} (no left/right receivers placed there). Holds by
          construction for nrc=4: left uses {1,2,3,4}, right uses {8,9,10,11}.
    """
    assert num_receivers_per_sender <= 4, (
        f"2-subdev mapping requires nrc <= 4 (got {num_receivers_per_sender}); "
        f"right-side cols 8-11 only give 4 slots without leaking into col 5-6 "
        f"(dynamic_worker_core_grid) or col 0/7 (sender columns)."
    )
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
            logger.info(
                f"[Prefetcher 2-subdev] remapping left sender row {y} → {replacement} "
                f"(grid max row {max_row})"
            )
            remapped_left_y.append(replacement)
        else:
            logger.warning(
                f"[Prefetcher 2-subdev] dropping left sender at row {y} (no rows available)"
            )

    left_senders = [(left_sender_col, r) for r in remapped_left_y]
    right_senders = [(right_sender_col, r) for r in valid_right_y]

    mapping = {}
    # Left receivers: cols 1..nrc at sender row
    for sx, sy in left_senders:
        mapping[(sx, sy)] = [(x, sy) for x in range(1, num_receivers_per_sender + 1)]
    # Right receivers: cols 8..(8+nrc-1) at sender row — strictly in the right
    # worker rect (8,0)-(11,7). No col-0 borrowing => no need for col-0
    # receivers and the right worker rect is genuinely rectangular.
    for sx, sy in right_senders:
        mapping[(sx, sy)] = [(8 + i, sy) for i in range(num_receivers_per_sender)]
    return mapping


def compute_2subdev_dummy_receivers(
    worker_rect_list,
    real_receivers_set,
    real_senders_set,
):
    """
    U6 — Compute galaxy-style dummy_receivers for the 2-sub-device + bbox-shortcut path.

    The matmul factory at `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2024-2038`
    iterates over `subdevice_cores.ranges()` (= worker_sub_device.ranges()) and, for
    each rectangular range, takes the BOUNDING BOX of the intersection with
    `non_idle_cores` (= `a.shard_spec().grid` = active receivers). The resulting
    `all_cores` (line 2038) is passed to `CreateCircularBuffer(program, all_cores,
    remote_cb_config, *global_cb)` (line 2157), which TT_FATALs if `all_cores`
    is not a subset of `global_cb->all_cores()` (circular_buffer.cpp:172).

    To make the bbox-rounded `all_cores` valid, the GlobalCB's `all_cores()` must
    include every bbox core. This function computes the set difference:
        dummy_receivers = (bbox(rect ∩ real_receivers) for rect in worker_rects)
                            - real_receivers
                            - real_senders

    These dummy cores are added to the GlobalCB's `sender_receiver_mapping` so
    the membership check passes. They run the matmul reader/writer kernels
    (kernel launches on `all_cores`), but with empty shards they perform no
    actual work — same pattern as galaxy.

    Args:
        worker_rect_list: list[ttnn.CoreRange] — the rectangular pieces of
            worker_sub_device (each one is bbox-shortcut-eligible if non-1D).
        real_receivers_set: set[(x, y)] — actual receiver coords (1:1 with
            tensor shards via sender_receiver_mapping).
        real_senders_set: set[(x, y)] — actual sender coords (must not be
            re-registered as dummy receivers; the GlobalCB enforces
            sender/receiver disjointness at all_cores assembly).

    Returns:
        list[(x, y)] — dummy receiver coords sorted by (x, y), suitable for
        wrapping into one or more CoreRangeSets in the GlobalCB mapping.
    """
    dummies = set()
    for rect in worker_rect_list:
        # Intersect rect with real_receivers
        rect_recv = set()
        for x in range(rect.start.x, rect.end.x + 1):
            for y in range(rect.start.y, rect.end.y + 1):
                if (x, y) in real_receivers_set:
                    rect_recv.add((x, y))
        if not rect_recv:
            # Empty intersection — bbox-shortcut path is not triggered (the
            # factory `continue`s on empty intersection, line 2027). No dummies
            # needed for this rect.
            continue
        # bbox of intersection
        xs = [x for x, _ in rect_recv]
        ys = [y for _, y in rect_recv]
        bb_x_min, bb_x_max = min(xs), max(xs)
        bb_y_min, bb_y_max = min(ys), max(ys)
        for x in range(bb_x_min, bb_x_max + 1):
            for y in range(bb_y_min, bb_y_max + 1):
                c = (x, y)
                if c in real_receivers_set:
                    continue
                if c in real_senders_set:
                    # Should never happen since senders are in sender_cols
                    # (0, 7) and the worker rects exclude those cols by
                    # construction; assert in case of misconfig.
                    raise AssertionError(
                        f"dummy_receivers bbox {c} overlaps a real sender — "
                        f"worker rect {rect.start}-{rect.end} should exclude "
                        f"sender columns."
                    )
                dummies.add(c)
    return sorted(dummies)


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
    Args:
        model_name (str): Model name (must contain a key from VERIFIED_MODEL_CONFIGS)
        num_devices (int): Number of devices for tensor parallelism
        ring_size (int): Total receiver cores (default 16, custom mapping uses 64/80)
    Returns:
        bool: True if supported on Blackhole with given config, False otherwise
    """
    verified_model_name = next((m for m in VERIFIED_MODEL_CONFIGS if m in model_name), None)
    if not is_blackhole() or verified_model_name is None:
        return False
    TILE_SIZE, MAX_CB_PAGES = 32, 65535
    BYTES_PER_TILE_BFP8 = 1088  # bfloat8_b tile size in bytes
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
    # Check memory constraints and kv heads divisible by num_devices
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
    Core locations for prefetcher sender/receiver cores.

    If receiver_mapping_override is provided, its keys become the sender cores and values
    become their receivers (all treated as "active"). This allows full custom placement.
    Otherwise, uses default architecture-specific sender/receiver layout.
    """

    num_receiver_cores: int
    mesh_device: ttnn.MeshDevice
    cfg: dict
    receiver_mapping_override: Optional[dict] = None  # {(x,y): [(rx,ry), ...]}

    def __post_init__(self):
        self._dram_banks = [ttnn.CoreCoord(b, 0) for b in self.cfg["dram_banks"]]
        self._sender_cols, self._sender_rows = self.cfg["sender_cols"], self.cfg["sender_rows"]
        self._receiver_cols = {k: tuple(v) for k, v in self.cfg["receiver_cols"].items()}
        self._use_override = self.receiver_mapping_override is not None

        # Process override: keys become senders, values become receivers
        self._override_senders = []  # List[CoreCoord] - ordered sender cores from override
        self._override_receivers = {}  # {(x,y): [CoreCoord]} - receivers per sender
        if self._use_override:
            for k, v in self.receiver_mapping_override.items():
                sender = k if isinstance(k, ttnn.CoreCoord) else ttnn.CoreCoord(k[0], k[1])
                self._override_senders.append(sender)
                key = (sender.x, sender.y)
                self._override_receivers[key] = [
                    c if isinstance(c, ttnn.CoreCoord) else ttnn.CoreCoord(c[0], c[1]) for c in v
                ]

    def _get_rows(self, active: Optional[bool], side: str) -> List[int]:
        rows = self._sender_rows[side]
        if active is True:
            return rows["active"]
        if active is False:
            return rows["inactive"]
        return rows["active"] + rows["inactive"]

    def _get_col_range(self, active: Optional[bool], side: str) -> tuple:
        start, end = self._receiver_cols[side]
        if active is True:
            return (start, start + self.num_receiver_cores)
        if active is False:
            return (start + self.num_receiver_cores, end)
        return (start, end)

    def sender_cores(self, active: Optional[bool] = None) -> List[ttnn.CoreCoord]:
        """Get sender cores. With override, all senders are 'active'. Without, uses default layout."""
        if self._use_override:
            # With override: all senders from override keys, no inactive concept
            return self._override_senders if active is None or active is True else []
        # Default behavior
        lc, rc = self._sender_cols["left"], self._sender_cols["right"]
        if active is True:
            return [ttnn.CoreCoord(lc, r) for r in self._get_rows(True, "left")] + [
                ttnn.CoreCoord(rc, r) for r in self._get_rows(True, "right")
            ]
        if active is False:
            return [ttnn.CoreCoord(lc, r) for r in self._get_rows(False, "left")] + [
                ttnn.CoreCoord(rc, r) for r in self._get_rows(False, "right")
            ]
        return (
            [ttnn.CoreCoord(lc, r) for r in self._get_rows(True, "left")]
            + [ttnn.CoreCoord(rc, r) for r in self._get_rows(True, "right")]
            + [ttnn.CoreCoord(lc, r) for r in self._get_rows(False, "left")]
            + [ttnn.CoreCoord(rc, r) for r in self._get_rows(False, "right")]
        )

    def _get_receivers(self, sender: ttnn.CoreCoord, receiver_active: Optional[bool]) -> List[ttnn.CoreCoord]:
        key = (sender.x, sender.y)
        if self._use_override:
            # With override: return all receivers for this sender (no active/inactive split)
            return self._override_receivers.get(key, [])
        # Default behavior
        side = "left" if sender.x == self._sender_cols["left"] else "right"
        col_start, col_end = self._get_col_range(receiver_active, side)
        return [ttnn.CoreCoord(c, sender.y) for c in range(col_start, col_end)]

    def receiver_cores(
        self, sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None
    ) -> List[ttnn.CoreRangeSet]:
        """Get receiver ranges per sender. Returns CoreRangeSet of receiver cores for each sender.

        Always creates individual CoreRange for each receiver to ensure consistent
        CoreRangeSet size across all senders (required by global circular buffer).
        """
        result = []
        for sender in self.sender_cores(active=sender_active):
            receivers = self._get_receivers(sender, receiver_active)
            if not receivers:
                continue
            # Always create individual CoreRanges for each receiver
            # This ensures all senders have the same CoreRangeSet structure
            result.append(ttnn.CoreRangeSet([ttnn.CoreRange(r, r) for r in receivers]))
        return result

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
        # Only include the worker sub-device (last) in the stall group.
        # Sender and receiver sub-devices run persistent kernels that never
        # generate completion signals; including them in the stall group
        # causes finish_nolock({}) — used internally by populate_mesh_buffer,
        # cpu(), and blocking buffer ops — to hang indefinitely.
        self.mesh_device.set_sub_device_stall_group([self.sub_devices_id[-1]])


class Prefetcher(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_tensors: int,
        num_layers: int,
        num_receiver_cores: int = None,
    ):
        """
        Prefetcher class that prefetches tensors from DRAM to L1.

        Args:
            receiver_mapping_override: If provided, keys become sender cores and values become
                their receiver cores. This overrides the default column 0/7 sender placement.
        """
        ### Device, Global CB, Parameters
        self.pf_config: dict = ARCH_CONFIG["blackhole"]
        self.legal_receiver_cores: List[int] = self.pf_config["legal_receiver_cores"]
        self.mesh_device: ttnn.MeshDevice = mesh_device
        self.enable_performance_mode: bool = True
        self.global_cb: Optional[ttnn.GlobalCircularBuffer] = None
        self.worker_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.receiver_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.num_tensors: int = num_tensors
        self.num_layers: int = num_layers
        self.num_senders: int = len(self.pf_config["dram_banks"])
        self.global_cb_size: int = 0  # Size of the global circular buffer in bytes storing prefetched matmul weights
        self.max_tensor_block_size: int = 0  # Max tensor block size is the largest block size of a tensor in bytes
        self.receiver_mapping_override: Optional[dict] = None
        self.model_name = os.getenv("HF_MODEL", "")
        assert self.model_name != "", "HF_MODEL is not set. DRAM Prefetcher must be run with a model."
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

        # U6 — env-gated galaxy-parity 2-sub-device path.
        # SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1 switches the sender/receiver
        # mapping AND the init() sub-device layout to galaxy's pattern:
        # rectangular worker_sub_device + dummy_receivers padding the
        # GlobalCB to cover the matmul bbox. See `init(Mode.DECODE)` and
        # `run()` below for the matched changes.
        self._real_2subdev = os.environ.get("SGLANG_TT_PREFETCHER_REAL_2SUBDEV", "0") == "1"
        if self._real_2subdev:
            logger.info(
                "[Prefetcher] U6: SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1 — "
                "using galaxy-parity 2-sub-device + dummy_receivers layout"
            )

        def _make_mapping(n_recv):
            if self._real_2subdev and self._mux_clamped:
                return generate_2subdev_safe_sender_receiver_mapping(n_recv, max_row=_mux_max_row)
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
            num_receiver_cores=self.num_receiver_cores,
            mesh_device=self.mesh_device,
            cfg=self.pf_config,
            receiver_mapping_override=self.receiver_mapping_override,
        )
        self.ring_size = self.num_receiver_cores * self.num_senders
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

        ### Dynamic worker core grid: deferred to method; see dynamic_worker_core_grid() below.

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
        self._prefill_manager_id = None
        self._decode_manager_id = None

        # Per-layer bisect knob (ATTACK 1 for prefetcher-correctness bug).
        # SGLANG_TT_PREFETCHER_LAYERS=N limits the prefetcher producer/consumer
        # path to the first N decoder layers; layers N..(num_layers-1) fall
        # back to plain DRAM-sharded matmul (no GlobalCB). Default (env unset
        # or N >= num_layers) preserves prior behavior — prefetcher active
        # for ALL layers. Read env at __init__ but only resolve against
        # num_layers in `effective_layer_limit` (some callers set num_layers
        # AFTER __init__ via prefetcher.num_layers = ...).
        try:
            self._env_layer_limit = int(os.getenv("SGLANG_TT_PREFETCHER_LAYERS", "0") or 0)
        except ValueError:
            self._env_layer_limit = 0

    @property
    def layer_limit(self):
        """Resolve the per-layer bisect limit against the (possibly late-set)
        num_layers attribute. Returns num_layers (no limit) if env unset,
        invalid, or out of range."""
        n = self.num_layers
        if not isinstance(n, int) or n <= 0:
            return n
        lim = getattr(self, "_env_layer_limit", 0)
        if lim > 0 and lim < n:
            return lim
        return n

    def use_for_layer(self, layer_num) -> bool:
        """Return True iff the prefetcher consumer path should be used for layer_num.

        Used by attention.py / mlp.py to short-circuit the GlobalCB consumer
        kwargs (global_cb, sub_device_id, ring-sharded weight, prefetcher-aware
        program / mem configs) when ATTACK-1 per-layer bisect is engaged.
        """
        try:
            n = int(layer_num)
        except (TypeError, ValueError):
            return True
        if n < 0:
            return True
        lim = self.layer_limit
        if not isinstance(lim, int) or lim <= 0:
            return True
        return n < lim

    @property
    def worker_start_core(self):
        """First valid start_core inside the (possibly carved-by-receivers) worker grid."""
        ranges = self.all_worker_cores_range_set.ranges()
        if not ranges:
            return ttnn.CoreCoord(1, 0)
        return ranges[0].start

    def dynamic_worker_core_grid(self, num_cores):
        """Return a rectangular CoreRangeSet of worker-only cores for sharded ops.

        Returns the safe rectangular worker zone for the MUX-clamped Blackhole grid.
        All callers (.num_cores()) read the actual core count from the returned value.

        Design constraints:
          - Must be a SINGLE rectangle (LayerNorm validator: num_cores == bbox_num_cores)
          - Must lie entirely within the worker sub-device (no sender or receiver cores)
          - Must remain valid before AND after prefetcher.init(Mode.DECODE)

        MUX-clamped Blackhole layout for nrc ≤ 4 (the only valid nrc for Qwen3-8B TP=2):
          Left senders: col 0, rows {1,3,5,7}
          Right senders: col 7, rows {0,2,4,6}
          Left receivers (nrc=4): cols 1–4, rows {1,3,5,7} (at left-sender rows)
          Right receivers (nrc=4): cols {8,9,10,1} at right-sender rows {0,2,4,6}

        Cols 5–6 have NO senders and NO receivers for any nrc ≤ 4 → ALL 8 rows are workers.
        (5,0)–(6,7) is a 2×8 = 16 core rectangle fully within the compute sub-device.

        Using 16 cores for both residual (requested 16) and norm (requested 32) shards:
          - num_cores() = 16 → callers adapt shard_width = dim // 16 accordingly
          - prefetcher_norm_grid must also be 16 cores (CoreGrid(y=8, x=2))
          - ShardedLayerNorm: shard bbox (5,0)-(6,7) shifted to (0,0)-(1,7) fits in 2×8 ✓

        num_cores_to_corerangeset_in_subcoregrids() is intentionally NOT used here: it
        produces non-rectangular results when all_worker_cores_range_set has holes (sender
        or receiver cores subtracted), causing both "Sharded layernorm does not support
        non-rectangular core grids" TT_FATAL and multi-sub-device dispatch crashes.
        """
        # Safe worker-only rectangle: cols 5-6, all rows 0-7
        # Valid for any nrc in legal_receiver_cores on MUX-clamped Blackhole (P150a, P300_X2).
        # num_cores argument is accepted but the actual core count is always 16;
        # callers use .num_cores() on the returned value so shard widths adapt automatically.
        _WORKER_START_X = 5
        _WORKER_END_X = 6  # 2 cols × 8 rows = 16 cores
        _WORKER_START_Y = 0
        _WORKER_END_Y = 7
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(
                ttnn.CoreCoord(_WORKER_START_X, _WORKER_START_Y),
                ttnn.CoreCoord(_WORKER_END_X, _WORKER_END_Y),
            )]
        )

    # NOTE: DRAM prefetched weights are prefetched in the order of the construction of the module
    def register_callback(self, callback: Callable[[], None]):
        self.callbacks.append(callback)

    def to_core_range_set(
        self, cores: List, return_list: bool = False
    ) -> Union[ttnn.CoreRangeSet, List[ttnn.CoreRangeSet]]:
        """Convert cores (CoreCoord/CoreRange/CoreRangeSet) to CoreRangeSet(s)."""
        assert cores, "No cores provided"

        def to_ranges(c):
            if isinstance(c, ttnn.CoreRangeSet):
                return c.ranges()
            elif isinstance(c, ttnn.CoreRange):
                return [c]
            elif isinstance(c, ttnn.CoreCoord):
                return [ttnn.CoreRange(c, c)]
            raise ValueError(f"Unsupported core type: {type(c)}")

        if return_list:
            return [ttnn.CoreRangeSet(to_ranges(c)) for c in cores]
        return ttnn.CoreRangeSet([r for c in cores for r in to_ranges(c)])

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
                # U6: in galaxy-parity 2-sub-device mode we include the
                # dummy-sender cores (otherwise idle col 0/7 rows) in the
                # sender_sub_device so the union of (sender ∪ worker)
                # equals the full compute grid. Without this, ops that
                # auto-determine sub-device IDs based on a kernel grid
                # covering compute_with_storage_grid_size (e.g. unsharded
                # ttnn.rms_norm) TT_FATAL at program.cpp:1808
                # `num_intersections == num_cores` because the kernel
                # group spans cores belonging to NO sub-device.
                # The existing 3-sub-device path covered the full grid via
                # (sender ∪ receiver ∪ compute_only).
                sender_cores_list = list(self.sender_cores(active=True))
                if self._real_2subdev and self.receiver_mapping_override:
                    # Compute dummy sender candidates = unused col 0/7 rows.
                    _real_senders_set = {(s.x, s.y) for s in sender_cores_list}
                    _max_y = max(
                        (r.end.y for r in self.all_core_range_set.ranges()),
                        default=7,
                    )
                    for col in (0, 7):
                        for row in range(_max_y + 1):
                            if (col, row) not in _real_senders_set:
                                sender_cores_list.append(ttnn.CoreCoord(col, row))
                sender_set = self.to_core_range_set(sender_cores_list)
                self.prefetcher_sub_device.add_sub_device(sender_set)
                # U6 — Galaxy-parity 2-sub-device path with rectangular worker
                # + dummy_receivers padding the GlobalCB. Bypasses the 3-subdev
                # carve-out so matmul + downstream CCL both auto-determine
                # `worker_sub_device_id` (since worker now includes receivers),
                # eliminating the U1 cross-sub-device dispatch sync gap.
                if self._real_2subdev and self.receiver_mapping_override:
                    # Compute real sender/receiver sets (for dummy enumeration).
                    real_senders_set = {(s.x, s.y) for s in self.sender_cores(active=True)}
                    real_receivers_set = set()
                    for r_set in self.receiver_cores(sender_active=None, receiver_active=True):
                        for cr in r_set.ranges():
                            for x in range(cr.start.x, cr.end.x + 1):
                                for y in range(cr.start.y, cr.end.y + 1):
                                    real_receivers_set.add((x, y))
                    # Build worker_sub_device as 2 RECTANGLES (galaxy parity).
                    # Senders live in cols 0 (left) and 7 (right); exclude
                    # those columns entirely from worker → 2 clean rects.
                    # _grid_max_y was computed at __init__ as _mux_max_row.
                    _max_y = max(
                        (r.end.y for r in self.all_core_range_set.ranges()),
                        default=7,
                    )
                    _max_x = max(
                        (r.end.x for r in self.all_core_range_set.ranges()),
                        default=11,
                    )
                    left_rect = ttnn.CoreRange(
                        ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, _max_y)
                    )
                    right_rect = ttnn.CoreRange(
                        ttnn.CoreCoord(8, 0), ttnn.CoreCoord(_max_x, _max_y)
                    )
                    worker_rects = [left_rect, right_rect]
                    worker_set = ttnn.CoreRangeSet(worker_rects)
                    # Compute dummy_receivers to pad GlobalCB.
                    self._dummy_receiver_coords = compute_2subdev_dummy_receivers(
                        worker_rects, real_receivers_set, real_senders_set,
                    )
                    self.prefetcher_sub_device.add_sub_device(worker_set)
                    # In 2-sub-device layout the worker INCLUDES receivers; so
                    # receiver_sub_device_id should resolve to worker_sub_device_id.
                    self._receiver_sub_device_idx = None
                    # Track for downstream all_worker_cores_range_set queries
                    # (used by lm_head / distributed_norm etc).
                    self.all_worker_cores_range_set = worker_set
                    logger.info(
                        f"[Prefetcher] U6 2-sub-device layout: "
                        f"worker_rects={[(r.start, r.end) for r in worker_rects]}, "
                        f"real_receivers={len(real_receivers_set)}, "
                        f"dummy_receivers={len(self._dummy_receiver_coords)}"
                    )
                elif self.receiver_mapping_override:
                    all_receivers = set()
                    for r_set in self.receiver_cores(sender_active=None, receiver_active=True):
                        for cr in r_set.ranges():
                            for x in range(cr.start.x, cr.end.x + 1):
                                for y in range(cr.start.y, cr.end.y + 1):
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
                        # sub_devices_id layout: [sender(0), receiver(1), worker(2)]
                        # The QKV ring-gather matmul takes x sharded on receiver cores;
                        # store receiver sub_device_id so the factory intersection
                        # (subdevice_cores ∩ x.shard_spec().grid) is non-empty.
                        self._receiver_sub_device_idx = 1
                        logger.info(
                            f"[Prefetcher] 3-sub-device layout: "
                            f"{len(all_receivers)} receiver cores isolated from worker grid"
                        )
                    else:
                        self._receiver_sub_device_idx = None
                        self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
                else:
                    self._receiver_sub_device_idx = None
                    self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()
                self._decode_manager_id = self.prefetcher_sub_device.manager_id
                # Layer 15 fix: the program cache may hold entries compiled under the
                # default manager's core partition. After switching to the decode
                # sub-device manager (3-sub-device geometry), stale cache hits cause
                # "Kernel group cores do not match sub device cores" at program.cpp:1811.
                # Flush and re-enable so subsequent compilations use the new geometry.
                self.mesh_device.disable_and_clear_program_cache()
                self.mesh_device.enable_program_cache()
                logger.info("[Prefetcher] Program cache cleared after decode sub-device manager switch")
            case Mode.PREFILL:
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                self.prefetcher_sub_device.add_sub_device(self.all_core_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()
                self._prefill_manager_id = self.prefetcher_sub_device.manager_id

        self.worker_sub_device_id = self.prefetcher_sub_device.sub_devices_id[-1]
        # In 3-sub-device layout: sub_devices_id[1] is the receiver sub_device.
        # Fall back to worker_sub_device_id for 2-sub-device layouts (no isolated receivers).
        _ridx = getattr(self, "_receiver_sub_device_idx", None)
        self.receiver_sub_device_id = (
            self.prefetcher_sub_device.sub_devices_id[_ridx]
            if _ridx is not None
            else self.worker_sub_device_id
        )
        logger.info("=" * 50)
        logger.info("[Prefetcher Initialization]")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Sender cores: {self.sender_cores(active=True)}")
        logger.info(f"  Receiver cores: {self.receiver_cores(sender_active=None, receiver_active=True)}")
        logger.info(f"  Number of receiver cores: {self.num_receiver_cores}")
        logger.info(f"  Number of tensors to prefetch: {self.num_tensors}")
        logger.info(f"  Number of layers: {self.num_layers}")
        logger.warning(
            f"DRAM Prefetcher has only been tested on these models: {list(VERIFIED_MODEL_CONFIGS.keys())} on BH DB, QB, LB. If using other models and other device types, expect potential errors. To check if the model is supported on the current device type, run is_prefetcher_supported(model_name, num_devices, ring_size)."
        )
        logger.info("=" * 50)
        self.init_decode_done = True if mode == Mode.DECODE else False
        self.init_prefill_done = True if mode == Mode.PREFILL else False

    def load_prefill_manager(self):
        """Activate the sub-device manager that was current when the prefill trace was captured."""
        if self._prefill_manager_id is not None:
            self.mesh_device.load_sub_device_manager(self._prefill_manager_id)

    def load_decode_manager(self):
        """Restore the decode sub-device manager after a prefill trace replay.

        load_sub_device_manager resets the stall group to all sub-device IDs
        ({0,1,2}) via reset_sub_device_stall_group().  We must re-apply the
        worker-only restriction immediately afterward, otherwise finish_nolock({})
        will wait for the persistent sender/receiver kernels and hang forever.
        """
        if self._decode_manager_id is not None:
            self.mesh_device.load_sub_device_manager(self._decode_manager_id)
            # Re-restrict stall group to worker sub-device only (see init_sub_device_manager).
            self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device.sub_devices_id[-1]])

    def load_default_manager(self):
        """Revert to the default (full-device, no sub-device partitioning) manager.

        Use this before running prefill ops while the decode 3-sub-device manager
        is active.  Under the default manager, ttnn operations use the full device
        grid without sub-device intersection validation, avoiding the
        "Kernel group cores do not match sub device cores" TT_FATAL that occurs
        when any op's kernel grid spans inactive sender/receiver rows (which are
        not assigned to any sub-device in the 3-sub-device decode layout).
        """
        self.mesh_device.clear_loaded_sub_device_manager()

    def ensure_prefill_manager(self):
        """Switch to the default (full-device, no sub-device partitioning) manager for prefill ops.

        Fix 16.5: the prior implementation created a constrained sub-device using
        all_core_range_set (rows 0-7 on MUX-clamped Blackhole), which is narrower than
        the full hardware grid (9 rows on 2×P150a).  Ops such as rotary_embedding_llama /
        rotary_embedding_hf use device->compute_with_storage_grid_size() to pick their
        kernel grid, which returns the full 9-row hardware grid.  The intersection check
        at program.cpp:1811 then fires: "Kernel group cores do not match sub device cores".

        Replacing the constrained sub-device with the DEFAULT manager
        (clear_loaded_sub_device_manager) disables intersection validation entirely, so
        ops may target any hardware cores, exactly as they did before the prefetcher was
        introduced.  Decode remains unaffected because load_decode_manager() is called
        after each prefill trace replay.
        """
        self.load_default_manager()
        logger.info("[Prefetcher] ensure_prefill_manager: activated default manager (full grid, no intersection check)")

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
        w_tiles_padded = math.ceil(w_tiles / self.ring_size) * self.ring_size
        max_tensor_tiles = (h_tiles_padded * w_tiles_padded) // self.ring_size
        # 2026-05-23 partial-BF16 fix: the C++ check at
        # dram_prefetcher_program_factory.cpp:108 computes
        #   max_tensor_size = max(tile_size_across_tensors) * max(block_tiles_across_tensors)
        # i.e. it takes the cross-product of the LARGEST tile size and the
        # LARGEST block-tile count, which may come from DIFFERENT tensors. The
        # original Python computed only the per-tensor product max, which
        # under-sizes the GlobalCB when partial-BF16 modes mix BF16 (large
        # tile) with BFP8 (large block-tile count). Track both maxes
        # separately and use the cross product to match the C++ sizing.
        self._max_tile_bytes = max(getattr(self, "_max_tile_bytes", 0), bytes_in_tile[tensor.dtype])
        self._max_block_tiles = max(getattr(self, "_max_block_tiles", 0), max_tensor_tiles)
        self.max_tensor_block_size = max(
            self._max_tile_bytes * self._max_block_tiles,
            self.max_tensor_block_size,
        )
        # U32 — track per-tensor byte offset within the GlobalCB.
        # The prefetcher writer (writer_l1.cpp) writes tensors sequentially:
        # for each tensor t, writes num_blocks (= ring_size) blocks of
        # block_size_per_receiver = (tensor_block_num_tiles * single_tile_bytes) / num_receivers_per_reader.
        # On Blackhole the GCB-per-receiver is ring_size's slice, so the
        # per-receiver block_size = tensor_block_num_tiles * tile_size_bytes / num_senders.
        # Equivalently: per-receiver per-tensor total bytes = ring_size * (block_num_tiles * tile_bytes) / num_senders
        #              = block_num_tiles * tile_bytes * num_receiver_cores.
        # Use the receiver-side semantics that match the matmul kernel:
        #   in1_block_size_bytes = max_tensor_tiles * tile_bytes  (per-receiver block, per ring iteration)
        #   per_tensor_bytes     = ring_size * in1_block_size_bytes
        # Wait wait — careful.  Each receiver gets ring_size's worth of
        # blocks of the tensor (one per ring iteration / ring_idx).
        # Producer writes per receiver: ring_size blocks of size
        # block_size_per_receiver = (block_num_tiles*tile_bytes)/num_receivers_per_reader.
        # Since ring_size = num_senders * num_receivers_per_reader (= num_blocks
        # in the producer), and each receiver only owns 1/num_receivers_per_reader
        # of the per-sender stride... the per-receiver advance after one
        # tensor is num_blocks * block_size_per_receiver bytes.
        # The matmul's `in1_block_size_bytes` (compile-time) matches the
        # producer's `block_size_per_receiver`, so:
        #   per_tensor_bytes_per_receiver = num_blocks * in1_block_size_bytes
        #                                 = ring_size * (block_num_tiles * tile_bytes / num_receivers_per_reader)
        # We don't know num_receivers_per_reader at this point but it
        # equals self.num_receiver_cores.  Compute that here.
        per_tensor_block_size_per_receiver = (
            max_tensor_tiles * bytes_in_tile[tensor.dtype]
        ) // self.num_receiver_cores
        per_tensor_bytes_per_receiver = self.ring_size * per_tensor_block_size_per_receiver

        if not hasattr(self, "_tensor_byte_offsets"):
            self._tensor_byte_offsets = []
        if not hasattr(self, "_running_tensor_offset_bytes"):
            self._running_tensor_offset_bytes = 0
        self._tensor_byte_offsets.append(self._running_tensor_offset_bytes)
        self._running_tensor_offset_bytes += per_tensor_bytes_per_receiver

        self.prefetched_tensors.append(tensor)
        self.prefetched_tensor_addr.append(tensor.buffer_address())
        logger.info(
            f"[DRAM Prefetcher] Inserted tensor of shape {tensor.shape} into prefetcher, "
            f"total in queue: {len(self.prefetched_tensor_addr)} "
            f"u32_offset_bytes={self._tensor_byte_offsets[-1]} "
            f"u32_per_tensor_bytes={per_tensor_bytes_per_receiver}"
        )

    def get_tensor_gcb_offset_bytes(self, tensor: ttnn.Tensor) -> int:
        """U32 — return the per-tensor byte offset within the GlobalCB
        for this tensor (or 0 if not registered or U32 isn't tracked).
        Used by mlp.py / attention.py to set
        SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES before each ttnn.linear
        call so the gathered matmul reads from the GCB region where
        the prefetcher actually wrote this tensor's data.
        Modulo the GCB-size wrap: the producer's wr_ptr wraps at
        (gcb_size - gcb_size%page_size) which for Qwen3-8B-balanced
        is 835584 (page_size divides 835584 evenly for every tensor).
        We apply modulo gcb_size here.

        U33 (2026-05-25) — *consumer-side* offset computation.
        The U32 Python formula tried to mirror the producer's per-tensor
        per-receiver write byte count (`num_blocks *
        block_size_per_receiver`), where block_size_per_receiver depends
        on `tensor_block_num_tiles = shard_h_tiles * shard_w_tiles /
        num_blocks`.  But the matmul **CONSUMER** reads tensor t at
        `num_blocks * in1_block_size_bytes[t]` bytes per dispatch, where
        `in1_block_size_bytes[t] = per_core_N[t] * in0_block_w[t] *
        tile_bytes[t]`.  Producer and consumer write/read the SAME data
        for tensor t — so the cumulative offset for tensor t equals the
        cumulative *consumer* bytes for all tensors written-and-read
        before t.  Computing this Python-side requires knowing each
        tensor's matmul program_config — which is known Python-side.
        Caller (mlp.py / attention.py) registers the per-tensor consumer
        bytes via `set_tensor_consumer_bytes()` BEFORE the first
        ttnn.linear dispatch.  Until registered, fall back to the U32
        producer-side estimate (preserves the no-op force-zero path).
        Env-gated SGLANG_TT_U33_FACTORY_OFFSET=1 (the misnomer — the
        OFFSET COMPUTATION runs Python-side; the env var name is kept
        for plan continuity with U33).
        """
        if not hasattr(self, "_tensor_byte_offsets") or not self.prefetched_tensors:
            return 0
        # U32 — force-zero offset escape hatch.  When SGLANG_TT_U32_FORCE_ZERO=1,
        # always return 0 regardless of which tensor.  Lets us isolate "U32
        # kernel define is benign at offset=0" from "U32 offset values are
        # incorrect".  Default 0 (real per-tensor offsets used).
        if os.environ.get("SGLANG_TT_U32_FORCE_ZERO", "0") == "1":
            return 0
        # U34 fix — lookup by Python identity + buffer_address fallback.
        # ttnn.Tensor.__eq__ may return a tensor (not bool), making
        # list.index() unreliable: it can return 0 (first match)
        # spuriously.  Mirror the U33 set_tensor_consumer_bytes lookup
        # explicitly here.
        idx = None
        for i, t in enumerate(self.prefetched_tensors):
            if t is tensor:
                idx = i
                break
        if idx is None:
            try:
                tgt_addr = tensor.buffer_address()
            except Exception:
                tgt_addr = None
            if tgt_addr is not None:
                for i, addr in enumerate(self.prefetched_tensor_addr):
                    if addr == tgt_addr:
                        idx = i
                        break
        if idx is None:
            return 0
        # Per-layer tensor list is the FIRST num_tensors entries; per-layer
        # offsets are the same across layers (producer's wr_ptr keeps the
        # same relative position MODULO the GCB wrap, and the matmul reads
        # from fifo_start + per_tensor_offset on EVERY layer's dispatch).
        # Take the offset from the per-layer position (idx % num_tensors).
        per_layer_idx = (idx % self.num_tensors) if self.num_tensors > 0 else idx

        # U33 — prefer consumer-side cumulative when SGLANG_TT_U33_FACTORY_OFFSET=1.
        # Use whichever consumer bytes have been registered so far; treat
        # unset entries as 0.  This means iteration-1 may use partial info
        # but the registrations all happen during the first pass and become
        # bytewise-stable from iteration 2 onwards.
        use_u33 = (
            os.environ.get("SGLANG_TT_U33_FACTORY_OFFSET", "0") == "1"
            and getattr(self, "_tensor_consumer_bytes", None) is not None
            and len(self._tensor_consumer_bytes) > 0
        )
        # U34 — producer-simulation align mode.  When SGLANG_TT_U34_LCM_ALIGN=1,
        # mirror the writer_l1.cpp + resize_remote_sender_cb_interface logic
        # exactly so each returned offset matches the producer's actual
        # `fifo_wr_ptr` at the start of tensor t.  The producer's
        # `resize_remote_sender_cb_interface` (remote_circular_buffer.h:110)
        # at the start of each tensor sets page_size = block_size_per_receiver
        # (= consumer's in1_block_size_bytes), ALIGN-UPs wr_ptr to that
        # page_size, and wraps to fifo_start if past fifo_limit_page_aligned
        # (= gcb_size - gcb_size%page_size).  Then writes num_blocks blocks
        # of `page_size` each, with mid-tensor wrap when dest_addr ==
        # fifo_limit_page_aligned (remote_circular_buffer.h:378).  Because the
        # producer per-tensor re-aligns to the tensor's own page_size, every
        # tensor's start address IS aligned to its own block_size_bytes — but
        # the cumulative-sum U33 computation did NOT mirror this and gave
        # un-aligned offsets (e.g. W1@696320 not multiple of 13824).  U34
        # replays the producer logic in Python to get correct, aligned
        # offsets.
        use_u34 = (
            os.environ.get("SGLANG_TT_U34_LCM_ALIGN", "0") == "1"
            and use_u33
        )
        gcb_size = self.max_tensor_block_size if self.max_tensor_block_size > 0 else 1
        if use_u34:
            # Recover per-tensor page_size from the registered consumer bytes:
            # consumer_bytes[t] = num_blocks * page_size[t]; page_size[t] =
            # consumer_bytes[t] / num_blocks.  num_blocks = ring_size.
            try:
                num_blocks = int(self.ring_size)
            except Exception:
                num_blocks = 32
            if num_blocks <= 0:
                num_blocks = 32
            _u34_trace = []
            # Simulate producer offsets for tensors 0..per_layer_idx.
            # Returns the wr_ptr at the START of tensor `per_layer_idx`.
            wr_ptr = 0
            for t in range(per_layer_idx + 1):
                cb_t = (
                    int(self._tensor_consumer_bytes[t])
                    if t < len(self._tensor_consumer_bytes)
                    else 0
                )
                ps = cb_t // num_blocks if cb_t > 0 else 0
                if ps <= 0:
                    # Tensor t hasn't been registered yet — fall back to U33
                    # cumulative (best-effort during iter-1).
                    continue
                cb_size_pa = gcb_size - (gcb_size % ps)
                fifo_limit_pa = cb_size_pa if cb_size_pa > 0 else gcb_size
                # resize: align wr_ptr up to ps; wrap if past fifo_limit_pa
                if (wr_ptr % ps) != 0:
                    wr_ptr = ((wr_ptr + ps - 1) // ps) * ps
                if wr_ptr >= fifo_limit_pa:
                    wr_ptr = 0
                if t == per_layer_idx:
                    # This is the tensor we're returning the offset for.
                    # Log only once per (per_layer_idx, offset) tuple to avoid spam.
                    if not hasattr(self, "_u34_logged"):
                        self._u34_logged = {}
                    _key = (per_layer_idx, wr_ptr)
                    if _key not in self._u34_logged:
                        self._u34_logged[_key] = True
                        logger.info(
                            f"[U34_RETURN] per_layer_idx={per_layer_idx} offset={wr_ptr} ps={ps} aligned={(wr_ptr%ps==0)}"
                        )
                    return wr_ptr
                # Otherwise, simulate writing num_blocks blocks of ps each.
                # Each block-write increments dest_addr by ps; if dest_addr ==
                # fifo_limit_pa, wrap to 0 (remote_circular_buffer.h:378).
                # If dest_addr overruns fifo_limit_pa mid-block (can happen
                # when ps doesn't divide fifo_limit_pa evenly — but resize
                # re-aligns so this rarely occurs), we wrap modulo gcb_size.
                for _ in range(num_blocks):
                    wr_ptr += ps
                    if wr_ptr == fifo_limit_pa:
                        wr_ptr = 0
                    elif wr_ptr > fifo_limit_pa:
                        wr_ptr = wr_ptr - fifo_limit_pa
            return wr_ptr
        if use_u33:
            # Sum consumer bytes for all tensors BEFORE this one in the
            # per-layer order.  Each tensor's matmul reads num_blocks *
            # in1_block_size_bytes per dispatch; producer wrote that many
            # bytes per tensor per receiver in the GCB; offsets are
            # cumulative until GCB-wrap.
            offset = 0
            for i in range(per_layer_idx):
                if i < len(self._tensor_consumer_bytes):
                    offset += int(self._tensor_consumer_bytes[i])
            return offset % gcb_size

        if self.num_tensors > 0:
            offset = self._tensor_byte_offsets[idx % self.num_tensors]
        else:
            offset = self._tensor_byte_offsets[idx]
        # Wrap to fit within the GlobalCB region.
        return offset % gcb_size

    def set_tensor_consumer_bytes(self, tensor: ttnn.Tensor, consumer_bytes_per_dispatch: int) -> None:
        """U33 — register per-tensor consumer bytes per dispatch.

        The matmul that consumes `tensor` reads
        `num_blocks * in1_block_size_bytes` bytes per dispatch where
        `in1_block_size_bytes = per_core_N * in0_block_w * tile_bytes`.
        Once every tensor's consumer bytes are registered,
        `get_tensor_gcb_offset_bytes` switches to consumer-side cumulative.

        Idempotent (overwrites previously set value).  No-op if tensor is
        not in `prefetched_tensors`.

        Use buffer_address as the lookup key (ttnn.Tensor equality may be
        unreliable across slicing/permutation; buffer address is the actual
        DRAM placement that the prefetcher actually reads).
        """
        if not hasattr(self, "_tensor_consumer_bytes"):
            self._tensor_consumer_bytes = []
        # Lookup by Python object identity first; fall back to buffer
        # address comparison if identity fails.  Note: buffer_address()
        # may surface previously-queued async errors — caller must handle.
        idx = None
        for i, t in enumerate(self.prefetched_tensors):
            if t is tensor:
                idx = i
                break
        if idx is None:
            try:
                tgt_addr = tensor.buffer_address()
            except Exception:
                tgt_addr = None
            if tgt_addr is not None:
                for i, addr in enumerate(self.prefetched_tensor_addr):
                    if addr == tgt_addr:
                        idx = i
                        break
        if idx is None:
            return
        per_layer_idx = (idx % self.num_tensors) if self.num_tensors > 0 else idx
        # Grow up to num_tensors slots (per-layer; offsets are the same across layers).
        n_slots = self.num_tensors if self.num_tensors > 0 else (per_layer_idx + 1)
        while len(self._tensor_consumer_bytes) < n_slots:
            self._tensor_consumer_bytes.append(0)
        was = self._tensor_consumer_bytes[per_layer_idx]
        self._tensor_consumer_bytes[per_layer_idx] = int(consumer_bytes_per_dispatch)
        if was != int(consumer_bytes_per_dispatch):
            gcb_size = self.max_tensor_block_size if self.max_tensor_block_size > 0 else 1
            cum = 0
            offs = []
            for i in range(n_slots):
                offs.append(cum % gcb_size)
                cum += int(self._tensor_consumer_bytes[i])
            # U34 — also log the producer-simulated (resize-aligned) offsets
            # so we can verify each tensor's offset is a multiple of its own
            # in1_block_size_bytes (page_size).
            try:
                num_blocks = int(self.ring_size)
            except Exception:
                num_blocks = 32
            if num_blocks <= 0:
                num_blocks = 32
            u34_offs = []
            wr_ptr = 0
            for i in range(n_slots):
                cb_i = int(self._tensor_consumer_bytes[i])
                ps = cb_i // num_blocks if cb_i > 0 else 0
                if ps <= 0:
                    u34_offs.append(None)
                    continue
                cb_size_pa = gcb_size - (gcb_size % ps)
                fifo_limit_pa = cb_size_pa if cb_size_pa > 0 else gcb_size
                if (wr_ptr % ps) != 0:
                    wr_ptr = ((wr_ptr + ps - 1) // ps) * ps
                if wr_ptr >= fifo_limit_pa:
                    wr_ptr = 0
                u34_offs.append((wr_ptr, ps, wr_ptr % ps))
                for _ in range(num_blocks):
                    wr_ptr += ps
                    if wr_ptr == fifo_limit_pa:
                        wr_ptr = 0
                    elif wr_ptr > fifo_limit_pa:
                        wr_ptr = wr_ptr - fifo_limit_pa
            logger.info(
                f"[U33] set_tensor_consumer_bytes per_layer_idx={per_layer_idx} "
                f"bytes={consumer_bytes_per_dispatch} (was={was}) "
                f"gcb_size={gcb_size} per_tensor_bytes={self._tensor_consumer_bytes} "
                f"cumulative_offsets={offs} u34_producer_sim_offsets={u34_offs}"
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
        # B.7 control test: when ALL SGLANG_TT_PREFETCHER_SKIP_* flags are set,
        # num_tensors==0 and the prefetcher has nothing to do. Skip both
        # GlobalCB creation and the dram_prefetcher op so callers' downstream
        # `stop()` is also a no-op (guarded below). All matmuls in this mode
        # use the `prefetch=False, num_global_cb_receivers=1` ring config
        # (SKIP_* fallback path), so neither global_cb nor sub_device_id is
        # actually read by the matmul kernels.
        if self.num_tensors == 0:
            if not getattr(self, "_zero_tensor_logged", False):
                logger.warning(
                    "[Prefetcher] run(): num_tensors==0 (all SKIP_* flags set); "
                    "skipping GlobalCB creation and dram_prefetcher op (B.7 control)"
                )
                self._zero_tensor_logged = True
            return
        # Create global cb buffer if it was not yet created.
        if self.global_cb is None:
            self.global_cb_size = self.max_tensor_block_size
            logger.info(f"[DRAM Prefetcher] Creating global CB with size: {self.global_cb_size}")
            # U6 — Galaxy-parity: augment sender_receiver_mapping with
            # (dummy_sender, dummy_receiver_set) pairs so the GlobalCB's
            # `all_cores()` covers the matmul factory's bbox-rounded
            # `all_cores` (line 2038 of matmul_multicore_reuse_mcast_1d_program_factory.cpp).
            # Dummy senders occupy otherwise-unused rows in the sender
            # columns; dummy receiver groups partition the per-rect bbox
            # gaps. Galaxy uses the exact same trick.
            _mapping = self.sender_receiver_mapping
            if (
                self._real_2subdev
                and getattr(self, "_dummy_receiver_coords", None)
            ):
                _dummy_mapping = self._build_dummy_sender_receiver_pairs()
                if _dummy_mapping:
                    _mapping = list(self.sender_receiver_mapping) + _dummy_mapping
                    logger.info(
                        f"[Prefetcher] U6 GlobalCB augmented with "
                        f"{len(_dummy_mapping)} dummy senders covering "
                        f"{sum(s.num_cores() for _, s in _dummy_mapping)} "
                        f"dummy receiver cores"
                    )
            self.global_cb = ttnn.create_global_circular_buffer(
                self.mesh_device,
                _mapping,
                self.global_cb_size,
            )

        # Create address tensor if it was not created yet
        if self.prefetched_tt_addr_tensor is None:
            self.prefetched_tt_addr_tensor = self.create_address_tensor()

        # Run prefetcher op (prefetcher op will start asynchronously prefetching weights until prefetcher.stop() is called)
        # ATTACK 1 layer-bisect: when SGLANG_TT_PREFETCHER_LAYERS=N limits
        # the consumer to the first N layers, the producer must match — it
        # writes num_tensors_per_layer * effective_layers pages to GlobalCB,
        # and consumers in layers >= N skip GlobalCB entirely (their matmul
        # reads weights directly from DRAM with no GlobalCB kwarg). The
        # producer's tensor_addrs buffer is layer-major (layer 0 tensors
        # first, layer 1 next, ...), so truncating num_layers just stops the
        # producer earlier in the address list. The 5 tensor handles in the
        # input list are still valid — they only carry shape/dtype/shard-spec
        # metadata; the per-layer address comes from tensor_addrs.
        _lim = self.layer_limit
        if isinstance(_lim, int) and _lim > 0:
            effective_num_layers = min(self.num_layers, _lim)
        else:
            effective_num_layers = self.num_layers
        if not getattr(self, "_layer_limit_logged", False):
            logger.warning(
                f"[Prefetcher] run(): num_layers={self.num_layers} "
                f"effective_num_layers={effective_num_layers} "
                f"(SGLANG_TT_PREFETCHER_LAYERS={getattr(self, '_env_layer_limit', 0)})"
            )
            self._layer_limit_logged = True
        self.garbage = ttnn.dram_prefetcher(
            self.prefetched_tensors[: self.num_tensors] + [self.prefetched_tt_addr_tensor],
            num_layers=effective_num_layers,
            global_cb=self.global_cb,
            enable_performance_mode=self.enable_performance_mode,
        )
        # Set worker sub device stall group
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device.sub_devices_id[-1]])
        return

    def _build_dummy_sender_receiver_pairs(self):
        """
        U6 — Build (dummy_sender CoreCoord, dummy_receivers CoreRangeSet) pairs
        to augment the GlobalCB so its `all_cores()` covers the matmul factory
        bbox.

        Dummy senders occupy otherwise-unused rows in the active sender
        columns (cols 0 and 7 on Blackhole). They are NOT registered in the
        active sender mapping → the dram_prefetcher op never assigns them
        work, but the GlobalCB metadata writer treats them as zero-page
        senders that simply hold CB config buffer entries.

        Dummy receivers are partitioned into per-sender groups so the
        zip(dummy_senders, dummy_receiver_groups) gives a valid mapping
        with no duplicate cores (CB invariant at
        global_circular_buffer.cpp:56).
        """
        if not getattr(self, "_dummy_receiver_coords", None):
            return []
        # Real sender cores (in active mapping); never use these as dummies.
        real_senders_set = {(s.x, s.y) for s in self.sender_cores(active=True)}
        # Enumerate dummy sender candidates: unused rows in cols 0 and 7,
        # clamped to the MUX grid (rows 0-7). These cores are NOT receivers
        # (we constructed the mapping so receivers live in cols 1-4 ∪ 8-11
        # under SGLANG_TT_PREFETCHER_REAL_2SUBDEV=1).
        candidates = []
        _max_y = max(
            (r.end.y for r in self.all_core_range_set.ranges()),
            default=7,
        )
        for col in (0, 7):
            for row in range(_max_y + 1):
                if (col, row) in real_senders_set:
                    continue
                # Sanity: must also not be a real receiver. Skip if so.
                # (Should never trip with the 2-subdev mapping; defensive.)
                _is_real_receiver = False
                for r_set in self.receiver_cores(sender_active=None, receiver_active=True):
                    for cr in r_set.ranges():
                        if cr.start.x <= col <= cr.end.x and cr.start.y <= row <= cr.end.y:
                            _is_real_receiver = True
                            break
                    if _is_real_receiver:
                        break
                if _is_real_receiver:
                    continue
                candidates.append(ttnn.CoreCoord(col, row))
        if not candidates:
            logger.warning(
                "[Prefetcher] U6 _build_dummy_sender_receiver_pairs: "
                "no dummy sender candidates available"
            )
            return []
        # Partition dummy receivers across candidates round-robin (1:N).
        dummy_groups = [[] for _ in candidates]
        for i, (x, y) in enumerate(self._dummy_receiver_coords):
            dummy_groups[i % len(candidates)].append(ttnn.CoreCoord(x, y))
        pairs = []
        for sender, recvs in zip(candidates, dummy_groups):
            if not recvs:
                continue
            recv_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(c, c) for c in recvs]
            )
            pairs.append((sender, recv_set))
        return pairs

    def stop(self):
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot stop prefetcher"
        # B.7 control: num_tensors==0 means run() was a no-op; nothing to deallocate.
        if self.num_tensors == 0:
            return
        assert self.garbage is not None, "Prefetcher has not been run. Cannot stop prefetcher"
        ttnn.deallocate(self.garbage)
        self.garbage = None
        return
