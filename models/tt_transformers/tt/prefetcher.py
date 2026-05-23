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
                sender_set = self.to_core_range_set(self.sender_cores(active=True))
                self.prefetcher_sub_device.add_sub_device(sender_set)
                # 3-sub-device layout: separate receivers from workers so the
                # prefetcher CB on receiver cores doesn't clash with model op
                # L1 allocations (RMSNorm etc.) on compute cores.
                if self.receiver_mapping_override:
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

    def stop(self):
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot stop prefetcher"
        assert self.garbage is not None, "Prefetcher has not been run. Cannot stop prefetcher"
        ttnn.deallocate(self.garbage)
        self.garbage = None
        return
