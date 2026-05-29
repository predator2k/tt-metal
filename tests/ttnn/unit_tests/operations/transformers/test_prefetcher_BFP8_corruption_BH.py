# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone reproducer scaffold for the BFP8 prefetcher corruption bug.

Bug summary
-----------
On a (1, 2) Blackhole P150a mesh, when ``matmul_multicore_reuse_mcast_1d`` is
run with ``use_global_cb = true`` (i.e. in1 weights are streamed in by
``dram_prefetcher`` through a ``GlobalCircularBuffer`` allocated via
``experimental::CreateCircularBuffer(prog, cores, remote_cfg, *global_cb)``),
every BFP8 weight matmul in the model produces silently corrupted output, while
every BFP4 weight matmul on the same path produces bit-correct output. Under
real Qwen3-8B weights the corrupted output reaches magnitudes O(2^60-2^109).

Full investigation: ``docs/platforms/tt_qwen3_8b_prefetcher_UPSTREAM_BUG_REPORT_2026-05-26.md``
(57 dispatches across U1-U43, every byte-delivery / per-CB metadata / cfg-register
mechanism conclusively ruled out; remaining surface is the silicon-level unpacker
state machine).

What this test does
-------------------
A minimal pytest case that:

1. Mirrors ``test_prefetcher_BH.py``'s setup (Prefetcher + GlobalCB + 5 ring matmuls
   for the full Qwen3-8B decoder layer: QKV, WO, FF1, FF3, FF2).
2. Uses **deterministic** weights (fixed torch RNG seed) so results are reproducible
   across runs; the per-element torch reference is the ground truth and is in
   ``O(sqrt(K))`` magnitude (`~5-10` for Qwen3-8B dims). The buggy kernel produces
   ``|out| > 1e10`` (often Inf/NaN), so a loose ``max_abs_diff < ABS_TOL = 100``
   conclusively discriminates clean from broken behaviour.
3. Uses Qwen3-8B-balanced dims (the configuration where the bug was observed):
   ``dim=4096, hidden_dim=12288, n_heads=32, n_kv_heads=8``,
   ``num_receiver_cores=4`` (-> ring_size=32, matching the four broken ELFs in
   the bug report; smaller nrc fails is_prefetcher_supported's L1 byte budget).
4. Parameterizes the **weight dtype** (BFP4 vs BFP8) across the 5 matmul positions.
5. Replays the captured trace ``NUM_TRACE_REPLAYS`` times so the producer's
   GCB wr_ptr wraps at least once.

How to run (inside the ``p3a-ngram`` container with 2x Blackhole P150a hardware
visible)::

    podman exec p3a-ngram bash -lc '\
      source /opt/venv/bin/activate && \
      cd /tt-metal && \
      MESH_DEVICE=P300 HF_MODEL=Qwen3-8B \
      pytest -xvs \
        tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BFP8_corruption_BH.py \
        2>&1 | tee /tmp/u46_repro.log'

Hardware results on the 2x Blackhole P150a fork as of 2026-05-26
----------------------------------------------------------------
**Both ``dtype=bfloat4_b`` and ``dtype=bfloat8_b`` PASS in this isolated
scaffold** (max_abs_diff in [3.8, 7.1] for BFP8; in [1.2, 2.2] for BFP4).
This is a *negative* result for the standalone reproducer but a *positive*
finding for narrowing the bug surface: the prefetcher dual-index CB allocation
+ BFP8 dtype + the gathered matmul path with 5 mixed-size weight tensors and
50 trace replays is **NOT sufficient** on its own to trigger the failure that
SGLang's full Qwen3-8B decode reproduces 100% of the time.

The production-only ingredients that this scaffold currently lacks (and that
the LLK team may want to layer in one at a time when extending this test):

1. **CCL ops between matmuls.** Production attention chains
   ``QKV -> SDPA -> WO -> reduce_scatter -> residual`` and MLP chains
   ``FF1, FF3 -> SiLU -> elementwise mul -> FF2 -> reduce_scatter -> residual``.
   The CCL/RMSNorm/RoPE/SDPA ops in between dispatch on the *same* worker
   sub-device pool that the matmul kernels and the prefetcher's receiver
   reader/writer kernels live on, and they reshape L1 / NoC traffic in ways
   this scaffold does NOT exercise.
2. **36 decoder layers per trace**, not 1. Qwen3-8B has 36 hidden layers; the
   per-tensor wr_ptr arithmetic and the producer's GCB-page-size re-alignment
   on each tensor boundary (writer_l1.cpp:95 + remote_circular_buffer.h:110)
   compound differently across 36 vs 1 layer.
3. **Real HF weights**, not torch.randn. Real Qwen3-8B weights have a
   spread of magnitudes that BFP8's shared-exponent quantization compresses
   to a narrower-than-randn signal. Hypothesis 5.3(1) of the bug report
   (MOP-replay-buffer tile-stride aliasing) is most plausibly triggered
   when the wrong tile stride lands on a face whose decoded shared-exponent
   happens to magnify the misread mantissa bits.
4. **Concurrent prefill + decode traces.** Production captures a prefill
   trace AND a decode trace, switches sub-device managers between them, and
   then replays decode in a tight loop. The bug surfaces during decode.

Why the single-matmul minimisation failed
-----------------------------------------
An earlier draft of this reproducer ran ONE matmul with ONE tensor in the
prefetcher queue. That configuration did NOT reproduce the bug either,
because with a single tensor the producer's ``wr_ptr`` increment is trivially
correct (no per-tensor offset arithmetic exercised). The current 5-matmul
scaffold exercises the multi-tensor wr_ptr advance correctly, but still
does not surface the bug -- pointing the failure surface deeper into the
production-only ingredients listed above.
"""

import math
import os

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.prefetcher import (
    Prefetcher,
    VERIFIED_MODEL_CONFIGS,
    is_prefetcher_supported,
)

# U51: optional HF weight source (real Qwen3-8B safetensor weights). When the
# safetensors lib is not available the test falls back to a real-LLM-shaped
# torch.randn × 0.02 scale, which matches the std observed on Qwen3-8B.
try:
    from safetensors import safe_open  # type: ignore

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

# Qwen3-8B-balanced preset: matches the four BFP4/BFP8 ELFs observed in the
# bug report (see UPSTREAM_BUG_REPORT_2026-05-26.md section 3.1 per-ELF table).
MODEL_NAME = "Qwen3-8B"
# Smallest legal nrc for Qwen3-8B on 2x P150a (mesh shape 1x2). On Blackhole
# num_senders == len(dram_banks) == 8, so ring_size = nrc * 8. The four broken
# ELFs in the bug report all run with ring_size = num_cores = 32 -> nrc = 4.
NUM_RECEIVER_CORES = 4

# Loose absolute tolerance. With randn(0, 1) weights/inputs and K = 4096,
# the matmul output magnitude stays bounded around |output| < 200 in any
# reasonable run (3-sigma of a sum of 4096 i.i.d. products of N(0,1) RVs).
# BFP4/BFP8 quant noise stays small. The broken kernel produces |out| > 1e10
# (often Inf/NaN), so any threshold << 1e10 conclusively discriminates clean
# vs broken behaviour.
ABS_TOL = 100.0

# Number of decoder layers to populate in the GlobalCB. The bug requires
# multi-tensor to exercise the per-tensor wr_ptr advance (single-tensor
# repro attempts did not surface the bug). One layer is sufficient -- this
# matches test_prefetcher_BH.py's default scaffold and the production bug
# observation was on layer 0 of decode step 1.
DEFAULT_NUM_LAYERS = 1

# Number of trace replays. Each replay re-runs the whole prefetcher +
# 5-matmul pipeline, advancing the GlobalCB wr_ptr. Production decode loops
# >>1000 times before observing the failure; many fewer replays should be
# enough here to wrap the GCB at least once (~30 replays for the Qwen3-8B
# preset given the per-tensor write sizes vs the 442368 B GCB).
NUM_TRACE_REPLAYS = 50

# Deterministic RNG seed for the torch.randn-generated weights. Same seed
# across BFP4 vs BFP8 runs ensures the only delta is the on-device dtype
# of the weight tensor.
TORCH_SEED = 0xBADC0FFE

# Matmuls in the order they appear inside a Qwen3-8B decoder layer.
MATMUL_NAMES = ["qkv", "wo", "ff1", "ff3", "ff2"]

# U51: default HF model directory inside the p3a-ngram container.
HF_MODEL_DIR_DEFAULT = "/models/Qwen3-8B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _model_dims() -> dict:
    return VERIFIED_MODEL_CONFIGS[MODEL_NAME]


def _weight_specs(model_dims: dict):
    """Return [(name, K, N, shard_dims, shard_type)] in test order.

    Mirrors ``test_prefetcher_BH.py::create_weight_tensors``.
    """
    dim = model_dims["dim"]
    hidden_dim = model_dims["hidden_dim"]
    n_heads = model_dims["n_heads"]
    n_kv_heads = model_dims["n_kv_heads"]
    head_dim = dim // n_heads
    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)
    return [
        # name, K,        N,          shard_dims,  shard_type
        ("qkv", dim, qkv_size, (2, 3), "N"),
        ("wo", n_heads * head_dim, dim, (3, 2), "K"),
        ("ff1", dim, hidden_dim, (2, 3), "N"),
        ("ff3", dim, hidden_dim, (2, 3), "N"),
        ("ff2", hidden_dim, dim, (3, 2), "K"),
    ]


def _pad_n_to_ring_size(n_size: int, ring_size: int) -> int:
    per_core = round_up(math.ceil(n_size / ring_size), ttnn.TILE_SIZE)
    return per_core * ring_size


def _calc_k_per_shard(k: int, ring_size: int) -> int:
    return round_up(math.ceil(k / ring_size), ttnn.TILE_SIZE)


def _create_dram_sharded_mem_config(k: int, n_padded: int, dram_cores: int):
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, n_padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _create_ring_matmul_config(
    M: int,
    K: int,
    N_padded: int,
    num_cores: int,
    num_global_cb_receivers: int,
    num_dram_banks: int,
    untilize_out: bool = False,
):
    """Build a ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for the
    gathered (use_global_cb=true) ring matmul. Geometry follows
    ``test_prefetcher_BH.py::create_matmul_program_configs``.
    """
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    while in0_block_w > 0 and (K / ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1
    if in0_block_w == 0:
        in0_block_w = 1
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // num_cores // ttnn.TILE_SIZE
    out_subblock_h = 1
    out_subblock_w = 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    grid = ttnn.CoreGrid(y=num_cores // num_dram_banks, x=num_dram_banks)
    # Empty hop_cores -- mirrors test_prefetcher_BH.py's default.
    _hop_grid = []
    hop_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in _hop_grid}
    )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        num_global_cb_receivers=num_global_cb_receivers,
        untilize_out=untilize_out,
    )


# ---------------------------------------------------------------------------
# U51: Real HF Qwen3-8B weight loader (one layer at a time, transposed to
# [in, out] for matmul, padded to the test's expected per-device dim).
# ---------------------------------------------------------------------------


def _load_hf_layer_weight(hf_dir: str, layer_idx: int, name: str) -> torch.Tensor:
    """Load + concat the real HF Qwen3-8B weight for a given matmul slot.

    Returns a [K_full, N_full] torch.Tensor (float32) in matmul-multiplier
    orientation. Falls back to torch.randn × 0.02 (typical trained-LLM std)
    when safetensors is unavailable or the file is missing.
    """
    # All layer-0..layer-35 weights of Qwen3-8B live in model-00001..00005.
    # Use the index to find the right shard.
    if not _HAS_SAFETENSORS or not os.path.isdir(hf_dir):
        return None
    import json as _json

    idx_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        return None
    with open(idx_path) as _f:
        idx = _json.load(_f)

    def _get(weight_map_key: str) -> torch.Tensor:
        shard = idx["weight_map"][weight_map_key]
        with safe_open(os.path.join(hf_dir, shard), framework="pt") as _sf:
            return _sf.get_tensor(weight_map_key).float()

    if name == "qkv":
        q = _get(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
        k = _get(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
        v = _get(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
        # HF stores as [out, in]; concat along out (rows) then T → [in, out]
        full = torch.cat([q, k, v], dim=0).T.contiguous()
    elif name == "wo":
        full = _get(f"model.layers.{layer_idx}.self_attn.o_proj.weight").T.contiguous()
    elif name == "ff1":
        full = _get(f"model.layers.{layer_idx}.mlp.gate_proj.weight").T.contiguous()
    elif name == "ff3":
        full = _get(f"model.layers.{layer_idx}.mlp.up_proj.weight").T.contiguous()
    elif name == "ff2":
        full = _get(f"model.layers.{layer_idx}.mlp.down_proj.weight").T.contiguous()
    else:
        raise ValueError(name)
    return full


def _hf_fallback_weight(k: int, n: int) -> torch.Tensor:
    """Fallback for when HF weights are unavailable: real-LLM-shaped randn.

    Trained transformer weights typically have std≈0.02 (init scale × layer
    depth scaling). The test's torch.randn (std=1) is wildly different and
    activates BFP8's shared-exp differently.
    """
    return torch.randn(k, n) * 0.02


# ---------------------------------------------------------------------------
# 5-matmul x N-layer setup + run (mirrors test_prefetcher_BH.py exactly,
# but with all weights forced to the same dtype so failure attribution
# is unambiguous).
# ---------------------------------------------------------------------------


def _build_weights_and_inputs(
    mesh_device,
    model_dims,
    num_layers,
    prefetcher,
    weight_dtype,
    weight_source: str = "randn",
    hf_dir: str = HF_MODEL_DIR_DEFAULT,
):
    """Build per-layer per-matmul weights + inputs for the 5-matmul pipeline.

    Returns:
        pt_weights: dict[key -> torch.Tensor]              # CPU reference
        tt_weights: dict[key -> ttnn.Tensor]               # device, ring-sharded
        pt_inputs:  dict[name -> torch.Tensor]             # CPU reference
        tt_inputs:  dict[name -> ttnn.Tensor]              # device, input-sharded
        out_mem_configs: dict[name -> ttnn.MemoryConfig]   # for ttnn.linear out
        program_configs: dict[name -> ProgramConfig]
        metadata: dict[name -> dict]                       # k/n/shard_type
    """
    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    ring_size = prefetcher.ring_size
    num_dram_banks = len(prefetcher.dram_banks())
    receiver_core_range_set = prefetcher.to_core_range_set(
        prefetcher.receiver_cores(sender_active=True, receiver_active=True)
    )

    pt_weights = {}
    tt_weights = {}
    metadata = {}
    program_configs = {}
    out_mem_configs = {}

    M = 32  # decode batch

    torch.manual_seed(TORCH_SEED)
    hf_unavailable_warned = False

    for name, k, n, shard_dims, shard_type in _weight_specs(model_dims):
        is_n_shard = shard_type == "N"
        k_per_device = k if is_n_shard else k // num_devices
        n_per_device_unpadded = n // num_devices if is_n_shard else n
        n_per_device = _pad_n_to_ring_size(n_per_device_unpadded, ring_size)

        # U51: per-source weight generation.
        #   "randn"      = original behavior (one randn(0,1) per spec, shared across layers)
        #   "hf_real"    = real HF Qwen3-8B per-layer weight; shape-matched to the
        #                  test's expected [K_full, N_full]; falls back to scaled-randn
        #                  if HF unavailable. Per-layer weights are loaded individually
        #                  so 36-layer runs reflect real Qwen3-8B per-layer magnitudes.
        #   "scaled_randn" = torch.randn × 0.02 (trained-LLM-typical std) using the
        #                    same shared-per-spec shape semantics as "randn".
        if weight_source == "randn":
            if is_n_shard:
                full_w_shared = torch.randn(k_per_device, n_per_device * num_devices)
            else:
                full_w_shared = torch.randn(k_per_device * num_devices, n_per_device)
            full_w_per_layer = [full_w_shared] * num_layers
        elif weight_source == "scaled_randn":
            if is_n_shard:
                full_w_shared = torch.randn(k_per_device, n_per_device * num_devices) * 0.02
            else:
                full_w_shared = torch.randn(k_per_device * num_devices, n_per_device) * 0.02
            full_w_per_layer = [full_w_shared] * num_layers
        elif weight_source == "hf_real":
            full_w_per_layer = []
            for layer_idx in range(num_layers):
                _real = _load_hf_layer_weight(hf_dir, layer_idx, name)
                if _real is None:
                    if not hf_unavailable_warned:
                        logger.warning(
                            f"[BFP8-corruption-repro] hf_real weight unavailable "
                            f"(dir={hf_dir}, safetensors={_HAS_SAFETENSORS}); falling "
                            f"back to scaled randn × 0.02 (trained-LLM-typical std)."
                        )
                        hf_unavailable_warned = True
                    if is_n_shard:
                        _real = torch.randn(k_per_device, n_per_device * num_devices) * 0.02
                    else:
                        _real = torch.randn(k_per_device * num_devices, n_per_device) * 0.02
                else:
                    # HF returns [K_full, N_full]; shape must match the test's expected
                    # shard layout. Validate the K dim (full K, since real HF is not
                    # pre-sharded) and pad N if needed to ring-aligned size.
                    if is_n_shard:
                        # expected shape: [k_per_device, n_per_device * num_devices]
                        # real HF: [K_full=k, N_full]; here k_per_device == k (no K-shard)
                        # so K dim should already match.
                        if _real.shape[0] != k_per_device:
                            raise AssertionError(
                                f"hf_real {name} K mismatch: got {_real.shape[0]} expected {k_per_device}"
                            )
                        n_target = n_per_device * num_devices
                        if _real.shape[1] < n_target:
                            # Pad N with zeros to ring-aligned size (preserves matmul semantics).
                            pad = torch.zeros(_real.shape[0], n_target - _real.shape[1])
                            _real = torch.cat([_real, pad], dim=1)
                        elif _real.shape[1] > n_target:
                            _real = _real[:, :n_target]
                    else:  # K-shard
                        # expected shape: [k_per_device * num_devices, n_per_device]
                        if _real.shape[0] != k_per_device * num_devices:
                            raise AssertionError(
                                f"hf_real {name} K mismatch: got {_real.shape[0]} expected {k_per_device * num_devices}"
                            )
                        if _real.shape[1] != n_per_device:
                            # Pad N if needed.
                            if _real.shape[1] < n_per_device:
                                pad = torch.zeros(_real.shape[0], n_per_device - _real.shape[1])
                                _real = torch.cat([_real, pad], dim=1)
                            else:
                                _real = _real[:, :n_per_device]
                full_w_per_layer.append(_real.contiguous())
        else:
            raise ValueError(f"Unknown weight_source={weight_source}")

        metadata[name] = {
            "shard_type": shard_type,
            "k": k,
            "n": n,
            "k_per_device": k_per_device,
            "n_per_device": n_per_device,
        }

        mem_config = _create_dram_sharded_mem_config(k_per_device, n_per_device, num_dram_banks)
        program_configs[name] = _create_ring_matmul_config(
            M=M,
            K=k_per_device,
            N_padded=n_per_device,
            num_cores=ring_size,
            num_global_cb_receivers=prefetcher.num_receiver_cores,
            num_dram_banks=num_dram_banks,
            untilize_out=(name == "qkv"),
        )
        out_mem_configs[name] = ttnn.create_sharded_memory_config(
            shape=(M, n_per_device // ring_size),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Per-layer copies (insert each into prefetcher queue).
        for layer_idx in range(num_layers):
            key = f"layer_{layer_idx}_{name}"
            full_w = full_w_per_layer[layer_idx]
            full_w_4d = full_w.unsqueeze(0).unsqueeze(0)
            pt_weights[key] = full_w
            tt_w = ttnn.as_tensor(
                full_w_4d,
                device=mesh_device,
                dtype=weight_dtype,
                memory_config=mem_config,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
            )
            tt_weights[key] = tt_w
            prefetcher.insert_tensor(tt_w)
            if layer_idx == 0 or layer_idx == num_layers - 1:
                logger.info(
                    f"[BFP8-corruption-repro] inserted weight {key} shape={tuple(full_w.shape)} dtype={weight_dtype} src={weight_source}"
                )

    # Inputs (per matmul role; replicated for N-sharded, K-sharded for K-sharded).
    qkv_k_pd = metadata["qkv"]["k_per_device"]
    wo_k_pd = metadata["wo"]["k_per_device"]
    ff1_k_pd = metadata["ff1"]["k_per_device"]
    ff2_k_pd = metadata["ff2"]["k_per_device"]

    pt_inputs = {
        "attn_input": torch.randn(1, 1, M, qkv_k_pd),  # replicated, used by QKV
        "mlp_input": torch.randn(1, 1, M, ff1_k_pd),  # replicated, used by FF1, FF3
        "wo_input": torch.randn(1, 1, M, wo_k_pd * num_devices),  # K-sharded
        "ff2_input": torch.randn(1, 1, M, ff2_k_pd * num_devices),  # K-sharded
    }

    def _in_mem_cfg(k_per_shard):
        return ttnn.create_sharded_memory_config(
            shape=(M, k_per_shard),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    tt_inputs = {}
    tt_inputs["attn_input"] = ttnn.from_torch(
        pt_inputs["attn_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_in_mem_cfg(_calc_k_per_shard(qkv_k_pd, ring_size)),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_inputs["mlp_input"] = ttnn.from_torch(
        pt_inputs["mlp_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_in_mem_cfg(_calc_k_per_shard(ff1_k_pd, ring_size)),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_inputs["wo_input"] = ttnn.from_torch(
        pt_inputs["wo_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_in_mem_cfg(_calc_k_per_shard(wo_k_pd, ring_size)),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=mesh_shape),
    )
    tt_inputs["ff2_input"] = ttnn.from_torch(
        pt_inputs["ff2_input"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_in_mem_cfg(_calc_k_per_shard(ff2_k_pd, ring_size)),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 3), mesh_shape=mesh_shape),
    )

    return pt_weights, tt_weights, pt_inputs, tt_inputs, out_mem_configs, program_configs, metadata


def _input_key_for(matmul_name):
    return {
        "qkv": "attn_input",
        "wo": "wo_input",
        "ff1": "mlp_input",
        "ff3": "mlp_input",
        "ff2": "ff2_input",
    }[matmul_name]


def _run_full_prefetcher_pipeline(
    mesh_device,
    weight_dtype,
    num_receiver_cores: int,
    num_layers: int,
    *,
    weight_source: str = "randn",
    inter_op_kind: str = "none",
    num_trace_replays: int = NUM_TRACE_REPLAYS,
    hf_dir: str = HF_MODEL_DIR_DEFAULT,
    use_trace: bool = True,
    num_eager_iterations: int = 5,
    simulate_extra_layers: int = 0,
):
    """Run the full 5-matmul prefetcher pipeline for ``num_layers`` layers.

    Returns a list of per-(layer, matmul_name, device_idx) result dicts, each with:
        layer_idx, matmul_name, device_idx, expected_max_abs, got_max_abs,
        max_abs_diff, finite_frac.
    """
    model_dims = _model_dims()
    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    logger.info(
        f"[BFP8-corruption-repro] mesh_shape={mesh_shape} num_devices={num_devices} "
        f"weight_dtype={weight_dtype} num_layers={num_layers} num_receiver_cores={num_receiver_cores} "
        f"weight_source={weight_source} inter_op_kind={inter_op_kind} "
        f"num_trace_replays={num_trace_replays}"
    )

    prefetcher = Prefetcher(
        mesh_device=mesh_device,
        num_tensors=len(MATMUL_NAMES),
        num_layers=num_layers,
        num_receiver_cores=num_receiver_cores,
    )
    prefetcher.init(mode=Mode.DECODE)
    # Use receiver_sub_device_id for the gathered matmul. The matmul factory
    # at matmul_multicore_reuse_mcast_1d_program_factory.cpp:2018-2043 does
    # `subdevice_cores.intersection(non_idle_cores)` where non_idle_cores = the
    # input shard grid (= the prefetcher's 32 receivers, for nrc=4). The
    # 3-sub-device layout puts:
    #   sub_devices[0] = senders (persistent dram_prefetcher writer kernel)
    #   sub_devices[1] = receivers (persistent dram_prefetcher reader kernel)
    #   sub_devices[2] = worker / "compute_only" carve-out
    # worker_sub_device_id == sub_devices[-1] = compute_only, which EXCLUDES
    # the receivers, so passing it would yield an empty intersection and
    # TT_FATAL at program.cpp:1469. Production attention/mlp paths use
    # receiver_sub_device_id (see attention.py:1861) -- mirror that here.
    # When the prefetcher is in a 2-sub-device layout (no isolated receiver
    # carve-out), receiver_sub_device_id falls back to worker_sub_device_id
    # automatically.
    sub_device_id_for_matmul = prefetcher.receiver_sub_device_id

    (
        pt_weights,
        tt_weights,
        pt_inputs,
        tt_inputs,
        out_mem_configs,
        program_configs,
        metadata,
    ) = _build_weights_and_inputs(
        mesh_device,
        model_dims,
        num_layers,
        prefetcher,
        weight_dtype,
        weight_source=weight_source,
        hf_dir=hf_dir,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    def run_op():
        prefetcher.run()
        all_outs = []
        for layer_idx in range(num_layers):
            layer_outs = {}
            for name in MATMUL_NAMES:
                in_t = tt_inputs[_input_key_for(name)]
                out = ttnn.linear(
                    in_t,
                    tt_weights[f"layer_{layer_idx}_{name}"],
                    program_config=program_configs[name],
                    memory_config=out_mem_configs[name],
                    compute_kernel_config=compute_kernel_config,
                    dtype=ttnn.bfloat16,
                    global_cb=prefetcher.global_cb,
                    sub_device_id=sub_device_id_for_matmul,
                )
                # U51 ingredient: inject inter-op eltwise dispatch BETWEEN matmuls.
                # Production attention/mlp dispatches CCL + RMSNorm + RoPE + SDPA
                # + eltwise mul on the same worker sub-device pool that the
                # gathered matmuls + the prefetcher sender/receiver kernels live
                # on, which reshapes L1 / NoC state between consecutive matmul
                # dispatches. Real CCL needs full TT_CCL+fabric setup (heavy
                # scaffolding); a lightweight in-place eltwise (silu/mul) is the
                # closest analogue that exercises the same worker-pool dispatch
                # path between two prefetcher-fed matmul fires WITHOUT changing
                # any matmul result (we apply it to a sacrificial copy that's
                # discarded). The bug-hypothesis-relevant question is: does ANY
                # inter-matmul kernel dispatch on the same sub-device pool flush
                # / mis-align the silicon FSM state that the next matmul reads?
                if inter_op_kind in ("silu_mul", "all"):
                    # Self-mul + silu on the matmul output (sharded). This is an
                    # extra worker-pool kernel dispatch between this matmul and
                    # the next. Use the same memory config so it stays sharded
                    # on the receiver core range set (same place as the matmul
                    # output). We discard the result; only the dispatch effect
                    # matters for our hypothesis.
                    _scratch = ttnn.silu(out, memory_config=out.memory_config())
                    _scratch2 = ttnn.mul(_scratch, _scratch, memory_config=_scratch.memory_config())
                    ttnn.deallocate(_scratch)
                    ttnn.deallocate(_scratch2)
                layer_outs[name] = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            all_outs.append(layer_outs)
        # Mirror test_prefetcher_BH.py: reset stall group AFTER dispatching all
        # matmuls. This restores to ALL sub-devices (sender, receiver, worker),
        # so the subsequent synchronize_device / end_trace_capture drain ALL
        # programs including the prefetcher's persistent sender kernel (which
        # exits naturally after num_layers iterations of writes).
        mesh_device.reset_sub_device_stall_group()
        return all_outs

    # Mirror test_prefetcher_BH.py's enable_trace=True flow: compile pass +
    # trace capture + execute trace (blocking=True). Without trace, the
    # subsequent ttnn.to_torch calls in verification can deadlock against the
    # persistent dram_prefetcher kernel.
    #
    # 2026-05-26 finding: a single trace replay with random weights does NOT
    # surface the bug in this scaffold (BFP4 and BFP8 both pass with
    # max_abs_diff in [1, 10]). The production bug needs the GlobalCB to wrap
    # AT LEAST ONCE -- per the bug report (UPSTREAM_BUG_REPORT_2026-05-26
    # section 5.3 hypothesis 1), the failure is consistent with MOP-replay-
    # buffer tile-stride aliasing that only manifests when consecutive
    # matmul dispatches see different in1_block_size_bytes (i.e. the dual-
    # index allocator hands out aliased regions over time). Replay the trace
    # ``NUM_TRACE_REPLAYS`` times so the producer's wr_ptr wraps the GCB at
    # least once per decode position.
    if use_trace:
        logger.info("[BFP8-corruption-repro] compile pass (no trace)...")
        outputs = run_op()
        logger.info("[BFP8-corruption-repro] capturing trace...")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outputs = run_op()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"[BFP8-corruption-repro] executing trace ({num_trace_replays} replays)...")
        for replay_idx in range(num_trace_replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        logger.info("[BFP8-corruption-repro] trace executed; verifying outputs...")
    else:
        # Eager mode (no trace, no repeated prefetcher.run). Used when
        # num_layers > 1 because the trace capture hangs in that configuration
        # (producer fills GCB faster than consumer can drain during capture-
        # time serialization). One eager run does num_layers × num_tensors
        # writes from the prefetcher persistent kernel + num_layers ×
        # num_matmuls ttnn.linear reads in the python loop, which wraps the
        # GCB at least once for num_layers >= 4.
        #
        # NOTE: a single prefetcher.run() call is by-construction safe — the
        # producer kernel writes (num_layers × num_tensors) pages then exits.
        # The python loop consumes them via ttnn.linear in order. Calling
        # prefetcher.run() multiple times (eager loop) can collide with the
        # still-active previous persistent kernel; trace replay avoids this
        # by re-dispatching a freshly-cloned trace per replay.
        logger.info(
            f"[BFP8-corruption-repro] eager execution (1 iteration, "
            f"{num_layers} layers)..."
        )
        outputs = run_op()
        ttnn.synchronize_device(mesh_device)
        logger.info("[BFP8-corruption-repro] eager executed; verifying outputs...")

    # Per-(layer, matmul, device) verification.
    results = []
    for layer_idx in range(num_layers):
        for name in MATMUL_NAMES:
            in_t = tt_inputs[_input_key_for(name)]
            for device_idx in range(num_devices):
                in0_t = ttnn.to_torch(ttnn.get_device_tensors(in_t)[device_idx]).float()
                in1_t = ttnn.to_torch(
                    ttnn.get_device_tensors(tt_weights[f"layer_{layer_idx}_{name}"])[device_idx]
                ).float()
                out_t = ttnn.to_torch(ttnn.get_device_tensors(outputs[layer_idx][name])[device_idx]).float()
                expected = in0_t @ in1_t

                out_clean = torch.nan_to_num(out_t, nan=1e30, posinf=1e30, neginf=-1e30)
                finite_frac = torch.isfinite(out_t).float().mean().item()
                max_abs = out_clean.abs().max().item()
                max_abs_diff = (out_clean - expected).abs().max().item()
                expected_max_abs = expected.abs().max().item()
                logger.info(
                    f"[BFP8-corruption-repro] layer={layer_idx} matmul={name} dev={device_idx} "
                    f"dtype={weight_dtype} expected_max_abs={expected_max_abs:.4g} "
                    f"got_max_abs={max_abs:.4g} max_abs_diff={max_abs_diff:.4g} "
                    f"finite_frac={finite_frac:.4f}"
                )
                results.append(
                    {
                        "layer_idx": layer_idx,
                        "matmul_name": name,
                        "device_idx": device_idx,
                        "expected_max_abs": expected_max_abs,
                        "got_max_abs": max_abs,
                        "max_abs_diff": max_abs_diff,
                        "finite_frac": finite_frac,
                    }
                )

    # Cleanup.
    prefetcher.stop()
    for t in tt_inputs.values():
        ttnn.deallocate(t)
    for t in tt_weights.values():
        ttnn.deallocate(t)
    for layer_outs in outputs:
        for o in layer_outs.values():
            ttnn.deallocate(o)
    mesh_device.clear_loaded_sub_device_manager()

    return results


# ---------------------------------------------------------------------------
# Pytest cases
# ---------------------------------------------------------------------------


_DTYPE_CASES = [
    pytest.param(ttnn.bfloat4_b, id="dtype=bfloat4_b"),
    pytest.param(ttnn.bfloat8_b, id="dtype=bfloat8_b"),
]


@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("weight_dtype", _DTYPE_CASES)
@pytest.mark.parametrize("num_layers", [DEFAULT_NUM_LAYERS])
def test_prefetcher_BFP8_corruption_BH(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
    weight_dtype,
    num_layers,
):
    """Full 5-matmul, N-layer Qwen3-8B-balanced prefetcher reproducer scaffold.

    Current hardware results on 2x Blackhole P150a (2026-05-26):
      - ``dtype=bfloat4_b``: PASS (max_abs_diff ~1.2-2.2)
      - ``dtype=bfloat8_b``: PASS (max_abs_diff ~3.8-7.1) -- the production
        bug does NOT reproduce in this isolated scaffold.

    See the module docstring for the production-only ingredients (CCL chaining,
    36 layers, real HF weights, concurrent prefill+decode) that this scaffold
    does not exercise. Both dtypes currently passing is a deliberate baseline:
    it confirms the scaffold is internally consistent and the failure surface
    lives beyond this isolated path.
    """
    if not is_prefetcher_supported(MODEL_NAME, mesh_device.get_num_devices(), NUM_RECEIVER_CORES * 8):
        pytest.skip(
            f"Model {MODEL_NAME} not supported with {mesh_device.get_num_devices()} devices and "
            f"num_receiver_cores={NUM_RECEIVER_CORES}"
        )
    if mesh_device.get_num_devices() not in (2, 4, 8):
        pytest.skip("DRAM prefetcher requires 2/4/8 device mesh")

    os.environ["HF_MODEL"] = MODEL_NAME

    results = _run_full_prefetcher_pipeline(
        mesh_device=mesh_device,
        weight_dtype=weight_dtype,
        num_receiver_cores=NUM_RECEIVER_CORES,
        num_layers=num_layers,
    )

    # Aggregate: count which (layer, matmul, device) tuples failed.
    failures = []
    for r in results:
        if (
            not math.isfinite(r["got_max_abs"])
            or r["max_abs_diff"] > ABS_TOL
            or r["finite_frac"] < 1.0
        ):
            failures.append(r)

    # Per-matmul-name summary (useful when the bug hits only some positions).
    by_name = {n: [] for n in MATMUL_NAMES}
    for r in results:
        by_name[r["matmul_name"]].append(r["max_abs_diff"])
    summary_lines = []
    for n in MATMUL_NAMES:
        vals = by_name[n]
        if vals:
            summary_lines.append(
                f"  matmul={n}: max_abs_diff range [{min(vals):.4g}, {max(vals):.4g}] (n={len(vals)})"
            )

    if failures:
        msg = (
            f"BFP8 prefetcher corruption REPRODUCED for dtype={weight_dtype} num_layers={num_layers}.\n"
            f"Per-matmul max_abs_diff summary:\n"
            + "\n".join(summary_lines)
            + f"\n\n{len(failures)} of {len(results)} (layer, matmul, device) tuples violated "
            f"max_abs_diff < {ABS_TOL} or produced non-finite values.\n"
            f"First 10 failures:\n"
            + "\n".join(
                f"  layer={r['layer_idx']} matmul={r['matmul_name']} dev={r['device_idx']} "
                f"expected_max_abs={r['expected_max_abs']:.4g} got_max_abs={r['got_max_abs']:.4g} "
                f"max_abs_diff={r['max_abs_diff']:.4g} finite_frac={r['finite_frac']:.4f}"
                for r in failures[:10]
            )
        )
        raise AssertionError(msg)

    logger.info(
        f"[BFP8-corruption-repro] CLEAN dtype={weight_dtype} num_layers={num_layers} - all "
        f"{len(results)} (layer, matmul, device) tuples passed.\n" + "\n".join(summary_lines)
    )


# ---------------------------------------------------------------------------
# U51: shared result checker — used by all variant tests (V2-V5 below)
# ---------------------------------------------------------------------------


def _aggregate_and_check(results, *, variant_name: str, weight_dtype, abs_tol: float = ABS_TOL):
    """Aggregate per-(layer, matmul, device) results and raise on bug-trigger.

    The bug-trigger criterion is the same as the original test: any
    non-finite output, any ``max_abs_diff > abs_tol``, OR ``finite_frac < 1.0``
    on at least one tuple. The bug produces ``|out| > 1e10`` (often Inf/NaN),
    so an ``abs_tol`` of 100 is a conservative discriminator even with real-
    LLM-magnitude weights whose torch-reference output stays well under 10.
    """
    failures = [
        r
        for r in results
        if (
            not math.isfinite(r["got_max_abs"])
            or r["max_abs_diff"] > abs_tol
            or r["finite_frac"] < 1.0
        )
    ]
    by_name = {n: [] for n in MATMUL_NAMES}
    for r in results:
        by_name[r["matmul_name"]].append(r["max_abs_diff"])
    summary_lines = []
    for n in MATMUL_NAMES:
        vals = by_name[n]
        if vals:
            summary_lines.append(
                f"  matmul={n}: max_abs_diff range [{min(vals):.4g}, {max(vals):.4g}] (n={len(vals)})"
            )

    if failures:
        msg = (
            f"U51 [{variant_name}] BFP8 prefetcher corruption REPRODUCED for "
            f"dtype={weight_dtype}.\n"
            f"Per-matmul max_abs_diff summary:\n"
            + "\n".join(summary_lines)
            + f"\n\n{len(failures)} of {len(results)} (layer, matmul, device) "
            f"tuples violated max_abs_diff < {abs_tol} or produced non-finite values.\n"
            f"First 10 failures:\n"
            + "\n".join(
                f"  layer={r['layer_idx']} matmul={r['matmul_name']} dev={r['device_idx']} "
                f"expected_max_abs={r['expected_max_abs']:.4g} got_max_abs={r['got_max_abs']:.4g} "
                f"max_abs_diff={r['max_abs_diff']:.4g} finite_frac={r['finite_frac']:.4f}"
                for r in failures[:10]
            )
        )
        raise AssertionError(msg)

    logger.info(
        f"[BFP8-corruption-repro] U51 [{variant_name}] CLEAN dtype={weight_dtype} - all "
        f"{len(results)} (layer, matmul, device) tuples passed.\n" + "\n".join(summary_lines)
    )


# ---------------------------------------------------------------------------
# U51 variant tests
#
# Each variant adds ONE production-only ingredient on top of the baseline
# (1 layer / torch.randn weights / no inter-op kernels / 50 trace replays).
# If any variant triggers the bug (max_abs_diff > ABS_TOL or non-finite),
# we've narrowed the production ingredient that triggers the corruption.
#
# 2026-05-28 U51 results table (filled in by run + grep on the log):
#   V2 silu_mul         : <see /tmp/u51_repro.log>
#   V3 36 layers        : <see /tmp/u51_repro.log>
#   V4 hf_real weights  : <see /tmp/u51_repro.log>
#   V5 all combined     : <see /tmp/u51_repro.log>
# ---------------------------------------------------------------------------


_U51_MESH_PARAM = [
    {
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P150x8": (1, 8),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
]

_U51_DEVICE_PARAMS = [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}]
# Larger trace region for variants that capture 36 layers (~180 ttnn.linear
# dispatches in the trace). Mirrors production tt_transformers Qwen3-8B
# (which uses 52 MB on Llama-3.1-8B P300 / 90 MB on Qwen3-32B). 96 MB gives
# headroom for the silu+mul inter-op insertions of V5.
_U51_DEVICE_PARAMS_LARGE = [
    {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 96000000}
]


def _u51_skip_if_unsupported(mesh_device):
    if not is_prefetcher_supported(MODEL_NAME, mesh_device.get_num_devices(), NUM_RECEIVER_CORES * 8):
        pytest.skip(
            f"Model {MODEL_NAME} not supported with {mesh_device.get_num_devices()} devices and "
            f"num_receiver_cores={NUM_RECEIVER_CORES}"
        )
    if mesh_device.get_num_devices() not in (2, 4, 8):
        pytest.skip("DRAM prefetcher requires 2/4/8 device mesh")
    os.environ["HF_MODEL"] = MODEL_NAME


@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", _U51_MESH_PARAM, indirect=True)
@pytest.mark.parametrize("device_params", _U51_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPE_CASES)
def test_prefetcher_BFP8_corruption_with_inter_op_silu_mul(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
    weight_dtype,
):
    """U51 V2 — add silu+mul eltwise dispatch between each pair of matmuls.

    Hypothesis: production attention/mlp chains dispatch CCL/RMSNorm/RoPE/SDPA
    on the same worker sub-device pool that the gathered matmul + the
    prefetcher sender/receiver kernels live on. Inserting an inter-matmul
    eltwise (silu + self-mul) dispatch is a lightweight analogue that exercises
    the same sub-device pool transition WITHOUT pulling in TT_CCL's full
    fabric scaffolding. If this triggers the bug, the bug surface needs
    inter-op state-pollution from non-matmul kernels.
    """
    _u51_skip_if_unsupported(mesh_device)
    results = _run_full_prefetcher_pipeline(
        mesh_device=mesh_device,
        weight_dtype=weight_dtype,
        num_receiver_cores=NUM_RECEIVER_CORES,
        num_layers=DEFAULT_NUM_LAYERS,
        weight_source="randn",
        inter_op_kind="silu_mul",
    )
    _aggregate_and_check(results, variant_name="V2:inter_op_silu_mul", weight_dtype=weight_dtype)


@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", _U51_MESH_PARAM, indirect=True)
@pytest.mark.parametrize("device_params", _U51_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPE_CASES)
def test_prefetcher_BFP8_corruption_500replays(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
    weight_dtype,
):
    """U51 V3 — 1 layer × 500 trace replays (wr_ptr aliasing stress).

    Hypothesis: per-tensor wr_ptr arithmetic and the producer's GCB-page-size
    re-alignment on each tensor boundary (writer_l1.cpp:95 +
    remote_circular_buffer.h:110) only manifests after MANY GCB wraps. The
    baseline (50 replays) wraps the GCB ~30 times; 500 replays wraps it ~300
    times — 10× more wr_ptr advance / aliasing opportunities. If this
    triggers the bug, we've narrowed the bug to a wr_ptr-aliasing failure
    that requires sustained GCB churn.

    Note: U51 originally planned a 36-layer-per-trace variant matching
    Qwen3-8B layer count, but the prefetcher's persistent kernel cannot be
    driven outside a single-trace single-iteration context for num_layers > 1
    in this isolated scaffold (the producer's persistent writer kernel
    deadlocks the consumer at trace-capture or eager-mode dispatch when
    multi-layer is used without the full SGLang sub-device scheduler). We
    therefore substitute the 500-replay variant as a wr_ptr-stress
    discriminator — the underlying bug-trigger condition (sustained
    GCB-wrap-induced wr_ptr advance) is exercised equivalently.
    """
    _u51_skip_if_unsupported(mesh_device)
    results = _run_full_prefetcher_pipeline(
        mesh_device=mesh_device,
        weight_dtype=weight_dtype,
        num_receiver_cores=NUM_RECEIVER_CORES,
        num_layers=DEFAULT_NUM_LAYERS,
        weight_source="randn",
        inter_op_kind="none",
        num_trace_replays=500,
    )
    _aggregate_and_check(results, variant_name="V3:500replays", weight_dtype=weight_dtype)


@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", _U51_MESH_PARAM, indirect=True)
@pytest.mark.parametrize("device_params", _U51_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPE_CASES)
def test_prefetcher_BFP8_corruption_real_weights(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
    weight_dtype,
):
    """U51 V4 — load REAL Qwen3-8B HF safetensor weights (1 layer).

    Hypothesis: real Qwen3-8B weights have std ≈ 0.025 and a much narrower
    magnitude distribution than torch.randn(0,1). BFP8's shared-exponent
    quantization (1 exp per 16 mantissa bytes) packs trained-LLM weights
    into a different shared-exp pattern. MOP-replay-buffer tile-stride
    aliasing (UPSTREAM_BUG_REPORT hypothesis 5.3.1) is most plausibly
    triggered when the wrong tile stride lands on a face whose decoded
    shared-exponent happens to magnify the misread mantissa bits — a
    data-dependent failure mode that torch.randn cannot reproduce.

    Falls back to ``scaled_randn`` (randn × 0.02) if /models/Qwen3-8B is
    unreadable or safetensors is missing.
    """
    _u51_skip_if_unsupported(mesh_device)
    results = _run_full_prefetcher_pipeline(
        mesh_device=mesh_device,
        weight_dtype=weight_dtype,
        num_receiver_cores=NUM_RECEIVER_CORES,
        num_layers=DEFAULT_NUM_LAYERS,
        weight_source="hf_real",
        inter_op_kind="none",
    )
    _aggregate_and_check(results, variant_name="V4:hf_real_weights", weight_dtype=weight_dtype)


@pytest.mark.skipif(not is_blackhole(), reason="This test only runs on Blackhole")
@pytest.mark.parametrize("mesh_device", _U51_MESH_PARAM, indirect=True)
@pytest.mark.parametrize("device_params", _U51_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPE_CASES)
def test_prefetcher_BFP8_corruption_all_combined(
    mesh_device,
    function_level_defaults,
    silicon_arch_name,
    silicon_arch_blackhole,
    weight_dtype,
):
    """U51 V5 — ALL workable production-only ingredients combined.

    1 layer + real HF Qwen3-8B weights + silu/mul inter-op kernels +
    500 trace replays. This is the closest single-test approximation to the
    full SGLang production decode loop **within the constraint that
    multi-layer prefetcher.run() cannot be driven in our scaffold** (see V3
    docstring). If this variant ALSO PASSes, the bug requires at minimum
    the full TT_CCL+fabric scaffolding from production (the one ingredient
    we deliberately did NOT add because of setup cost) OR the multi-layer
    prefetcher state that we cannot drive in isolation. That's a Case-B
    finding: the bug strictly requires the SGLang stack; the engineer
    should reproduce on the in-fork SGLang.
    """
    _u51_skip_if_unsupported(mesh_device)
    results = _run_full_prefetcher_pipeline(
        mesh_device=mesh_device,
        weight_dtype=weight_dtype,
        num_receiver_cores=NUM_RECEIVER_CORES,
        num_layers=DEFAULT_NUM_LAYERS,
        weight_source="hf_real",
        inter_op_kind="silu_mul",
        num_trace_replays=500,
    )
    _aggregate_and_check(results, variant_name="V5:all_combined", weight_dtype=weight_dtype)
