# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce


# U5 — Prefetcher cross-sub-device output barrier (2026-05-24).
# See attention.py:_u5_pref_subdev for full rationale. Inlined here to avoid
# inter-module import-order coupling between mlp.py and attention.py.
def _u5_pref_subdev(prefetcher):
    import os as _os
    if prefetcher is None:
        return None
    # U16 (2026-05-25): SGLANG_TT_W2_RS_BARRIER does NOT route here.
    # `receiver_sub_device` runs persistent kernels (no completion
    # signal), so routing CCL to it deadlocks finish_nolock.  The
    # W2_RS_BARRIER fix is a dispatch.cpp-level cached-path BARRIER
    # flag — see tt_metal/impl/program/dispatch.cpp:451.  This
    # function stays canonical (worker) under U16.
    # SGLANG_TT_PREFETCHER_OUTPUT_BARRIER retained ONLY for the U5
    # diagnostic re-run path; setting it still deadlocks (RULED OUT).
    if _os.environ.get("SGLANG_TT_PREFETCHER_OUTPUT_BARRIER", "0") == "1":
        return prefetcher.receiver_sub_device_id
    return prefetcher.worker_sub_device_id
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        # Define the prefetcher object
        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        # If padding was applied (e.g. via env var), add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        # Note: unsqueeze(0).unsqueeze(0) makes weights 4D [1, 1, H, W] to match attention weights
        # This is required for the dram_prefetcher to correctly interpret all weights
        # U11 (2026-05-24): SGLANG_TT_PREFETCHER_ZERO_WEIGHTS=1 replaces every
        # MLP / attention weight tensor with all-zeros BEFORE upload to device.
        # Combined with the matching attention.py injection, this makes the
        # entire matmul math `out = x @ 0 = 0`. Used by the U11 functional test
        # to distinguish "math is broken under ENABLE_GLOBAL_CB" (output stays
        # at ~2^20 even with zero inputs) from "input data flow is wrong"
        # (output goes to ~0 with zero inputs, proving math itself works).
        # Cache filename is suffixed to avoid loading prior non-zero cache.
        import os as _os_u11_mlp
        _u11_zero_weights = _os_u11_mlp.environ.get("SGLANG_TT_PREFETCHER_ZERO_WEIGHTS", "0") == "1"

        def _maybe_zero(t):
            if _u11_zero_weights:
                return torch.zeros_like(t)
            return t

        def _u11_cache(name):
            base = cache_name(name)
            if base is None:
                return None
            if _u11_zero_weights:
                return type(base)(str(base) + "_u11zero")
            return base

        def as_sharded_tensor(name, ttnn_dtype, dims):
            # First get the raw weight and transpose it
            raw_weight = torch_weight(name[:2])  # This is 2D: [H, W]
            # Pad if needed
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            # Make 4D: [1, 1, H, W] - CRITICAL for prefetcher to work correctly
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)
            torch_tensor = _maybe_zero(torch_tensor)

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=ttnn_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
                ),
                cache_file_name=_u11_cache(name),
            )
            return result

        # Sharded weights
        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)  # cross_block uses the configuration of the first decoder

        # When prefetcher is enabled, use consistent dtypes across all layers to avoid
        # race conditions caused by different block sizes
        use_prefetcher = prefetcher is not None

        self.decoders_optimizations = self.args.decoders_optimizations

        ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        self.w1 = as_sharded_tensor(
            "w1_sharded", ff1_3_dtype, dims=w1_dims
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)

        # Default activation is SILU
        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

        # Insert the tensors into the prefetcher if it is used
        if self.prefetcher is not None:
            # SGLANG_TT_PREFETCHER_SKIP_W1/_W3/_W2: skip individual MLP weights
            # from the prefetcher's per-layer queue. Bug-narrowing ablation:
            # combined with SKIP_WO/SKIP_WQKV, we can isolate the corruption
            # source weight or dtype (BFP4 vs BFP8).
            import os as _os_mlp
            self._skip_w1 = _os_mlp.environ.get("SGLANG_TT_PREFETCHER_SKIP_W1", "0") == "1"
            self._skip_w3 = _os_mlp.environ.get("SGLANG_TT_PREFETCHER_SKIP_W3", "0") == "1"
            self._skip_w2 = _os_mlp.environ.get("SGLANG_TT_PREFETCHER_SKIP_W2", "0") == "1"
            # U3 (2026-05-23): permuted-DRAM-grid prefetcher experiment
            self._permuted_dram_grid = _os_mlp.environ.get(
                "SGLANG_TT_PREFETCHER_PERMUTED_DRAM_GRID", "0"
            ) == "1"

            # Phase-B.8 reroute fix: the canonical `w1_w3_mem_config` /
            # `w2_mem_config` lay weights out on `dram_weight_grid`
            # (a contiguous (0,0)-(N-1,0) range). The `prefetch=False,
            # num_global_cb_receivers=1` ring matmul kernel reads weights via
            # the optimal-DRAM-bank-to-worker mapping (see
            # matmul_multicore_reuse_mcast_1d_program_factory.cpp:2466+),
            # which expects shard i to live on DRAM bank
            # `prefetcher.dram_banks()[i]` (a permuted order, e.g.
            # [1,3,2,0,5,7,6,4] on Blackhole). When the SKIP_* fallback
            # routes a canonical-grid weight through this kernel, shard
            # ordering is scrambled — producing the "0/10 garbage" we
            # observed in Phase A/B.1/B.7. Build SKIP-only ring-layout
            # variants of any SKIP'd weight that mirror lm_head's pattern
            # (`dram_grid=prefetcher.to_core_range_set(prefetcher.dram_banks())`).
            # The canonical full-prefetcher path is untouched.
            def _ring_mem_config(k, n):
                return args.create_dram_sharded_mem_config(
                    k=k,
                    n=n,
                    dram_grid=self.prefetcher.to_core_range_set(self.prefetcher.dram_banks()),
                )

            def _make_skip_ring_weight(name, dtype, dims, k, n):
                # Re-load + transpose like as_sharded_tensor, but with the
                # prefetcher-bank DRAM grid.
                raw_weight = torch_weight(name[:2])
                padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
                torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)
                torch_tensor = _maybe_zero(torch_tensor)
                cache = _u11_cache(f"{name}_skip_ring")
                return ttnn.as_tensor(
                    torch_tensor,
                    dtype=dtype,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=dims, mesh_shape=args.cluster_shape
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_ring_mem_config(k, n),
                    cache_file_name=cache,
                )

            def _make_pdg_weight(name, dtype, dims, k, n):
                # U3: same as _make_skip_ring_weight but with _pdg suffix for
                # cache separation. The permuted-DRAM-grid placement is the
                # SAME as the skip_ring path; only the use site differs (we
                # register THIS for the prefetcher GlobalCB path, not the
                # SKIP fallback).
                raw_weight = torch_weight(name[:2])
                padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
                torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)
                torch_tensor = _maybe_zero(torch_tensor)
                cache = _u11_cache(f"{name}_pdg")
                return ttnn.as_tensor(
                    torch_tensor,
                    dtype=dtype,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=dims, mesh_shape=args.cluster_shape
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_ring_mem_config(k, n),
                    cache_file_name=cache,
                )

            if not args.is_galaxy:
                if self._skip_w1:
                    self.w1_skip_ring = _make_skip_ring_weight(
                        "w1_sharded",
                        ff1_3_dtype,
                        w1_dims,
                        args.dim,
                        args.hidden_dim // args.num_devices,
                    )
                else:
                    self.w1_skip_ring = None
                if self._skip_w3:
                    self.w3_skip_ring = _make_skip_ring_weight(
                        "w3_sharded",
                        ff1_3_dtype,
                        w1_dims,
                        args.dim,
                        args.hidden_dim // args.num_devices,
                    )
                else:
                    self.w3_skip_ring = None
                if self._skip_w2:
                    self.w2_skip_ring = _make_skip_ring_weight(
                        "w2_sharded",
                        ff2_dtype,
                        w2_dims,
                        args.hidden_dim // args.num_devices,
                        args.dim,
                    )
                else:
                    self.w2_skip_ring = None
            else:
                # Galaxy path: existing default keeps working since SKIP_*
                # is not validated for Galaxy in Phase B.
                self.w1_skip_ring = None
                self.w3_skip_ring = None
                self.w2_skip_ring = None

            # U3 permuted-DRAM-grid variants (non-galaxy only)
            if self._permuted_dram_grid and not args.is_galaxy:
                self.w1_pdg = _make_pdg_weight(
                    "w1_sharded", ff1_3_dtype, w1_dims,
                    args.dim, args.hidden_dim // args.num_devices,
                )
                self.w3_pdg = _make_pdg_weight(
                    "w3_sharded", ff1_3_dtype, w1_dims,
                    args.dim, args.hidden_dim // args.num_devices,
                )
                self.w2_pdg = _make_pdg_weight(
                    "w2_sharded", ff2_dtype, w2_dims,
                    args.hidden_dim // args.num_devices, args.dim,
                )
            else:
                self.w1_pdg = None
                self.w3_pdg = None
                self.w2_pdg = None

            def register_weights():
                if not self._skip_w1:
                    self.prefetcher.insert_tensor(
                        self.w1_pdg if self._permuted_dram_grid and self.w1_pdg is not None else self.w1
                    )
                if not self._skip_w3:
                    self.prefetcher.insert_tensor(
                        self.w3_pdg if self._permuted_dram_grid and self.w3_pdg is not None else self.w3
                    )
                if not self._skip_w2:
                    self.prefetcher.insert_tensor(
                        self.w2_pdg if self._permuted_dram_grid and self.w2_pdg is not None else self.w2
                    )

            self.prefetcher.register_callback(register_weights)
        else:
            self._skip_w1 = False
            self._skip_w3 = False
            self._skip_w2 = False
            self._permuted_dram_grid = False
            self.w1_skip_ring = None
            self.w3_skip_ring = None
            self.w2_skip_ring = None
            self.w1_pdg = None
            self.w3_pdg = None
            self.w2_pdg = None

    def _u26c_prealloc_w2_out(self):
        """U26 Path C — pre-allocate the w2_out destination buffer OUTSIDE the trace.

        Lazily creates a persistent device tensor at MLP forward dispatch time,
        but BEFORE trace capture closes, with the exact mem_config / shape that
        ttnn.linear(w2) would otherwise allocate.  Passed via
        optional_output_tensor= so the matmul writes INTO this preallocated
        buffer rather than allocating a fresh in-trace one each iteration.

        This is the Path C workaround for the U25 / U26 0xa6700 stomp:
        - REALLOCATE_W2 (U22) moves the buffer post-matmul but breaks under
          real weights because reallocate fires inside the captured trace.
        - Pre-allocating outside the trace gives the trace replay a stable
          destination address (chosen by the allocator at first call), and
          no in-trace allocator activity happens for w2_out.
        Returns the persistent tensor, or None if any step fails.
        """
        if getattr(self, "_u26c_w2_out", None) is not None:
            return self._u26c_w2_out
        try:
            _mc = self.args.get_mlp_ff2_mem_config(Mode.DECODE, self.prefetcher)
            # Match the W2 matmul's output dtype: ccl_dtype on Galaxy, bf16
            # otherwise.  See `dtype=self.args.ccl_dtype if TG else
            # activation_dtype or ttnn.bfloat16` in forward().
            _dtype = (
                self.args.ccl_dtype if self.args.is_galaxy else ttnn.bfloat16
            )
            # Shape: [1, 1, 32, dim] — matches the W2 matmul output before
            # reduce_scatter (full dim, per-receiver-shard width = dim/ring_size).
            _shape = (1, 1, 32, self.args.dim)
            _t = ttnn.allocate_tensor_on_device(
                ttnn.Shape(_shape),
                _dtype,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                _mc,
            )
            self._u26c_w2_out = _t
            print(
                f"[U26C_PREALLOC] layer={self.layer_num} "
                f"w2_out.addr=0x{_t.buffer_address():x} "
                f"shape={tuple(_t.shape)} dtype={_dtype}",
                flush=True,
            )
            return self._u26c_w2_out
        except Exception as _e:
            print(f"[U26C_PREALLOC] ERROR: {type(_e).__name__}: {_e}", flush=True)
            self._u26c_w2_out = None
            return None

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)  # cross_block uses the configuration of the first decoder
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:  # 512 if Blackhole, 1024 if Wormhole
            # Reshape input to to fit on device and parallelize computation
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)

        # SGLANG_TT_PREFETCHER_SKIP_W{1,3} ablation: when a weight is excluded
        # from the prefetcher queue, the matmul reads its DRAM-sharded weight
        # directly (no GlobalCB read) using a prefetch=False ring program
        # config (same pattern as lm_head.py / SKIP_WO).
        if self.prefetcher is not None and mode == Mode.DECODE and (
            getattr(self, "_skip_w1", False) or getattr(self, "_skip_w3", False)
        ):
            _pc_w13_skip = self.args.matmul_1d_ring_config(
                1,
                32,
                self.args.dim,
                self.args.hidden_dim // self.args.cluster_shape[1],
                self.prefetcher.ring_size,
                num_global_cb_receivers=1,
                prefetch=False,
            )
        else:
            _pc_w13_skip = None

        _w1_use_skip = self.prefetcher is not None and mode == Mode.DECODE and getattr(self, "_skip_w1", False)
        _w3_use_skip = self.prefetcher is not None and mode == Mode.DECODE and getattr(self, "_skip_w3", False)
        # U3: when permuted-DRAM-grid prefetcher path is active, use the pdg
        # weight variant for the prefetcher (full-prefetcher GlobalCB) path.
        _use_pdg = (
            self.prefetcher is not None and mode == Mode.DECODE
            and getattr(self, "_permuted_dram_grid", False)
        )

        def _w1_weight():
            if _w1_use_skip and getattr(self, "w1_skip_ring", None) is not None:
                return self.w1_skip_ring
            if _use_pdg and getattr(self, "w1_pdg", None) is not None:
                return self.w1_pdg
            return self.w1

        def _w3_weight():
            if _w3_use_skip and getattr(self, "w3_skip_ring", None) is not None:
                return self.w3_skip_ring
            if _use_pdg and getattr(self, "w3_pdg", None) is not None:
                return self.w3_pdg
            return self.w3

        w1_out = ttnn.linear(
            x,
            _w1_weight(),
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=_pc_w13_skip if _w1_use_skip else pc_1,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=None if _w1_use_skip else (self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None),
            sub_device_id=self.prefetcher.receiver_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        w3_out = ttnn.linear(
            x,
            _w3_weight(),
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=_pc_w13_skip if _w3_use_skip else pc_3,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=None if _w3_use_skip else (self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None),
            sub_device_id=self.prefetcher.receiver_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        ttnn.deallocate(x)

        if TG:
            # if mode == "decode" and self.dim!=8192:
            #     w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            #     w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
            if self.dim == 8192 or mode == Mode.PREFILL:
                input_mem_cfg = w1_out.memory_config()

                cluster_axis = 1
                w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w1_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=self.tt_ccl.get_num_links(cluster_axis),
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

                w3_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w3_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=1,
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                # NOTE: In MLP All-reduce hard codes to 2 links, so we do not get the dynamic link count from the CCL class
                # to avoid any performance regressions.
                # U5: when env-gated (SGLANG_TT_PREFETCHER_OUTPUT_BARRIER=1),
                # inject prefetcher receiver_sub_device_id so the all_reduce
                # dispatches on the same stream as the w1/w3 matmul
                # (eliminates the cross-sub-device dispatch sync gap). When
                # env is unset OR prefetcher is None, pass nothing — preserves
                # the canonical sync `ttnn.reduce_scatter` path.
                import os as _u5_os
                _u5_kwargs = {}
                # U16: W2_RS_BARRIER is dispatch-cpp-level; do NOT route
                # to receiver here (deadlock — persistent kernels).
                # Only the legacy OUTPUT_BARRIER reroutes (RULED OUT).
                if (
                    self.prefetcher is not None
                    and mode == Mode.DECODE
                    and _u5_os.environ.get("SGLANG_TT_PREFETCHER_OUTPUT_BARRIER", "0") == "1"
                ):
                    _u5_kwargs["subdevice_id"] = self.prefetcher.receiver_sub_device_id
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                    **_u5_kwargs,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                    **_u5_kwargs,
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if mode == Mode.DECODE and not TG and self.prefetcher is None:
            # w2 may use a different core grid, this is a no-op if they already match
            w2_in = ttnn.to_memory_config(w2_in, self.args.get_mlp_binary_mult_mem_config(mode))

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == Mode.PREFILL):
            cluster_axis = 1
            w2_in = ttnn.experimental.all_gather_async(
                w2_in,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=2,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            if mode == Mode.DECODE:
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        # U26 Phase A — adjacent-buffer ownership probe (env-gated).
        # Enumerate all L1 buffers in [0xa6000, 0xb0000] BEFORE the W2 matmul
        # allocates w2_out at 0xa6700.  Whatever lives at 0xa8700 NOW (before
        # W2 dispatches) is the SIBLING we are hunting.  Bracketed with an
        # AFTER-W2 enumeration to confirm w2_out lands at 0xa6700.  Runs on
        # EVERY layer (not just layer 0) so we can tell whether 0xa8700's
        # owner is a previous-layer persistent buffer or a per-iteration
        # allocation.  Default-off.
        import os as _u26a_os
        if (mode == Mode.DECODE
                and _u26a_os.environ.get("SGLANG_TT_U26_ADJ_BUFFER_PROBE", "0") == "1"):
            try:
                self._u26_iter = getattr(self, "_u26_iter", 0) + 1
                _u26a_lo = 0xa6000
                _u26a_hi = 0xb0000
                _u26a_devs = (
                    self.mesh_device.get_devices()
                    if hasattr(self.mesh_device, "get_devices")
                    else [self.mesh_device]
                )
                _u26a_bufs = ttnn._ttnn.reports.get_buffers(list(_u26a_devs))
                _u26a_near = sorted(
                    [(int(_b.address), _b.buffer_type, _b.buffer_layout,
                      _b.max_size_per_bank)
                     for _b in _u26a_bufs
                     if _u26a_lo <= int(_b.address) <= _u26a_hi]
                )
                print(
                    f"[U26_ADJ_PRE_W2] iter={self._u26_iter} "
                    f"layer={getattr(self, 'layer_num', '?')} mode={mode} "
                    f"bufs_in_range={len(_u26a_near)}",
                    flush=True,
                )
                for _b in _u26a_near[:32]:
                    print(
                        f"[U26_ADJ_PRE_W2]   addr=0x{_b[0]:x} bt={_b[1]} "
                        f"bl={_b[2]} sz_per_bank={_b[3]}",
                        flush=True,
                    )
            except Exception as _u26a_e:
                print(f"[U26_ADJ_PRE_W2] ERROR: {type(_u26a_e).__name__}: {_u26a_e}",
                      flush=True)

        if seq_len > 128 and mode != Mode.DECODE:
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2,
            )
        else:
            _w2_use_skip = self.prefetcher is not None and mode == Mode.DECODE and getattr(self, "_skip_w2", False)
            if _w2_use_skip:
                _pc_w2_skip = self.args.matmul_1d_ring_config(
                    1,
                    32,
                    self.args.hidden_dim // self.args.cluster_shape[1],
                    self.args.dim,
                    self.prefetcher.ring_size,
                    num_global_cb_receivers=1,
                    prefetch=False,
                )
            else:
                _pc_w2_skip = None
            def _w2_weight():
                if _w2_use_skip and getattr(self, "w2_skip_ring", None) is not None:
                    return self.w2_skip_ring
                if (self.prefetcher is not None and mode == Mode.DECODE
                        and getattr(self, "_permuted_dram_grid", False)
                        and getattr(self, "w2_pdg", None) is not None):
                    return self.w2_pdg
                return self.w2

            # U26 Path C — pre-allocated w2_out destination (env-gated).
            # NOTE (U26 finding): pre-allocating via ttnn.allocate_tensor_on_device
            # BEFORE trace capture fails with "Tensor is not allocated" once
            # the trace capture closes — the allocation's storage is invalidated
            # between the compile pass and the capture pass.  Kept here as a
            # historical record; do NOT enable until pre-alloc lifetime is
            # threaded properly through trace capture.
            import os as _u26c_os
            _u26c_active = (
                mode == Mode.DECODE
                and self.prefetcher is not None
                and not _w2_use_skip
                and _u26c_os.environ.get("SGLANG_TT_U26_PREALLOC_W2", "0") == "1"
            )
            _u26c_out_t = self._u26c_prealloc_w2_out() if _u26c_active else None
            w2_out = ttnn.linear(
                w2_in,
                _w2_weight(),
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
                program_config=_pc_w2_skip if _w2_use_skip else pc_2,
                memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
                core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                global_cb=None if _w2_use_skip else (self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None),
                sub_device_id=self.prefetcher.receiver_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
                optional_output_tensor=_u26c_out_t,
            )
        # U28 — capture w2_in shard grid BEFORE deallocation.
        try:
            if (mode == Mode.DECODE and self.prefetcher is not None
                    and _u26a_os.environ.get("SGLANG_TT_U28_PER_CORE_PROBE", "0") == "1"
                    and (int(getattr(self, "layer_num", -1)) == 0)
                    and not getattr(self, "_u28_w2in_logged", False)):
                _u28_w2in_mc = w2_in.memory_config()
                _u28_w2in_ss = _u28_w2in_mc.shard_spec
                _u28_w2in_cores = []
                if _u28_w2in_ss is not None:
                    for _u28_cr in _u28_w2in_ss.grid.ranges():
                        for _ux in range(_u28_cr.start.x, _u28_cr.end.x + 1):
                            for _uy in range(_u28_cr.start.y, _u28_cr.end.y + 1):
                                _u28_w2in_cores.append((_ux, _uy))
                print(
                    f"[U28_W2_IN_GRID] layer=0 "
                    f"mem_layout={_u28_w2in_mc.memory_layout} "
                    f"num_cores={len(_u28_w2in_cores)} "
                    f"shard_shape={_u28_w2in_ss.shape if _u28_w2in_ss else 'N/A'}",
                    flush=True,
                )
                print(f"[U28_W2_IN_GRID]   cores={sorted(_u28_w2in_cores)}", flush=True)
                self._u28_w2in_logged = True
        except Exception as _u28e:
            print(f"[U28_W2_IN_GRID] ERROR: {type(_u28e).__name__}: {_u28e}", flush=True)
        ttnn.deallocate(w2_in)

        # U26 Phase A — POST-W2 buffer enumeration.  Pair with PRE_W2 above.
        # Confirms w2_out lands at 0xa6700 and shows whether any NEW buffer
        # was allocated between PRE and POST.
        if (mode == Mode.DECODE
                and _u26a_os.environ.get("SGLANG_TT_U26_ADJ_BUFFER_PROBE", "0") == "1"):
            try:
                _u26a_lo = 0xa6000
                _u26a_hi = 0xb0000
                _u26a_devs = (
                    self.mesh_device.get_devices()
                    if hasattr(self.mesh_device, "get_devices")
                    else [self.mesh_device]
                )
                _u26a_bufs = ttnn._ttnn.reports.get_buffers(list(_u26a_devs))
                _u26a_near = sorted(
                    [(int(_b.address), _b.buffer_type, _b.buffer_layout,
                      _b.max_size_per_bank)
                     for _b in _u26a_bufs
                     if _u26a_lo <= int(_b.address) <= _u26a_hi]
                )
                _u26a_w2 = w2_out.buffer_address()
                print(
                    f"[U26_ADJ_POST_W2] iter={self._u26_iter} "
                    f"layer={getattr(self, 'layer_num', '?')} "
                    f"w2_out.addr=0x{_u26a_w2:x} "
                    f"bufs_in_range={len(_u26a_near)}",
                    flush=True,
                )
                for _b in _u26a_near[:32]:
                    print(
                        f"[U26_ADJ_POST_W2]   addr=0x{_b[0]:x} bt={_b[1]} "
                        f"bl={_b[2]} sz_per_bank={_b[3]}",
                        flush=True,
                    )
            except Exception as _u26a_e:
                print(f"[U26_ADJ_POST_W2] ERROR: {type(_u26a_e).__name__}: {_u26a_e}",
                      flush=True)

        # ---- U18 Probe P2 (env-gated): W2 matmul output buffer address ----
        # Print w2_out.buffer_address() on EVERY layer 0 invocation so we
        # can compare to U17's RS reader in_addr=0xa6700.  If the address
        # CHANGES across iterations -> trace replay isn't using the
        # captured addr -> P2 confirmed.  If it's stable at 0xa6700 ->
        # the config side is right and the bug is P1 or P3.
        import os as _u18p_os
        if _u18p_os.environ.get("SGLANG_TT_U18_ADDR_PROBE", "0") == "1":
            try:
                _u18p_layer = (
                    (int(self.layer_num) == 0) if hasattr(self, "layer_num") else True
                )
                if _u18p_layer:
                    self._u18_forward_count = getattr(self, "_u18_forward_count", 0) + 1
                    _u18p_ba = w2_out.buffer_address()
                    _u18p_mc = w2_out.memory_config()
                    _u18p_ss = _u18p_mc.shard_spec
                    print(
                        f"[U18_W2_ADDR] iter={self._u18_forward_count} "
                        f"layer={getattr(self, 'layer_num', '?')} "
                        f"w2_out.addr=0x{_u18p_ba:x} "
                        f"shape={tuple(w2_out.shape)} "
                        f"shard_grid={_u18p_ss.grid if _u18p_ss else 'None'}",
                        flush=True,
                    )
            except Exception as _u18p_e:
                print(f"[U18_W2_ADDR] ERROR: {type(_u18p_e).__name__}: {_u18p_e}", flush=True)

        # ---- U15 Probe F (env-gated, layer 0 only): pre-AR W2 matmul output ----
        import os as _u15f_os
        _u15f_layer = (
            (int(self.layer_num) == 0) if hasattr(self, "layer_num") else True
        )
        # ---- U15 Probe G: L1 buffer enumeration at W2 dispatch time ----
        if _u15f_layer and _u15f_os.environ.get("SGLANG_TT_U15_PROBE_W2_L1", "0") == "1":
            try:
                _u15g_target = 0xa6700
                _u15g_devs = (
                    self.mesh_device.get_devices()
                    if hasattr(self.mesh_device, "get_devices")
                    else [self.mesh_device]
                )
                _u15g_bufs = ttnn._ttnn.reports.get_buffers(list(_u15g_devs))
                _u15g_near = [
                    (int(_b.address), _b.buffer_type, _b.buffer_layout,
                     _b.max_size_per_bank)
                    for _b in _u15g_bufs
                    if (_u15g_target - 0x10000) <= int(_b.address) <= (_u15g_target + 0x10000)
                ]
                _u15g_near.sort()
                print(
                    f"[U15_PROBE_G] W2-time near 0x{_u15g_target:x}: "
                    f"{len(_u15g_near)} bufs",
                    flush=True,
                )
                for _b in _u15g_near[:32]:
                    print(
                        f"[U15_PROBE_G]   addr=0x{_b[0]:x} bt={_b[1]} bl={_b[2]} "
                        f"sz_per_bank={_b[3]}",
                        flush=True,
                    )
                # Print GlobalCB info if accessible.
                try:
                    _u15g_gcb = self.prefetcher.global_cb if self.prefetcher is not None else None
                    if _u15g_gcb is not None:
                        _u15g_gcb_addr = _u15g_gcb.buffer_address
                        _u15g_gcb_size = _u15g_gcb.size
                        print(
                            f"[U15_PROBE_G] GlobalCB buffer_address={_u15g_gcb_addr() if callable(_u15g_gcb_addr) else _u15g_gcb_addr} "
                            f"size={_u15g_gcb_size() if callable(_u15g_gcb_size) else _u15g_gcb_size}",
                            flush=True,
                        )
                except Exception as _u15g_e:
                    print(f"[U15_PROBE_G] gcb info: {_u15g_e}", flush=True)
            except Exception as _u15g_e:
                print(f"[U15_PROBE_G] ERROR: {type(_u15g_e).__name__}: {_u15g_e}", flush=True)
        if _u15f_layer and _u15f_os.environ.get("SGLANG_TT_U15_PROBE_W2", "0") == "1":
            try:
                _u15f_ba = w2_out.buffer_address()
                _u15f_mc = w2_out.memory_config()
                _u15f_ss = _u15f_mc.shard_spec
                print(
                    f"[U15_PROBE_F] w2_out buffer_address=0x{_u15f_ba:x} "
                    f"shape={tuple(w2_out.shape)} "
                    f"memory_layout={_u15f_mc.memory_layout} "
                    f"shard_grid={_u15f_ss.grid if _u15f_ss else 'None'} "
                    f"shard_shape={_u15f_ss.shape if _u15f_ss else 'N/A'}",
                    flush=True,
                )
            except Exception as _u15f_e:
                print(f"[U15_PROBE_F] addr ERROR: {type(_u15f_e).__name__}: {_u15f_e}", flush=True)
            if _u15f_os.environ.get("SGLANG_TT_U15_PROBE_TOTORCH", "0") == "1":
                try:
                    _u15f_shards = ttnn.get_device_tensors(w2_out)
                    for _i, _sh in enumerate(_u15f_shards):
                        _u15f_t = ttnn.to_torch(_sh).float().cpu()
                        _u15f_flat = _u15f_t.flatten()
                        print(
                            f"[U15_PROBE_F_TOTORCH] w2_out shard={_i} "
                            f"shape={tuple(_u15f_t.shape)} "
                            f"max_abs={_u15f_flat.abs().max().item():.6e} "
                            f"nnz={int((_u15f_flat != 0).sum().item())}/{_u15f_flat.numel()} "
                            f"first16={_u15f_flat[:16].tolist()}",
                            flush=True,
                        )
                except Exception as _u15f_e:
                    print(
                        f"[U15_PROBE_F_TOTORCH] ERROR: {type(_u15f_e).__name__}: {_u15f_e}",
                        flush=True,
                    )

        # U19 — Python-side workaround test: insert a ttnn.fill(w2_out, 0)
        # BEFORE tt_all_reduce.  This forces w2_out's L1 to zero via a
        # captured trace op (no host sync needed).  Under zero weights,
        # tt_all_reduce's RS reader should now consistently see zero
        # and the output should become the deterministic U11 zero-weight
        # signature.  Under REAL weights, this DESTROYS W2's contribution
        # — diagnostic only.  Confirms (or refutes) "L1 0xa6700 is
        # genuinely stomped between W2 exit and RS reader".
        import os as _u19f_os
        if (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U19_FILL_W2_ZERO", "0") == "1"):
            try:
                # ttnn.fill needs (tensor, value).  Use 0 in same dtype.
                ttnn.fill(w2_out, 0)
            except Exception as _u19f_e:
                print(f"[U19_FILL_W2_ZERO] ERROR: {type(_u19f_e).__name__}: {_u19f_e}",
                      flush=True)

        # U19 — CLONE w2_out into a fresh L1 buffer before tt_all_reduce.
        # If the bug is "stomper writes to L1 0xa6700 between W2 exit and
        # RS reader", routing the RS through a CLONED tensor at a
        # DIFFERENT L1 address sidesteps the stomp at 0xa6700.  Preserves
        # W2's correct contribution — should fix the bug under REAL
        # weights, NOT just under zero weights.
        if (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U19_CLONE_W2", "0") == "1"):
            try:
                w2_out_orig = w2_out
                w2_out = ttnn.clone(w2_out_orig, memory_config=w2_out_orig.memory_config())
                ttnn.deallocate(w2_out_orig)
            except Exception as _u19c_e:
                print(f"[U19_CLONE_W2] ERROR: {type(_u19c_e).__name__}: {_u19c_e}",
                      flush=True)

        # U19 — COPY w2_out onto itself as a NO-OP DISPATCH BARRIER.
        # The op reads-then-writes the same L1 region — IF the stomper
        # fires BEFORE this copy, the copy preserves the stomped bytes
        # (data unchanged), and RS reader still sees stomp.  IF the
        # stomper fires DURING/AFTER this copy, the copy's NoC reads
        # happen first.  This isolates whether the FILL fix is about
        # (a) writing zero (data-destructive) or (b) just dispatching
        # any op between W2 and RS (timing).  Diagnostic.
        if (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U19_COPY_W2", "0") == "1"):
            try:
                ttnn.copy(w2_out, w2_out)
            except Exception as _u19cp_e:
                print(f"[U19_COPY_W2] ERROR: {type(_u19cp_e).__name__}: {_u19cp_e}",
                      flush=True)

        # U22 — Path B: sharding-preserving address relocation.
        # CLONE_W2 (U19) reallocates but `ttnn.clone` may perturb the
        # shard map.  These three variants attempt to move w2_out off
        # the cursed L1 0xa6700 slot while PRESERVING the original
        # sharding spec exactly:
        #   _RESHARD_W2:    ttnn.to_memory_config to same mem cfg
        #   _REALLOCATE_W2: ttnn.reallocate -> ttnn::move; defrag
        #                   primitive; usually preserves spec.
        #   _ASSIGN_W2:     ttnn.assign(w2_out, w2_out) identity assign
        # Each is mutually exclusive (try one at a time).  If any of
        # the three preserves coherence under REAL weights AND moves
        # the buffer address away from 0xa6700, we have a workaround.
        # A debug print emits the post-relocation buffer_address so we
        # can confirm relocation actually happened.
        _u22_dbg = _u19f_os.environ.get("SGLANG_TT_U22_PRINT_ADDR", "0") == "1"
        if (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U22_RESHARD_W2", "0") == "1"):
            try:
                _u22_orig = w2_out
                _u22_old_addr = w2_out.buffer_address() if _u22_dbg else 0
                w2_out = ttnn.to_memory_config(_u22_orig, _u22_orig.memory_config())
                if _u22_orig is not w2_out:
                    ttnn.deallocate(_u22_orig)
                if _u22_dbg:
                    _u22_new_addr = w2_out.buffer_address()
                    print(f"[U22_RESHARD_W2] old=0x{_u22_old_addr:x} "
                          f"new=0x{_u22_new_addr:x}", flush=True)
            except Exception as _u22r_e:
                print(f"[U22_RESHARD_W2] ERROR: {type(_u22r_e).__name__}: {_u22r_e}",
                      flush=True)
        elif (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U22_REALLOCATE_W2", "0") == "1"):
            try:
                _u22_old_addr = w2_out.buffer_address() if _u22_dbg else 0
                # NOTE: passing None lets move pick its own memcfg
                # (preserves input mem cfg).  Passing explicit
                # memory_config triggers move_sharded path.
                _u22_pass_mc = _u19f_os.environ.get(
                    "SGLANG_TT_U22_REALLOC_PASS_MC", "1") == "1"
                if _u22_pass_mc:
                    w2_out = ttnn.reallocate(w2_out, w2_out.memory_config())
                else:
                    w2_out = ttnn.reallocate(w2_out)
                if _u22_dbg:
                    _u22_new_addr = w2_out.buffer_address()
                    print(f"[U22_REALLOCATE_W2] old=0x{_u22_old_addr:x} "
                          f"new=0x{_u22_new_addr:x} "
                          f"pass_mc={_u22_pass_mc}", flush=True)
            except Exception as _u22a_e:
                print(f"[U22_REALLOCATE_W2] ERROR: {type(_u22a_e).__name__}: {_u22a_e}",
                      flush=True)
        elif (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U22_ASSIGN_W2", "0") == "1"):
            try:
                _u22_old_addr = w2_out.buffer_address() if _u22_dbg else 0
                # ttnn.assign(input, memory_config=...) creates a NEW
                # tensor with the given memory_config and copies data
                # in.  Unlike move_sharded (which has the per-core
                # chunk-size bug), this uses a proper allocator path.
                _u22_orig = w2_out
                w2_out = ttnn.assign(_u22_orig, memory_config=_u22_orig.memory_config())
                if _u22_orig is not w2_out:
                    ttnn.deallocate(_u22_orig)
                if _u22_dbg:
                    _u22_new_addr = w2_out.buffer_address()
                    print(f"[U22_ASSIGN_W2] old=0x{_u22_old_addr:x} "
                          f"new=0x{_u22_new_addr:x}", flush=True)
            except Exception as _u22s_e:
                print(f"[U22_ASSIGN_W2] ERROR: {type(_u22s_e).__name__}: {_u22s_e}",
                      flush=True)

        # U22 — Path B-double-prime: BARRIER CLONE.  Take ttnn.clone(w2_out)
        # but THROW AWAY the snapshot.  This forces a NoC READ of
        # 0xa6700 on every receiver core, which (hypothesis) flushes
        # / orders the stomp before subsequent ops run.
        if (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U22_BARRIER_CLONE_W2", "0") == "1"):
            try:
                _u22b_snap = ttnn.clone(w2_out, memory_config=w2_out.memory_config())
                ttnn.deallocate(_u22b_snap)  # discard
                if _u22_dbg:
                    print(f"[U22_BARRIER_CLONE_W2] w2_addr=0x{w2_out.buffer_address():x}",
                          flush=True)
            except Exception as _u22b_e:
                print(f"[U22_BARRIER_CLONE_W2] ERROR: {type(_u22b_e).__name__}: {_u22b_e}",
                      flush=True)

        # U22 — Path B-prime: SNAPSHOT_AND_RESTORE pattern.
        # ttnn.clone(w2_out) → snapshot at a different L1 slot
        # (not 0xa6700; presumably non-cursed).  Then immediately
        # ttnn.copy(snapshot, w2_out) — writes snapshot's bytes BACK
        # to w2_out's L1 0xa6700.  If the stomp fires BETWEEN the
        # clone (which captures correct data) and the copy (which
        # restores it), the restore wins and the RS reader sees the
        # correct value at 0xa6700.  Preserves w2_out's identity and
        # shard spec exactly — RS reader path unaffected.
        if (mode == Mode.DECODE
                and _u19f_os.environ.get("SGLANG_TT_U22_SNAPSHOT_RESTORE_W2", "0") == "1"):
            try:
                _u22sr_snap = ttnn.clone(w2_out, memory_config=w2_out.memory_config())
                # Now write snapshot back to w2_out's L1 in-place.
                ttnn.copy(_u22sr_snap, w2_out)
                ttnn.deallocate(_u22sr_snap)
                if _u22_dbg:
                    print(f"[U22_SNAPSHOT_RESTORE_W2] w2_addr=0x{w2_out.buffer_address():x}",
                          flush=True)
            except Exception as _u22sr_e:
                print(f"[U22_SNAPSHOT_RESTORE_W2] ERROR: {type(_u22sr_e).__name__}: {_u22sr_e}",
                      flush=True)

        # U28 Phase 1 — per-core W2 output address probe.  Confirms or
        # refutes hypothesis U28-α (per-core address mismatch): W2's PACK
        # writes to per-core differing L1 base addresses, but the RS
        # reader uses the bank-averaged address as the universal NoC
        # read target.  If the per-core addresses ENUMERATED HERE differ
        # from each other, U28-α is confirmed: receivers like (2,7) at
        # 0xa6700 vs (2,5) at 0xa4700 will be read by the RS reader at
        # the SAME (bank-averaged) L1 offset, so cores not at the
        # bank-averaged offset return stale pre-existing data.
        if (mode == Mode.DECODE
                and _u26a_os.environ.get("SGLANG_TT_U28_PER_CORE_PROBE", "0") == "1"):
            try:
                _u28_layer_num = int(getattr(self, "layer_num", -1))
                self._u28_iter = getattr(self, "_u28_iter", 0) + 1
                if self._u28_iter <= 144:
                    _u28_is_pc = bool(w2_out.is_per_core_allocated())
                    try:
                        _u28_bavg = w2_out.buffer_address()
                    except Exception:
                        _u28_bavg = -1
                    print(
                        f"[U28_W2_PER_CORE] iter={self._u28_iter} "
                        f"layer={_u28_layer_num} "
                        f"is_per_core_allocated={_u28_is_pc} "
                        f"bank_avg_addr={'NA' if _u28_bavg == -1 else hex(_u28_bavg)}",
                        flush=True,
                    )
                    # Enumerate cores from shard spec; query per-core address
                    # for each.  For NON per-core-allocated buffers, the
                    # nanobind call still returns the bank-averaged value, so
                    # a uniform result here doesn't prove same per-core L1.
                    _u28_mc = w2_out.memory_config()
                    _u28_ss = _u28_mc.shard_spec
                    _u28_grid = _u28_ss.grid if _u28_ss else None
                    _u28_cores = []
                    if _u28_grid is not None:
                        try:
                            for _u28_cr in _u28_grid.ranges():
                                for _ux in range(_u28_cr.start.x, _u28_cr.end.x + 1):
                                    for _uy in range(_u28_cr.start.y, _u28_cr.end.y + 1):
                                        _u28_cores.append(ttnn.CoreCoord(_ux, _uy))
                        except Exception:
                            pass
                    if not _u28_cores:
                        for _ux in range(0, 8):
                            for _uy in range(0, 8):
                                _u28_cores.append(ttnn.CoreCoord(_ux, _uy))
                    _u28_addrs = {}
                    for _u28_c in _u28_cores[:64]:
                        try:
                            _u28_a = w2_out.experimental_per_core_buffer_address(_u28_c)
                            _u28_addrs.setdefault(_u28_a, []).append(
                                (int(_u28_c.x), int(_u28_c.y))
                            )
                        except Exception:
                            pass
                    print(
                        f"[U28_W2_PER_CORE] iter={self._u28_iter} "
                        f"layer={_u28_layer_num} "
                        f"unique_addrs={len(_u28_addrs)} "
                        f"cores_probed={sum(len(_v) for _v in _u28_addrs.values())}",
                        flush=True,
                    )
                    for _u28_a in sorted(_u28_addrs.keys()):
                        _u28_cs = _u28_addrs[_u28_a]
                        print(
                            f"[U28_W2_PER_CORE]   addr=0x{_u28_a:x} "
                            f"count={len(_u28_cs)} cores={_u28_cs[:8]}",
                            flush=True,
                        )
                    if _u28_layer_num == 0 and self._u28_iter <= 4:
                        try:
                            _u28_grid_str = str(_u28_ss.grid) if _u28_ss else "None"
                            _u28_shape_str = str(_u28_ss.shape) if _u28_ss else "N/A"
                            _u28_orient = str(_u28_ss.orientation) if _u28_ss else "N/A"
                            print(
                                f"[U28_W2_PER_CORE_GRID] layer=0 "
                                f"mem_layout={_u28_mc.memory_layout} "
                                f"shard_grid={_u28_grid_str} "
                                f"shard_shape={_u28_shape_str} "
                                f"orient={_u28_orient}",
                                flush=True,
                            )
                        except Exception as _u28_ge:
                            print(f"[U28_W2_PER_CORE_GRID] ERROR: {_u28_ge}", flush=True)
                        # Also dump prefetcher.global_cb.sender_cores +
                        # receiver_cores (under the gathered matmul path,
                        # the W2 PACK kernel runs on sender cores; receiver
                        # cores get weights pushed in via NoC).  If w2_out's
                        # shard grid contains cores that are NOT in
                        # sender_cores, those cores' L1 slot for w2_out is
                        # allocated but NEVER written by W2 PACK — RS reader
                        # reads stale bytes.  This is the U28 root cause.
                        try:
                            _u28_pref = self.prefetcher
                            if _u28_pref is not None:
                                _u28_srm = getattr(_u28_pref, "sender_receiver_mapping", None)
                                if _u28_srm:
                                    _u28_senders = []
                                    _u28_receivers = set()
                                    for _entry in _u28_srm:
                                        try:
                                            _scc, _rcr_set = _entry
                                            _u28_senders.append((int(_scc.x), int(_scc.y)))
                                            for _rcr in _rcr_set.ranges():
                                                for _rx in range(_rcr.start.x, _rcr.end.x + 1):
                                                    for _ry in range(_rcr.start.y, _rcr.end.y + 1):
                                                        _u28_receivers.add((_rx, _ry))
                                        except Exception:
                                            pass
                                    _u28_recv_sorted = sorted(_u28_receivers)
                                    print(
                                        f"[U28_GCB_GRID] layer=0 "
                                        f"num_senders={len(_u28_senders)} "
                                        f"num_unique_receivers={len(_u28_recv_sorted)}",
                                        flush=True,
                                    )
                                    print(
                                        f"[U28_GCB_GRID]   senders={_u28_senders}",
                                        flush=True,
                                    )
                                    print(
                                        f"[U28_GCB_GRID]   receivers={_u28_recv_sorted}",
                                        flush=True,
                                    )
                                    # Compute set difference: w2_out shard
                                    # grid cores NOT in (senders ∪ receivers).
                                    _u28_w2_cores = set()
                                    if _u28_grid is not None:
                                        for _u28_cr in _u28_grid.ranges():
                                            for _ux in range(_u28_cr.start.x, _u28_cr.end.x + 1):
                                                for _uy in range(_u28_cr.start.y, _u28_cr.end.y + 1):
                                                    _u28_w2_cores.add((_ux, _uy))
                                    _u28_all_gcb = set(_u28_senders) | set(_u28_recv_sorted)
                                    _u28_in_w2_not_gcb = sorted(_u28_w2_cores - _u28_all_gcb)
                                    _u28_in_gcb_not_w2 = sorted(_u28_all_gcb - _u28_w2_cores)
                                    print(
                                        f"[U28_GCB_GRID]   w2_out_cores_not_in_gcb={_u28_in_w2_not_gcb}",
                                        flush=True,
                                    )
                                    print(
                                        f"[U28_GCB_GRID]   gcb_cores_not_in_w2={_u28_in_gcb_not_w2}",
                                        flush=True,
                                    )
                                else:
                                    print("[U28_GCB_GRID] no sender_receiver_mapping", flush=True)
                            else:
                                print("[U28_GCB_GRID] prefetcher=None", flush=True)
                        except Exception as _u28_in_e:
                            print(f"[U28_GCB_GRID] ERROR: {type(_u28_in_e).__name__}: {_u28_in_e}", flush=True)
            except Exception as _u28_e:
                print(
                    f"[U28_W2_PER_CORE] ERROR: {type(_u28_e).__name__}: {_u28_e}",
                    flush=True,
                )

        # U26 Phase A — buffer enumeration IMMEDIATELY before tt_all_reduce.
        # By the time we get here, w2_out has been deallocated/reallocated
        # by the U22 Path B variants (if active).  We want to see the L1
        # layout RIGHT BEFORE RS dispatches.
        if (mode == Mode.DECODE
                and _u26a_os.environ.get("SGLANG_TT_U26_ADJ_BUFFER_PROBE", "0") == "1"):
            try:
                _u26a_lo = 0xa6000
                _u26a_hi = 0xb0000
                _u26a_devs = (
                    self.mesh_device.get_devices()
                    if hasattr(self.mesh_device, "get_devices")
                    else [self.mesh_device]
                )
                _u26a_bufs = ttnn._ttnn.reports.get_buffers(list(_u26a_devs))
                _u26a_near = sorted(
                    [(int(_b.address), _b.buffer_type, _b.buffer_layout,
                      _b.max_size_per_bank)
                     for _b in _u26a_bufs
                     if _u26a_lo <= int(_b.address) <= _u26a_hi]
                )
                _u26a_w2 = w2_out.buffer_address()
                print(
                    f"[U26_ADJ_PRE_RS] iter={self._u26_iter} "
                    f"layer={getattr(self, 'layer_num', '?')} "
                    f"w2_out.addr=0x{_u26a_w2:x} "
                    f"bufs_in_range={len(_u26a_near)}",
                    flush=True,
                )
                for _b in _u26a_near[:32]:
                    print(
                        f"[U26_ADJ_PRE_RS]   addr=0x{_b[0]:x} bt={_b[1]} "
                        f"bl={_b[2]} sz_per_bank={_b[3]}",
                        flush=True,
                    )
            except Exception as _u26a_e:
                print(f"[U26_ADJ_PRE_RS] ERROR: {type(_u26a_e).__name__}: {_u26a_e}",
                      flush=True)

        # U28 Phase 2 — TEST: insert host-side sync between W2
        # (runs on receiver_sub_device) and tt_all_reduce → RS (runs on
        # worker_sub_device).  receiver_sub_device is NOT in the stall_group
        # so RS dispatched on worker sub-device starts before W2's PACK
        # on receiver_sub_device completes — this is the U28 cross-subdev
        # race hypothesis.  NOTE: host-side synchronize_device fails inside
        # trace capture (Event Synchronization not supported in trace).
        # Diagnostic-only; not a real fix path.
        if (mode == Mode.DECODE and self.prefetcher is not None
                and _u26a_os.environ.get("SGLANG_TT_U28_W2_RS_SYNC", "0") == "1"):
            try:
                ttnn.synchronize_device(
                    self.mesh_device,
                    sub_device_ids=[self.prefetcher.receiver_sub_device_id],
                )
            except Exception as _u28_se:
                # Expected to fail inside trace ('Event Synchronization is
                # not supported during trace capture').  Logged once for
                # diagnostic clarity.
                if not getattr(self, "_u28_sync_err_logged", False):
                    print(
                        f"[U28_W2_RS_SYNC] ERROR: {type(_u28_se).__name__}: {_u28_se}",
                        flush=True,
                    )
                    self._u28_sync_err_logged = True

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            sharded=(mode == Mode.DECODE),
            memory_config=self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out),
            rs_memory_config=self.model_config["MLP_RS_CONFIG"]["rs_memory_config"]
            if mode == Mode.DECODE
            else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
            chunks_per_sync=self.model_config["MLP_RS_CONFIG"]["chunks_per_sync"] if mode == Mode.DECODE else 10,
            num_workers_per_link=self.model_config["MLP_RS_CONFIG"]["num_workers_per_link"]
            if mode == Mode.DECODE
            else 2,
            subdevice_id=_u5_pref_subdev(self.prefetcher)
            if mode == Mode.DECODE
            else None,
        )

        # U26 Phase A — POST-RS buffer enumeration.  Show what was allocated
        # by tt_all_reduce / reduce_scatter_minimal_async.  Confirms whether
        # 0xa8700 ends up as the RS output / intermediate buffer slot.
        if (mode == Mode.DECODE
                and _u26a_os.environ.get("SGLANG_TT_U26_ADJ_BUFFER_PROBE", "0") == "1"):
            try:
                _u26a_lo = 0xa6000
                _u26a_hi = 0xb0000
                _u26a_devs = (
                    self.mesh_device.get_devices()
                    if hasattr(self.mesh_device, "get_devices")
                    else [self.mesh_device]
                )
                _u26a_bufs = ttnn._ttnn.reports.get_buffers(list(_u26a_devs))
                _u26a_near = sorted(
                    [(int(_b.address), _b.buffer_type, _b.buffer_layout,
                      _b.max_size_per_bank)
                     for _b in _u26a_bufs
                     if _u26a_lo <= int(_b.address) <= _u26a_hi]
                )
                _u26a_red = w2_out_reduced.buffer_address()
                print(
                    f"[U26_ADJ_POST_RS] iter={self._u26_iter} "
                    f"layer={getattr(self, 'layer_num', '?')} "
                    f"reduced.addr=0x{_u26a_red:x} "
                    f"bufs_in_range={len(_u26a_near)}",
                    flush=True,
                )
                for _b in _u26a_near[:32]:
                    print(
                        f"[U26_ADJ_POST_RS]   addr=0x{_b[0]:x} bt={_b[1]} "
                        f"bl={_b[2]} sz_per_bank={_b[3]}",
                        flush=True,
                    )
            except Exception as _u26a_e:
                print(f"[U26_ADJ_POST_RS] ERROR: {type(_u26a_e).__name__}: {_u26a_e}",
                      flush=True)
        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        if mode == Mode.DECODE:
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced
