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
        def as_sharded_tensor(name, type, dims):
            # First get the raw weight and transpose it
            raw_weight = torch_weight(name[:2])  # This is 2D: [H, W]
            # Pad if needed
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            # Make 4D: [1, 1, H, W] - CRITICAL for prefetcher to work correctly
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=type,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
                ),
                cache_file_name=cache_name(name),
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
                cache = cache_name(f"{name}_skip_ring")
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
                cache = cache_name(f"{name}_pdg")
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
            )
        ttnn.deallocate(w2_in)

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
