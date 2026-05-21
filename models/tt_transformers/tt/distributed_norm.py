# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm
from models.tt_transformers.tt.common import Mode


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl, prefetcher=None, TG=False, ag_config_key=None, enable_all_gather=True):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.ag_config_key = ag_config_key

        # Flag to control whether all_gather is performed after distributed norm (can be disabled when output should remain sharded)
        self.enable_all_gather = enable_all_gather

        if TG:
            core_grid_ln = (
                min(4, args.dim // 4 // 32 // 8),
                8,
            )  # dividing by 4 and 8 for num_cols and num_rows of mesh, and 32 for tile size
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG

    def forward(self, x, mode: Mode, norm_config=None):
        """Apply a norm, possibly gathering inputs if required."""

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None

        if self.TG:
            if mode == Mode.DECODE:
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.args.model_config[self.ag_config_key]["num_links"]
                if self.ag_config_key and mode == "decode"
                else self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=self.args.model_config[self.ag_config_key]["chunks_per_sync"]
                if self.ag_config_key and mode == "decode"
                else 10,
                num_workers_per_link=self.args.model_config[self.ag_config_key]["num_workers_per_link"]
                if self.ag_config_key and mode == "decode"
                else 2,
                num_buffers_per_channel=2,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        # P3a.2 patch: on single-chip the sharded program_config triggers a
        # std::bad_optional_access deep in ttnn.rms_norm — some field expected
        # in multi-device program_configs is empty. Force the unsharded
        # ttnn.rms_norm path on num_devices==1.
        # P3a.2 patch (Blackhole P300_X2 MUX): we route attn_input through DRAM
        # (see get_attn_input_mem_config), so the input is no longer sharded.
        # The sharded norm path then dereferences an empty std::optional
        # (memory_config/shard_spec) on DRAM input → crash. Force unsharded.
        # Tenstorrent-p1: When prefetcher is ON, the unsharded path's large
        # per-core static CB (~1.26 MB) collides with prefetcher's global CB
        # (~418 KB) at core (0,0) → "static CB clash with L1 buffer" crash. Skip
        # the BH force-unsharded for prefetcher case so the sharded path runs;
        # if input is DRAM we reshard it to L1 first so the sharded factory's
        # TT_FATAL("requires shard spec") doesn't fire.
        from models.common.utility_functions import is_blackhole as _is_bh_p3a2
        _force_unsharded = (
            not self.args.is_multichip
            or (_is_bh_p3a2() and self.prefetcher is None)
        )
        _in_sh = (mode == Mode.DECODE) and not _force_unsharded
        _out_sh = (mode == Mode.DECODE) and not _force_unsharded
        if _force_unsharded:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        elif (
            self.prefetcher is not None
            and x.memory_config().buffer_type == ttnn.BufferType.DRAM
        ):
            # Tenstorrent-p1: prefetcher path takes sharded norm. If input arrived
            # on DRAM (P3a.2 attn_input routing), reshard to L1 WIDTH-sharded.
            # WIDTH (not HEIGHT) because the kernel TT_FATAL rejects HEIGHT_SHARDED
            # (layernorm_device_operation.cpp:162-164). For WIDTH-sharded, shard
            # height MUST equal physical tensor height (tensor_spec.cpp:143), so
            # we use the full padded height and shard only the width across cores.
            head_dim = x.padded_shape[-1]
            full_height = x.padded_shape[-2]
            num_cores = self.prefetcher.all_worker_cores_range_set.num_cores()
            # Pick a divisor of num_cores that lets head_dim split cleanly into TILE-aligned chunks.
            tile_w = ttnn.TILE_SIZE
            head_dim_tiles = (head_dim + tile_w - 1) // tile_w
            # Use as many cores as evenly divide the head_dim tiles
            effective_cores = num_cores
            while effective_cores > 1 and head_dim_tiles % effective_cores != 0:
                effective_cores -= 1
            shard_w_tiles = head_dim_tiles // effective_cores
            shard_w = shard_w_tiles * tile_w
            # Build a CoreRangeSet that holds effective_cores from the worker grid
            worker_ranges = list(self.prefetcher.all_worker_cores_range_set.ranges())
            shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.prefetcher.worker_start_core,
                effective_cores,
                self.prefetcher.all_worker_cores_range_set,
                row_wise=True,
            )
            x = ttnn.to_memory_config(
                x,
                ttnn.create_sharded_memory_config(
                    shape=(full_height, shard_w),
                    core_grid=shard_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
        x = self.norm(
            x, mode=mode, in_sharded=_in_sh, out_sharded=_out_sh, norm_config=norm_config
        )

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode) and self.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        return x
