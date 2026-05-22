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
        # Note: self.prefetcher.all_worker_cores_range_set is read lazily in
        # forward() (not cached here) because prefetcher.init() may further
        # refine the set by subtracting receiver cores after model construction.

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
            if self.prefetcher is not None:
                # Prefetcher / trace-replay path: use async CCL with semaphore
                # management handled by the prefetcher sub-device.
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
                    subdevice_id=self.prefetcher.worker_sub_device_id,
                )
            else:
                # Standalone no-prefetcher path: use synchronous all_gather to avoid
                # the double-buffered semaphore slot reuse problem. In async mode,
                # CCL semaphore slots are shared within a single decode step: slot 0
                # is used by calls #1, #3, #5 (every other call). After call #1
                # completes, slot 0's semaphore is non-zero. When call #3 runs on
                # slot 0, the device sees non-zero and skips, producing a zero buffer.
                # Synchronous all_gather bypasses this by using internal barriers.
                num_links = (
                    self.args.model_config[self.ag_config_key]["num_links"]
                    if self.ag_config_key and mode == "decode"
                    else self.tt_ccl.get_num_links(1)
                )
                x = ttnn.all_gather(
                    x,
                    dim=3,
                    num_links=num_links,
                    memory_config=input_mem_cfg,
                    topology=self.args.ccl_topology(),
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
        # Tenstorrent-p1 (Vector 3): When prefetcher is ON, the unsharded path's
        # large per-core static CB (~1.26 MB) collides with prefetcher's GlobalCB
        # (~418-835 KB) at core (0,0) → "static CB clash with L1 buffer" crash
        # during prefill under DEFAULT manager (GlobalCB kept alive).  Skip the
        # BH force-unsharded when prefetcher is active so the sharded path runs;
        # if input is DRAM (BH attn_input routing) reshard to L1 WIDTH-sharded
        # first so the sharded factory's TT_FATAL("requires shard spec") doesn't fire.
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
            mode == Mode.DECODE
            and self.prefetcher is not None
            and x.memory_config().buffer_type == ttnn.BufferType.DRAM
        ):
            # Tenstorrent-p1: prefetcher DECODE path takes sharded norm. If input arrived
            # on DRAM (P3a.2 attn_input routing), reshard to L1 WIDTH-sharded.
            # DECODE-only: in PREFILL, x arrives DRAM after all_gather(DRAM) and should
            # stay DRAM — resharding with the non-rectangular all_worker_cores_range_set
            # would produce a non-rectangular shard spec that fails the LayerNorm validator
            # ("Sharded layernorm does not support non-rectangular core grids").
            # WIDTH (not HEIGHT) because the kernel TT_FATAL rejects HEIGHT_SHARDED
            # (layernorm_device_operation.cpp). For WIDTH-sharded, shard height MUST
            # equal the physical tensor height, so we use the full padded height
            # and shard only the width dimension across cores.
            # Use the safe rectangular worker grid (5,0)-(6,7) = 16 worker-only cores,
            # matching dynamic_worker_core_grid's return value, to avoid:
            #   (1) non-rectangular shard spec → LayerNorm bbox validator crash
            #   (2) shard grid spanning receiver/sender sub-devices → dispatch crash
            head_dim = x.padded_shape[-1]
            full_height = x.padded_shape[-2]
            worker_grid = self.prefetcher.dynamic_worker_core_grid(16)
            num_cores = worker_grid.num_cores()
            tile_w = ttnn.TILE_SIZE
            head_dim_tiles = (head_dim + tile_w - 1) // tile_w
            # Use as many cores as evenly divide the head_dim tiles.
            effective_cores = num_cores
            while effective_cores > 1 and head_dim_tiles % effective_cores != 0:
                effective_cores -= 1
            shard_w_tiles = head_dim_tiles // effective_cores
            shard_w = shard_w_tiles * tile_w
            # effective_cores may be < num_cores; use the first effective_cores from
            # the worker grid. Since worker_grid is a single rectangle (5,0)-(6,7),
            # take the first effective_cores cores row-wise (first few columns).
            if effective_cores == num_cores:
                shard_grid = worker_grid
            else:
                shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    worker_grid.ranges()[0].start,
                    effective_cores,
                    worker_grid,
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
        # Pass the worker-only CoreRangeSet so LayerNorm avoids sender-core L1 clash.
        # Read lazily: prefetcher.init() (called by switch_mode) may further refine
        # all_worker_cores_range_set after model construction.
        _norm_crs = (
            self.prefetcher.all_worker_cores_range_set
            if self.prefetcher is not None
            else None
        )
        x = self.norm(
            x, mode=mode, in_sharded=_in_sh, out_sharded=_out_sh, norm_config=norm_config,
            core_range_set=_norm_crs,
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
