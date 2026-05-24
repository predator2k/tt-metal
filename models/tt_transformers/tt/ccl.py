# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.modules.tt_ccl import get_num_links as get_common_num_links


def get_num_links(mesh_device, cluster_axis=None):
    """
    Get the number of available Ethernet links for CCL operations.

    This function queries the fabric control plane to determine the maximum number
    of usable links for collective communication operations.

    Args:
        mesh_device: The mesh device to query.
        cluster_axis: Optional cluster axis to query links for.
            - 0: Query links along the vertical axis (North-South direction).
            - 1: Query links along the horizontal axis (East-West direction).
            - None: Query links across all axes and return the minimum.

    Returns:
        int: The number of available links

    Example:
        >>> num_links = get_num_links(mesh_device)
        >>> num_links_axis0 = get_num_links(mesh_device, cluster_axis=0)
    """
    return get_common_num_links(mesh_device, cluster_axis)


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]

        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]

        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

        # cluster-axis-0, cluster-axis-1, no-cluster-axis
        for i in range(3):
            # double buffered semaphores
            for _ in range(2):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                )

                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                )

    def get_num_links(self, cluster_axis=None):
        """
        Get the number of available Ethernet links for CCL operations on this mesh device.

        Args:
            cluster_axis: Optional cluster axis to query links for.
                - 0: Query links along the vertical axis (North-South direction).
                - 1: Query links along the horizontal axis (East-West direction).
                - None: Query links across all axes and return the minimum.

        Returns:
            int: The number of available links (minimum 1).
        """
        return get_num_links(self.mesh_device, cluster_axis)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        semaphore_index = 2 if not cluster_axis else cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if not cluster_axis else cluster_axis
        current_idx = self.ag_semaphores_idx[semaphore_index]
        self.ag_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.ag_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if not cluster_axis else cluster_axis
        current_idx = self.rs_semaphores_idx[semaphore_index]
        self.rs_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.rs_semaphore_handles[semaphore_index][current_idx]


def tt_all_reduce(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=None,
    num_all_gather_links=None,
    topology=ttnn.Topology.Linear,
    memory_config=None,
    rs_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    sharded=False,
    dtype=ttnn.bfloat16,
    use_composite=False,
    chunks_per_sync=10,
    num_workers_per_link=2,
    subdevice_id=None,
):
    """
    Perform an all-reduce operation across devices in a mesh.

    Args:
        input_tensor: The input tensor to reduce.
        mesh_device: The mesh device to perform the operation on.
        tt_ccl: The TT_CCL instance for semaphore management.
        cluster_axis: The cluster axis for the reduction (default: 0).
        dim: The dimension to reduce along (default: 0).
        num_reduce_scatter_links: Number of links for reduce_scatter. If None, uses max available.
        num_all_gather_links: Number of links for all_gather. If None, uses max available.
        topology: The topology to use (default: ttnn.Topology.Linear).
        memory_config: Memory configuration for the output.
        sharded: Whether to use sharded memory config.
        dtype: Data type for CCL operations.
        use_composite: Whether to use composite reduce_scatter + all_gather.

    Returns:
        The reduced tensor.
    """
    # Skip CCL if single device or only 1 device on the target axis
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Auto-detect num_links if not provided
    if num_reduce_scatter_links is None:
        num_reduce_scatter_links = tt_ccl.get_num_links(cluster_axis)
    if num_all_gather_links is None:
        num_all_gather_links = tt_ccl.get_num_links(cluster_axis)

    # Ensure dim 0 and 1 are 1
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K: reduce_scatter
    if 1 in list(mesh_device.shape):
        if input_tensor.is_sharded() and not sharded:
            input_tensor_sharded = input_tensor
            input_tensor = ttnn.sharded_to_interleaved(input_tensor_sharded, ttnn.L1_MEMORY_CONFIG)
            input_tensor_sharded.deallocate(True)

        # Standalone no-prefetcher path: drain in-flight CCL, then reset the next
        # RS semaphore slot before use to prevent stale completion values.
        # subdevice_id is None iff prefetcher is None (callers pass
        # prefetcher.worker_sub_device_id or None), so we use that as the proxy.
        if subdevice_id is not None:
            # U15 Probe E: env-gated. Print input_tensor.buffer_address right
            # before reduce_scatter_minimal_async — this is the address the
            # consumer's reader kernel sees as `input_tensor_address`.
            import os as _u15e_os
            _u15e_call_ct_attr = "_u15_rs_call_ct"
            _u15e_global = globals()
            _u15e_global.setdefault(_u15e_call_ct_attr, 0)
            _u15e_global[_u15e_call_ct_attr] += 1
            _u15e_idx = _u15e_global[_u15e_call_ct_attr]
            if _u15e_os.environ.get("SGLANG_TT_U15_PROBE_RS", "0") == "1" and _u15e_idx <= 8:
                try:
                    _u15e_ba = input_tensor.buffer_address()
                    _u15e_mc = input_tensor.memory_config()
                    _u15e_ss = _u15e_mc.shard_spec
                    # Caller stack frame: where in the model is RS#k coming from?
                    import traceback as _u15e_tb
                    _u15e_stack = _u15e_tb.extract_stack(limit=6)[:-1]
                    _u15e_callers = " <- ".join(
                        f"{_f.name}@{_f.filename.split('/')[-1]}:{_f.lineno}"
                        for _f in _u15e_stack[::-1]
                    )
                    print(
                        f"[U15_PROBE_E] RS#{_u15e_idx} input_tensor.buffer_address=0x{_u15e_ba:x} "
                        f"shape={tuple(input_tensor.shape)} "
                        f"memory_layout={_u15e_mc.memory_layout} "
                        f"dim={dim} cluster_axis={cluster_axis}",
                        flush=True,
                    )
                    print(f"[U15_PROBE_E] RS#{_u15e_idx} stack: {_u15e_callers}", flush=True)
                    # Probe E2: PRE-RS to_torch. If nonzero, corruption is
                    # BEFORE the RS dispatch. If zero, RS's reader is wrong.
                    if _u15e_os.environ.get("SGLANG_TT_U15_PROBE_RS_TOTORCH", "0") == "1":
                        try:
                            _u15e_shards = ttnn.get_device_tensors(input_tensor)
                            for _i, _sh in enumerate(_u15e_shards):
                                _u15e_t = ttnn.to_torch(_sh).float().cpu()
                                _u15e_flat = _u15e_t.flatten()
                                print(
                                    f"[U15_PROBE_E2_TOTORCH] RS#{_u15e_idx} PRE-RS "
                                    f"shard={_i} shape={tuple(_u15e_t.shape)} "
                                    f"max_abs={_u15e_flat.abs().max().item():.6e} "
                                    f"nnz={int((_u15e_flat != 0).sum().item())}/{_u15e_flat.numel()}",
                                    flush=True,
                                )
                        except Exception as _u15e_ex:
                            print(
                                f"[U15_PROBE_E2_TOTORCH] RS#{_u15e_idx} ERROR: "
                                f"{type(_u15e_ex).__name__}: {_u15e_ex}",
                                flush=True,
                            )
                except Exception as _u15e_ex:
                    print(f"[U15_PROBE_E] ERROR: {type(_u15e_ex).__name__}: {_u15e_ex}", flush=True)
            # Prefetcher / trace-replay path: use async reduce_scatter.
            reduced = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor,
                persistent_output_buffers=None,
                dim=dim,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=num_reduce_scatter_links,
                memory_config=memory_config,
                intermediate_memory_config=rs_memory_config,
                topology=topology,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=2,
                subdevice_id=subdevice_id,
            )
            # Probe E (post): also dump the produced reduced tensor's address
            # AND its bytes (to_torch).  If reduced has nonzero data with zero
            # inputs, the reduce_scatter's reader read nonzero from L1 0xa6700.
            if _u15e_os.environ.get("SGLANG_TT_U15_PROBE_RS", "0") == "1" and _u15e_idx <= 8:
                try:
                    _u15e_ra = reduced.buffer_address()
                    _u15e_rmc = reduced.memory_config()
                    print(
                        f"[U15_PROBE_E] RS#{_u15e_idx} reduced.buffer_address=0x{_u15e_ra:x} "
                        f"shape={tuple(reduced.shape)} "
                        f"memory_layout={_u15e_rmc.memory_layout}",
                        flush=True,
                    )
                except Exception as _u15e_ex:
                    print(f"[U15_PROBE_E] post ERROR: {type(_u15e_ex).__name__}: {_u15e_ex}", flush=True)
                if _u15e_os.environ.get("SGLANG_TT_U15_PROBE_RS_TOTORCH", "0") == "1":
                    try:
                        _u15e_rshards = ttnn.get_device_tensors(reduced)
                        for _i, _sh in enumerate(_u15e_rshards):
                            _u15e_rt = ttnn.to_torch(_sh).float().cpu()
                            _u15e_rflat = _u15e_rt.flatten()
                            _u15e_first_nonzero_idx = None
                            if int((_u15e_rflat != 0).sum().item()) > 0:
                                _u15e_first_nonzero_idx = int((_u15e_rflat != 0).nonzero(as_tuple=True)[0][0].item())
                            print(
                                f"[U15_PROBE_E2_TOTORCH] RS#{_u15e_idx} POST-RS "
                                f"shard={_i} shape={tuple(_u15e_rt.shape)} "
                                f"max_abs={_u15e_rflat.abs().max().item():.6e} "
                                f"nnz={int((_u15e_rflat != 0).sum().item())}/{_u15e_rflat.numel()} "
                                f"first_nonzero_idx={_u15e_first_nonzero_idx} "
                                f"first16={_u15e_rflat[:16].tolist()}",
                                flush=True,
                            )
                    except Exception as _u15e_ex:
                        print(
                            f"[U15_PROBE_E2_TOTORCH] RS#{_u15e_idx} POST ERROR: "
                            f"{type(_u15e_ex).__name__}: {_u15e_ex}",
                            flush=True,
                        )
        else:
            # Standalone no-prefetcher path: use synchronous reduce_scatter to avoid
            # CCL semaphore slot reuse issue (see distributed_norm.py).
            reduced = ttnn.reduce_scatter(
                input_tensor,
                dim=dim,
                num_links=num_reduce_scatter_links,
                memory_config=memory_config,
                topology=topology,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=2,
            )
        input_tensor.deallocate(True)
        return reduced

    # TG: all_reduce
    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)  # typecast and to interleaved
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)  # to sharded

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:  # prefill
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    if not use_composite:
        gathered_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )

        if sharded:
            gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

        reduced_tensor = ttnn.experimental.fast_reduce_nc(
            gathered_tensor,
            dims=[dim],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG,
        )

        gathered_tensor.deallocate(True)
    else:
        input_mem_cfg = input_tensor.memory_config()

        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )

    # Reshape the reduced tensor to the original shape
    reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)

    return reduced_tensor


def tt_all_gather(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis,
    dim,
    num_links=None,
    memory_config=None,
    sharded=False,
    topology=ttnn.Topology.Linear,
    dtype=ttnn.bfloat16,
    subdevice_id=None,
):
    """
    Perform an all-gather operation across devices in a mesh.

    Args:
        input_tensor: The input tensor to gather.
        mesh_device: The mesh device to perform the operation on.
        tt_ccl: The TT_CCL instance for semaphore management.
        cluster_axis: The cluster axis for the gather operation.
        dim: The dimension to gather along.
        num_links: Number of links to use. If None, uses max available.
        memory_config: Memory configuration for the output.
        sharded: Whether to use sharded memory config.
        topology: The topology to use (default: ttnn.Topology.Linear).
        dtype: Data type for CCL operations.

    Returns:
        The gathered tensor.
    """
    # Skip CCL if single device or only 1 device on the target axis
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Auto-detect num_links if not provided
    if num_links is None:
        num_links = tt_ccl.get_num_links(cluster_axis)

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)  # typecast and to interleaved
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)  # to sharded

    if cluster_axis is None:
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )
    else:
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )
    input_tensor.deallocate(True)
    return gathered


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, tt_ccl, compute_kernel_config, num_links=None):
    """
    Perform distributed RMS normalization across devices.

    Args:
        inp: Input tensor.
        epsilon: Small value for numerical stability.
        gamma: Scale parameter.
        mesh_device: The mesh device.
        tt_ccl: The TT_CCL instance for semaphore management.
        compute_kernel_config: Compute kernel configuration.
        num_links: Number of links to use. If None, uses max available for cluster_axis=1.

    Returns:
        The normalized tensor.
    """
    # Auto-detect num_links if not provided
    if num_links is None:
        num_links = tt_ccl.get_num_links(cluster_axis=1)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))  # TODO: Figure out why we need this
    tt_stats_gathered = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dim=3,
        cluster_axis=1,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_stats.deallocate(True)

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats_gathered, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    tt_stats_gathered.deallocate(True)
    # inp.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp,
    epsilon,
    gamma,
    mesh_device,
    tt_ccl,
    ln_sharded_input_memcfg,
    ln_sharded_progcfg,
    ln_sharded_stats_memcfg,
    num_links=None,
):
    """
    Perform sharded distributed RMS normalization across devices.

    Args:
        inp: Input tensor.
        epsilon: Small value for numerical stability.
        gamma: Scale parameter.
        mesh_device: The mesh device.
        tt_ccl: The TT_CCL instance for semaphore management.
        ln_sharded_input_memcfg: Memory config for sharded input.
        ln_sharded_progcfg: Program config for sharded layernorm.
        ln_sharded_stats_memcfg: Memory config for sharded stats.
        num_links: Number of links to use. If None, uses max available for cluster_axis=1.

    Returns:
        The normalized tensor.
    """
    # Auto-detect num_links if not provided
    cluster_axis = 1
    if num_links is None:
        num_links = tt_ccl.get_num_links(cluster_axis)

    inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=ln_sharded_progcfg)

    # All gather stats
    tt_stats = ttnn.experimental.all_gather_async(
        tt_stats,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        memory_config=ln_sharded_stats_memcfg,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        epsilon=epsilon,
        weight=gamma,
        program_config=ln_sharded_progcfg,
        stats=tt_stats,
    )
    tt_stats.deallocate(True)

    return tt_out
