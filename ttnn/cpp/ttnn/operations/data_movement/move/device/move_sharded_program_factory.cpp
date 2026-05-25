// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_sharded_program_factory.hpp"

#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>

namespace ttnn::prim {

namespace {

// Helper: return the actual L1 base address of the buffer ON THE GIVEN CORE.
// For per-core-allocated buffers this differs across cores; for uniform
// allocations the value is identical to buffer.address() on every core.
//
// Why this exists (U23 fix):
//   move_sharded previously used buffer->address() (bank-averaged) as the
//   single chunk_size = dst_addr - src_addr for all cores.  When a sharded
//   buffer has per-core differing L1 base addresses, that delta is wrong on
//   any core whose true (dst - src) does not equal the bank-averaged delta.
//   The backward-copy kernel then reads stale data from an L1 region below
//   the actual source.  By threading the true per-core addresses through as
//   runtime args, the kernel can compute its own correct chunk_size on
//   every core.
inline tt::tt_metal::DeviceAddr per_core_address(
    const tt::tt_metal::Buffer& buffer, const CoreCoord& core) {
    if (tt::tt_metal::experimental::per_core_allocation::is_per_core_allocation(buffer)) {
        return tt::tt_metal::experimental::per_core_allocation::get_per_core_address(buffer, core);
    }
    // Uniform sharded path: every core has the same L1 base.
    return buffer.address();
}

}  // namespace

MoveShardedProgramFactory::cached_program_t MoveShardedProgramFactory::create(
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto shard_spec = input.shard_spec().value();
    const auto shard_shape = shard_spec.shape;
    const auto shard_grid = shard_spec.grid;
    const auto& input_shape = input.logical_shape();
    const DataType input_dtype = input.dtype();
    const Layout input_layout = input.layout();
    TT_FATAL(
        input_layout == output.layout() && input_dtype == output.dtype() &&
            shard_shape == output.shard_spec().value().shape && input_shape == output.logical_shape(),
        "Error");
    const uint32_t src_cb_sharded = tt::CBIndex::c_0;
    const uint32_t dst_cb_sharded = tt::CBIndex::c_1;

    const uint32_t total_size_bytes = input.buffer()->aligned_size_per_bank();
    const uint32_t page_size_bytes = input.buffer()->aligned_page_size();

    CircularBufferConfig src_cb_sharded_config =
        CircularBufferConfig(total_size_bytes, {{src_cb_sharded, cb_data_format}})
            .set_page_size(src_cb_sharded, page_size_bytes);
    src_cb_sharded_config.set_globally_allocated_address(*input.buffer());
    const CBHandle src_sharded_cb = tt::tt_metal::CreateCircularBuffer(program, shard_grid, src_cb_sharded_config);

    CircularBufferConfig dst_cb_sharded_config =
        CircularBufferConfig(total_size_bytes, {{dst_cb_sharded, cb_data_format}})
            .set_page_size(dst_cb_sharded, page_size_bytes);
    dst_cb_sharded_config.set_globally_allocated_address(*output.buffer());
    const CBHandle dst_sharded_cb = tt::tt_metal::CreateCircularBuffer(program, shard_grid, dst_cb_sharded_config);

    TT_FATAL(
        input.buffer()->alignment() == output.buffer()->alignment(),
        "Expected input buffer alignment ({} B) and output buffer alignment ({} B) to be equal",
        input.buffer()->alignment(),
        output.buffer()->alignment());
    const uint32_t alignment = input.buffer()->alignment();

    std::vector<uint32_t> reader_compile_time_args = {src_cb_sharded, dst_cb_sharded};
    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp",
        shard_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = reader_compile_time_args});

    // Per-core runtime args: thread the ACTUAL per-core src/dst L1 addresses so
    // the kernel computes its own correct chunk_size on every core (U23 fix).
    //
    // Runtime arg layout (per core):
    //   [0] total_size_bytes
    //   [1] num_chunks
    //   [2] move_chunk_size_bytes  (per-core dst - src delta)
    //   [3] remainder_chunk_size_bytes
    //   [4] src_addr_override      (per-core actual src L1 base; 0 = use CB)
    //   [5] dst_addr_override      (per-core actual dst L1 base; 0 = use CB)
    const auto cores = corerange_to_cores(shard_grid, std::nullopt, true);
    std::vector<std::vector<uint32_t>> per_core_runtime_args;
    per_core_runtime_args.reserve(cores.size());
    for (const auto& core : cores) {
        const uint32_t src_addr = static_cast<uint32_t>(per_core_address(*input.buffer(), core));
        const uint32_t dst_addr = static_cast<uint32_t>(per_core_address(*output.buffer(), core));
        // dst_addr must be strictly greater than src_addr for backward copy;
        // this invariant comes from the caller picking the "move" path.
        TT_FATAL(
            dst_addr > src_addr,
            "move_sharded backward-copy requires dst_addr ({:#x}) > src_addr ({:#x}) on core ({}, {})",
            dst_addr,
            src_addr,
            core.x,
            core.y);
        const uint32_t move_chunk_size_bytes = dst_addr - src_addr;
        TT_FATAL(
            move_chunk_size_bytes % alignment == 0,
            "Expected chunk size bytes to move to be {} byte aligned on core ({}, {}), got {:#x}",
            alignment,
            core.x,
            core.y,
            move_chunk_size_bytes);
        const uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
        const uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;
        per_core_runtime_args.push_back({
            total_size_bytes,
            num_chunks,
            move_chunk_size_bytes,
            remainder_chunk_size_bytes,
            src_addr,
            dst_addr,
        });
    }
    tt::tt_metal::SetRuntimeArgs(program, kernel_id, cores, per_core_runtime_args);

    return {
        std::move(program),
        MoveShardedProgramFactory::shared_variables_t{
            .kernel_id = kernel_id,
            .src_sharded_cb = src_sharded_cb,
            .dst_sharded_cb = dst_sharded_cb,
            .total_size_bytes = total_size_bytes,
            .cores = cores}};
}

void MoveShardedProgramFactory::override_runtime_arguments(
    MoveShardedProgramFactory::cached_program_t& cached_program,
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    Program& program = cached_program.program;
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.src_sharded_cb, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.dst_sharded_cb, *dst_buffer);

    const uint32_t alignment = src_buffer->alignment();
    const uint32_t total_size_bytes = cached_program.shared_variables.total_size_bytes;

    // Recompute per-core runtime args on cache hit; addresses may have moved
    // (e.g., a freed-and-reallocated output buffer).  Uniform sharded path
    // produces identical args on every core, matching pre-U23 behavior.
    for (const auto& core : cached_program.shared_variables.cores) {
        const uint32_t src_addr = static_cast<uint32_t>(per_core_address(*src_buffer, core));
        const uint32_t dst_addr = static_cast<uint32_t>(per_core_address(*dst_buffer, core));
        TT_FATAL(
            dst_addr > src_addr,
            "move_sharded backward-copy requires dst_addr ({:#x}) > src_addr ({:#x}) on core ({}, {})",
            dst_addr,
            src_addr,
            core.x,
            core.y);
        const uint32_t move_chunk_size_bytes = dst_addr - src_addr;
        TT_FATAL(
            move_chunk_size_bytes % alignment == 0,
            "Expected chunk size bytes to move to be {} byte aligned on core ({}, {}), got {:#x}",
            alignment,
            core.x,
            core.y,
            move_chunk_size_bytes);
        const uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
        const uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;
        std::vector<uint32_t> new_runtime_args = {
            total_size_bytes,
            num_chunks,
            move_chunk_size_bytes,
            remainder_chunk_size_bytes,
            src_addr,
            dst_addr,
        };
        SetRuntimeArgs(program, cached_program.shared_variables.kernel_id, core, new_runtime_args);
    }
}

}  // namespace ttnn::prim
