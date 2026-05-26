// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"
#include "internal/mod_div_lib.h"

#ifdef SFPU_ACTIVATION
#include "bmm_fused_activation.hpp"
#endif

#if defined(SGLANG_TT_PREFETCHER_LLK_PROBE) || defined(SGLANG_TT_PREFETCHER_PACK_PROBE) || \
    defined(SGLANG_TT_PREFETCHER_CONSUMER_PROBE) || defined(SGLANG_TT_U18_PACK_PROBE)
// U12 / U13 / U14 / U18: LLK / PACK / CONSUMER debug probes.  Loaded only
// when an env-gated SGLANG_TT_PREFETCHER_*_PROBE / SGLANG_TT_U18_PACK_PROBE
// compile-time define is propagated by the program factory (gated by the
// corresponding environment variable, only on the gathered/use_global_cb
// path so canonical builds are untouched).
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#include "api/debug/dprint_tensix_unpack.h"
#include "api/debug/dprint_tensix_pack.h"
#include "api/debug/dprint_tile.h"  // CB_WR_PTR / cb_addr_shift
#endif

#if defined(SGLANG_TT_U35_KERNEL_OFFSET_PROBE)
// U35 — verify the offset RT arg actually reaches the kernel.  Prints
// the value once per worker (first batch iter, UNPACK risc) so we can
// correlate with the Python-set SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES
// for each tensor.  No-op unless the env-gated define is propagated
// through the matmul factory (mm_kernel_defines).
#ifndef SGLANG_TT_DPRINT_INCLUDED
#define SGLANG_TT_DPRINT_INCLUDED
#include "api/debug/dprint.h"
#endif
#endif

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    experimental::CircularBuffer mm_partials_cb(mm_partials_cb_id);
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    mm_partials_cb.wait_front(out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    mm_partials_cb.pop_front(out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr;
}

FORCE_INLINE void update_local_cb_rd_ptr(uint32_t cb_id, uint32_t val) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    local_cb.fifo_rd_ptr = val;
}

FORCE_INLINE uint32_t get_local_cb_start_addr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    uint32_t fifo_limit = local_cb.fifo_limit;
    uint32_t fifo_start_addr = fifo_limit - fifo_size;
    return fifo_start_addr;
}

FORCE_INLINE bool is_tensor_split(uint32_t cb_id, uint32_t tensor_size_bytes) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t fifo_limit = local_cb.fifo_limit;
    bool split = (fifo_limit - fifo_rd_ptr) < tensor_size_bytes / L1_ALIGNMENT;
    return split;
}

FORCE_INLINE void calculate_next_block_index_and_update_rd_ptr(
    uint32_t cb_id,
    uint32_t num_blocks,
    uint32_t block_size_bytes,
    uint32_t curr_block_index,
    uint32_t cb_start_addr,
    uint32_t rd_ptr_start_addr,
    bool tensor_split,
    uint32_t* updated_block_index,
    uint32_t* updated_rd_ptr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t next_block_index = curr_block_index + 1;
    uint32_t next_fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t block_size_bytes_aligned = block_size_bytes / L1_ALIGNMENT;
    bool reach_limit = local_cb.fifo_rd_ptr == local_cb.fifo_limit;
    bool last_block = curr_block_index == (num_blocks - 1);
    if (tensor_split) {
        if (reach_limit) {
            local_cb.fifo_rd_ptr = cb_start_addr;
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr = cb_start_addr + block_size_bytes_aligned;
            }
        } else {
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr += block_size_bytes_aligned;
            }
        }
    } else {
        if (last_block) {
            next_block_index = 0;
            next_fifo_rd_ptr = rd_ptr_start_addr;
        } else {
            next_fifo_rd_ptr += block_size_bytes_aligned;
        }
    }
    *updated_block_index = next_block_index;
    *updated_rd_ptr = next_fifo_rd_ptr;
}

FORCE_INLINE void update_rd_ptr_to_ring_index(
    uint32_t cb_id, uint32_t block_size_bytes, uint32_t ring_index, bool tensor_split) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);

    if (tensor_split) {
        if ((local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT) >= local_cb.fifo_limit) {
            uint32_t fifo_size = local_cb.fifo_size;
            uint32_t fifo_limit = local_cb.fifo_limit;
            uint32_t fifo_start_addr = fifo_limit - fifo_size;
            uint32_t fifo_size_skip_bytes = local_cb.fifo_rd_ptr - fifo_start_addr;
            local_cb.fifo_rd_ptr =
                fifo_start_addr +
                (fifo_size_skip_bytes + ring_index * block_size_bytes / L1_ALIGNMENT) % local_cb.fifo_size;

        } else {
            local_cb.fifo_rd_ptr = local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT;
        }
    } else {
        local_cb.fifo_rd_ptr = local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT;
    }
}

// Named CB arg lookup tables for batch-indexed output and partials CBs.
// The factory emits "cb_mm_out_0" .. "cb_mm_out_N" and "cb_mm_partials_0" .. "cb_mm_partials_N"
// as named compile-time args. These tables let fill_named_cb_array resolve them by index.
constexpr const char* mm_out_cb_names[] = {
    "cb_mm_out_0",
    "cb_mm_out_1",
    "cb_mm_out_2",
    "cb_mm_out_3",
    "cb_mm_out_4",
    "cb_mm_out_5",
    "cb_mm_out_6",
    "cb_mm_out_7",
    "cb_mm_out_8",
    "cb_mm_out_9",
    "cb_mm_out_10",
    "cb_mm_out_11",
    "cb_mm_out_12",
    "cb_mm_out_13",
    "cb_mm_out_14",
    "cb_mm_out_15",
};
constexpr const char* mm_partials_cb_names[] = {
    "cb_mm_partials_0",
    "cb_mm_partials_1",
    "cb_mm_partials_2",
    "cb_mm_partials_3",
    "cb_mm_partials_4",
    "cb_mm_partials_5",
    "cb_mm_partials_6",
    "cb_mm_partials_7",
    "cb_mm_partials_8",
    "cb_mm_partials_9",
    "cb_mm_partials_10",
    "cb_mm_partials_11",
    "cb_mm_partials_12",
    "cb_mm_partials_13",
    "cb_mm_partials_14",
    "cb_mm_partials_15",
};

template <uint32_t N>
constexpr std::array<uint32_t, N> fill_named_cb_array(const char* const* names) {
    std::array<uint32_t, N> arr{};
    for (uint32_t i = 0; i < N; ++i) {
        arr[i] = get_named_compile_time_arg_val(names[i]);
    }
    return arr;
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                                  // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t in1_tensor_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(8);           // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(9);               // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output
    constexpr bool in1_is_dram_interleaved = get_compile_time_arg_val(16);     // in1 is in dram
    constexpr bool in1_is_dram_sharded = get_compile_time_arg_val(17);
    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t in2_cb_id = get_named_compile_time_arg_val("cb_in2");
    constexpr uint32_t sync_cb = get_named_compile_time_arg_val("cb_sync");
    constexpr uint32_t sync_cb2 = get_named_compile_time_arg_val("cb_sync2");

#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type =
        static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
    constexpr uint32_t activation_param0 = get_named_compile_time_arg_val("activation_param0");
    constexpr uint32_t activation_param1 = get_named_compile_time_arg_val("activation_param1");
    constexpr uint32_t activation_param2 = get_named_compile_time_arg_val("activation_param2");
#endif

    experimental::CircularBuffer in1_cb(in1_cb_id);
    experimental::CircularBuffer sync_buf(sync_cb);
    experimental::CircularBuffer sync2_buf(sync_cb2);

    constexpr std::array<uint32_t, batch> mm_out_cb_ids = fill_named_cb_array<batch>(mm_out_cb_names);
    constexpr std::array<uint32_t, batch> mm_partials_cb_ids = fill_named_cb_array<batch>(mm_partials_cb_names);

    constexpr uint32_t ring_size = num_blocks;
    constexpr bool in1_is_dram = in1_is_dram_interleaved || in1_is_dram_sharded;

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE || core_type == (uint32_t)CORE_TYPE::HOP_CORE) {
        return;
    }
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* unpadded_in0_shard_widths_in_tiles = (uint32_t*)get_arg_addr(rt_args_idx);
    rt_args_idx += ring_size;

#ifdef SGLANG_TT_U32_GCB_OFFSET
    // U32 — per-tensor GCB byte offset.  The prefetcher writes tensors
    // SEQUENTIALLY into the GlobalCB starting at fifo_start (tensor 0 at
    // offset 0, tensor 1 at offset = num_blocks * block_size_per_receiver[0],
    // ...).  But every matmul kernel's `setup_local_cb_read_write_interfaces`
    // resets the local CB rd_ptr to `fifo_start` at kernel entry — so without
    // this offset, every matmul reads from offset 0 (tensor 0's bytes),
    // regardless of which tensor it actually owns.  Under zero weights every
    // offset reads 0 → x*0=0 is correct.  Under real weights, mis-aligned
    // reads decode other tensors' BFP4/BFP8 bytes as huge magnitudes
    // (U7's 2^60-2^109 signature; U30 PACK-side 86% NaN/Inf).
    //
    // The byte offset is set by the Python caller (mlp.py / attention.py)
    // via env var SGLANG_TT_U32_GCB_TENSOR_OFFSET_BYTES at ttnn.linear
    // program-create time; the factory bakes the value into this kernel's
    // runtime args and updates it on every cache hit via
    // override_gather_in0_program_parameters.
    uint32_t u32_tensor_offset_bytes = get_arg_val<uint32_t>(rt_args_idx++);
#endif

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

#ifdef SFPU_ACTIVATION
    ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks > 1 && (out_block_num_tiles / out_subblock_num_tiles) > 1;

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_ids[0], in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

#ifdef SGLANG_TT_PREFETCHER_DST_ZERO
    // U9: Explicit DST zero before any matmul accumulation. mm_block_init's
    // llk_math_pack_sync_init only resets the dest_offset_id / section base
    // pointers; it does NOT issue a ZEROACC. If DST sections still hold values
    // from a prior program (e.g., a layout/reshard before this matmul), the
    // first matmul_block(... idst=0 ...) accumulates onto stale state. The
    // gathered (prefetcher) path triggers this because the GlobalCB program
    // layout schedules an earlier program on the same DST without the normal
    // pack_dest_section_done epilogue running on these cores.
    MATH((ckernel::zeroacc()));
#endif

    for (uint32_t b = 0; b < batch; b++) {
#ifdef ENABLE_GLOBAL_CB
        uint32_t in1_cb_start_addr = 0;
        uint32_t in1_rd_ptr_start_addr = 0;
        uint32_t curr_in1_block_index = 0;
        bool in1_tensor_split = 0;
        uint32_t next_in1_block_index;
        uint32_t next_in1_rd_ptr_addr;

    #ifndef SGLANG_TT_PREFETCHER_BYPASS_GCB_INIT
        // U10 Block A: top-of-batch init — captures the START of THIS matmul's
        // view into GlobalCB, the ring index, the tensor_split path, and
        // advances rd_ptr to this core's ring-slot. Bypassing this leaves
        // rd_ptr wherever the previous matmul's Block D left it; the locals
        // (in1_rd_ptr_start_addr/in1_tensor_split) stay 0 so Block D's
        // restore-and-advance will land on the start of CB instead of the
        // saved start. This bypass intentionally produces wrong outputs IF
        // the kernel still runs; the key signal is whether the garbage
        // signature changes class.
        UNPACK((in1_cb_start_addr = get_local_cb_start_addr(in1_cb_id)));
    #ifdef SGLANG_TT_U32_GCB_OFFSET
        // U32 — instead of letting in1_rd_ptr_start_addr be wherever
        // setup_local_cb_read_write_interfaces last reset it (= fifo_start),
        // set it to fifo_start + tensor_offset_bytes so that THIS matmul
        // reads from the GCB region where the prefetcher actually wrote
        // this tensor's data.  The fifo_rd_ptr is stored in SHIFTED L1
        // units (1 unit = L1_ALIGNMENT bytes); add the offset in the
        // same shifted units.  ring_idx advancement (below) then steps
        // by ring_idx * block_size_bytes / L1_ALIGNMENT from this base.
        #ifdef SGLANG_TT_U35_KERNEL_OFFSET_PROBE
        // U35 — print the offset RT arg ACTUALLY received by the kernel,
        // along with the CB's fifo_start (in shifted units) and the
        // computed new rd_ptr.  Once per batch iter (b==0); UNPACK side
        // only (a single risc print per worker core per launch).  This
        // proves whether the Python-side env var → factory RT arg →
        // kernel reception path is intact, OR whether the kernel sees
        // 0 / a stale value despite the Python side passing the right
        // value.
        if (b == 0) {
            UNPACK((DPRINT << "[U35_KERNEL_OFFSET ring_idx=" << ring_idx
                           << " offset_bytes=" << u32_tensor_offset_bytes
                           << " in1_block_size_bytes=" << in1_block_size_bytes
                           << " in1_cb_start_shifted=" << in1_cb_start_addr
                           << " new_rd_ptr_shifted="
                           << (in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT)
                           << "]" << ENDL()));
        }
        #endif
        UNPACK((update_local_cb_rd_ptr(
            in1_cb_id,
            in1_cb_start_addr + u32_tensor_offset_bytes / L1_ALIGNMENT)));
    #endif
        UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
        UNPACK((curr_in1_block_index = ring_idx));
        UNPACK((in1_tensor_split = is_tensor_split(in1_cb_id, in1_tensor_size_bytes)));
        UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_idx, in1_tensor_split)));
    #else
        // Even when bypassed, init the locals so Block D doesn't read uninit.
        // Re-capture start addr but skip the ring-index advance and tensor_split
        // probe — i.e. force the kernel to assume contiguous (no-split) layout
        // and to start reading from wherever the CB rd_ptr currently sits.
        UNPACK((in1_cb_start_addr = get_local_cb_start_addr(in1_cb_id)));
        UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
        UNPACK((curr_in1_block_index = ring_idx));
        // in1_tensor_split stays false (constant) — disables wrap-around in
        // calculate_next_block_index_and_update_rd_ptr / update_rd_ptr_to_ring_index.
    #endif
#endif
        const uint32_t mm_out_cb_id = mm_out_cb_ids[b];
        const uint32_t mm_partials_cb_id = mm_partials_cb_ids[b];
        experimental::CircularBuffer mm_out_cb(mm_out_cb_id);
        experimental::CircularBuffer mm_partials_cb(mm_partials_cb_id);

        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

#ifdef PACK_RELU
        // for each batch we start we relu disabled so that intermediate results are not relu'd
        if constexpr (batch > 1) {
            PACK((llk_pack_relu_config(ReluType::NO_RELU)));
        }
#endif

        if constexpr (batch > 1) {
            PACK((pack_reconfig_data_format(mm_partials_cb_id)));
        }

        // Wait to receive in1
        sync2_buf.wait_front(1);
        sync2_buf.pop_front(1);

        for (uint32_t block = 0; block < num_blocks; block++) {
            const uint32_t curr_ring_idx = (ring_idx + block) % ring_size;
            uint32_t unpadded_in0_block_w = unpadded_in0_shard_widths_in_tiles[curr_ring_idx];

            // Wait for in1 block
            if constexpr (in1_is_dram) {
                in1_cb.wait_front(in1_block_num_tiles);
            }

            const uint32_t input0_cb_id = block == 0 ? in0_cb_id : in2_cb_id;
            experimental::CircularBuffer input0_cb(input0_cb_id);
            bool last_out = block == (num_blocks - 1);
// Configure packer once for pack out without Bias
#if not defined FUSE_BIAS and defined PACK_RELU
            if (last_out) {
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
            }
#endif

            // Wait to receive in0 block
            if (block == 0) {
                input0_cb.reserve_back(in0_block_num_tiles);
                input0_cb.push_back(in0_block_num_tiles);
            }
            input0_cb.wait_front(in0_block_num_tiles);

#ifdef ENABLE_GLOBAL_CB
    #ifndef SGLANG_TT_PREFETCHER_BYPASS_GCB_BLOCK
            // U10 Block B: per-block-start — computes next_in1_block_index and
            // next_in1_rd_ptr_addr that Block C consumes at end of this block.
            // Bypassing Block B alone would leave the next_* locals uninitialized
            // for Block C to write — so we also bypass Block C in lockstep.
            UNPACK((calculate_next_block_index_and_update_rd_ptr(
                in1_cb_id,
                num_blocks,
                in1_block_size_bytes,
                curr_in1_block_index,
                in1_cb_start_addr,
                in1_rd_ptr_start_addr,
                in1_tensor_split,
                &next_in1_block_index,
                &next_in1_rd_ptr_addr)));
    #else
            // Force a trivial linear advance: next = curr + 1; next_rd_ptr = curr + block_size.
            // No tensor_split wrap, no num_blocks check. Reads only consecutive
            // blocks starting at the current rd_ptr. This will read past valid
            // L1 if num_blocks > 1, but isolates whether the wrap arithmetic
            // is the bug source.
            UNPACK((next_in1_block_index = curr_in1_block_index + 1));
            UNPACK((next_in1_rd_ptr_addr =
                        get_local_cb_rd_ptr(in1_cb_id) + in1_block_size_bytes / L1_ALIGNMENT));
    #endif
#endif

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
#ifdef ENABLE_GLOBAL_CB
                int in1_index_subblock_offset = 0;
#else
                // This should always be 0 when reading in1 from DRAM
                int in1_index_subblock_offset = in1_is_dram ? 0 : in1_block_num_tiles * (curr_ring_idx);
#endif
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();
                    if (enable_reload) {
                        reload_from_cb_to_dst(
                            input0_cb_id,
                            in1_cb_id,
                            mm_partials_cb_id,
                            in1_transpose_tile,
                            out_subblock_num_tiles,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                    }

#ifndef SKIP_COMPUTE
                    // Compute output sub-block
                    uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
                    uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                    uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                    // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                    for (uint32_t inner_dim_idx = 0; inner_dim_idx < unpadded_in0_block_w; ++inner_dim_idx) {
                        // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                        // accumulation is done by iterating matmul_block across inner dim
                        // in0_block_w is passed as innder dim (kt) to matmul_block, internally used to stride in0
                        matmul_block(
                            input0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            in1_transpose_tile,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
#ifdef SGLANG_TT_PREFETCHER_LLK_PROBE
                        // U13 Part 2 (widened) — verify U12's probe fired on EVERY
                        // gathered ELF.  U12 v2's gate
                        // `in0_subblock==0 && in1_subblock==0 && inner_dim_idx==0`
                        // could miss an ELF if that ELF only ever passes through
                        // non-zero subblock indices on the static-cached path.  U13
                        // widens to fire on the FIRST iteration the kernel sees,
                        // regardless of subblock/inner indices, then drops the
                        // remaining budget on first matches per (b, block) so we
                        // still capture later blocks if budget allows.  Adds the
                        // matmul-shape "ELF tag" (built from CT args) so we can
                        // de-duplicate post-hoc by ELF instead of by core.
                        {
                            constexpr uint32_t u13_elf_tag =
                                (in0_block_w * 1u) ^
                                (in0_num_subblocks * 131u) ^
                                (in1_num_subblocks * 17u) ^
                                (num_blocks * 7919u) ^
                                (out_subblock_h * 31u) ^
                                (out_subblock_w * 257u) ^
                                (batch * 65537u);
                            static uint32_t u13_budget = 16;
                            if (u13_budget > 0) {
                                u13_budget--;
                                MATH((
                                    {
                                        uint32_t dst_rd[8];
                                        ckernel::dbg_get_array_row(
                                            ckernel::dbg_array_id::DEST, 0, dst_rd);
                                        DPRINT << "[U13_DST elf=0x" << HEX()
                                               << u13_elf_tag
                                               << DEC() << " ring=" << ring_idx
                                               << " b=" << b
                                               << " blk=" << block
                                               << " is0=" << in0_subblock
                                               << " is1=" << in1_subblock
                                               << " kk=" << inner_dim_idx
                                               << " r0=";
                                        bool any_nonzero = false;
                                        for (int i = 0; i < 8; ++i) {
                                            DPRINT << "0x" << HEX() << dst_rd[i] << " ";
                                            if (dst_rd[i] != 0) any_nonzero = true;
                                        }
                                        DPRINT << (any_nonzero ? "NONZERO" : "zero")
                                               << "]" << ENDL();
                                    }
                                ));
                            }
                        }
#endif
                        in0_index++;                  // stride right by 1
                        in1_index += in1_per_core_w;  // to stride down by 1 need to stride by in_per_core_w (should be
                                                      // called in1_block_w)
                    }

#endif  // SKIP_COMPUTE

                    if (last_out) {
                        if constexpr (untilize_out) {
                            pack_untilize_dest_init<out_subblock_num_tiles>(mm_out_cb_id);
                        }
                        tile_regs_commit();
                        // Pack out to output buffer
                        mm_out_cb.reserve_back(out_subblock_num_tiles);

#if not defined FUSE_BIAS and defined SFPU_ACTIVATION
                        apply_activation_from_pack<
                            activation_type,
                            activation_param0,
                            activation_param1,
                            activation_param2>(out_subblock_num_tiles);
#else
                        tile_regs_wait();
#endif

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                        PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC

                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                        uint32_t start_dst_index = 0;

#ifdef SGLANG_TT_PREFETCHER_PACK_PROBE
                        // U13 Part 1b — SYNC'D DST probe at the pack site (BEFORE
                        // pack_tile_block).  Uses dprint_tensix_dest_reg<>(0) which
                        // internally calls dbg_halt() to drain the FPU pipeline and
                        // ensure DST RAM holds the post-final-matmul snapshot at
                        // read time.  This is the load-bearing measurement: if DST
                        // tile 0 row 0 reads as zero here AND the subsequent
                        // U13_PACK_OUT probe reads NONZERO, PACK is definitively
                        // writing garbage despite zero DST.
                        {
                            constexpr uint32_t u13d_elf_tag =
                                (in0_block_w * 1u) ^
                                (in0_num_subblocks * 131u) ^
                                (in1_num_subblocks * 17u) ^
                                (num_blocks * 7919u) ^
                                (out_subblock_h * 31u) ^
                                (out_subblock_w * 257u) ^
                                (batch * 65537u);
                            static uint32_t u13d_budget = 8;
                            if (u13d_budget > 0) {
                                u13d_budget--;
                                DPRINT << "[U13_DST_AT_PACK elf=0x" << HEX()
                                       << u13d_elf_tag
                                       << DEC() << " b=" << b
                                       << " blk=" << block
                                       << " is0=" << in0_subblock
                                       << " is1=" << in1_subblock
                                       << " (next: sync'd DST tile0)]" << ENDL();
                                MATH((
                                    {
                                        uint32_t dst_rd[8];
                                        // Force-drain pipeline so DST RAM reflects
                                        // the last matmul_block's write.  Cheaper
                                        // than dprint_tensix_dest_reg (no full tile
                                        // print) but uses the same dbg_halt()
                                        // barrier internally.
                                        ckernel::tensix_sync();
                                        ckernel::dbg_get_array_row(
                                            ckernel::dbg_array_id::DEST, 0, dst_rd);
                                        DPRINT << "[U13_DST_SYNC elf=0x" << HEX()
                                               << u13d_elf_tag
                                               << " b=" << b
                                               << " blk=" << block
                                               << " r0=";
                                        bool any_nonzero = false;
                                        for (int i = 0; i < 8; ++i) {
                                            DPRINT << "0x" << HEX() << dst_rd[i] << " ";
                                            if (dst_rd[i] != 0) any_nonzero = true;
                                        }
                                        DPRINT << (any_nonzero ? "NONZERO" : "zero")
                                               << "]" << ENDL();
                                    }
                                ));
                            }
                        }
#endif

                        if constexpr (untilize_out) {
                            pack_untilize_dest<out_subblock_num_tiles>(mm_out_cb_id);
                        } else {
                            pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);
                        }

#ifdef SGLANG_TT_PREFETCHER_PACK_PROBE
                        // U13 Part 1 — PACK-side L1 probe immediately AFTER
                        // pack_tile_block writes to mm_out_cb.  Reads the first 16
                        // bytes (8 BF16 / 4 FP32 words) at the location PACK just
                        // wrote and prints them.  Combined with U11 zero-weight
                        // injection + U12's DST=0 confirmation, this discriminates
                        //   * mm_out_cb == zero  -> PACK is correct; bug is in
                        //                          mm_partials spill/reload, an
                        //                          un-probed ELF, or a downstream
                        //                          consumer reading at a wrong L1
                        //                          offset.
                        //   * mm_out_cb != zero  -> PACK is writing garbage despite
                        //                          DST being zero; bug is in the
                        //                          PACK pipeline (formatter, wr_ptr,
                        //                          datum-stride, ...).
                        //
                        // The probe runs on the PACK TRISC thread (which issued the
                        // pack_tile_block writes), then prints up to 8 dwords from
                        // the L1 address PACK just used.  Per-ELF static budget
                        // keeps log size bounded; CT ELF tag de-dupes per matmul
                        // shape.
                        {
                            constexpr uint32_t u13p_elf_tag =
                                (in0_block_w * 1u) ^
                                (in0_num_subblocks * 131u) ^
                                (in1_num_subblocks * 17u) ^
                                (num_blocks * 7919u) ^
                                (out_subblock_h * 31u) ^
                                (out_subblock_w * 257u) ^
                                (batch * 65537u);
                            static uint32_t u13p_out_budget = 16;
                            PACK((
                                {
                                    if (u13p_out_budget > 0) {
                                        u13p_out_budget--;
                                        // CB_WR_PTR: (fifo_wr_ptr << cb_addr_shift)
                                        // — yields the L1 byte address of the next
                                        // write slot.  Since cb.push_back has NOT
                                        // run yet, fifo_wr_ptr still points to the
                                        // base where PACK just wrote.
                                        //
                                        // Force packer write completion to L1 so
                                        // the subsequent volatile read sees the
                                        // post-pack bytes (else we may race the
                                        // packer's L1 NoC write).
                                        ckernel::tensix_sync();
                                        uint32_t l1_addr = CB_WR_PTR(mm_out_cb_id);
                                        volatile tt_l1_ptr uint32_t* p =
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
                                        uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                                        bool any_nonzero =
                                            v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
                                        DPRINT << "[U13_PACK_OUT elf=0x" << HEX()
                                               << u13p_elf_tag
                                               << " l1=0x" << l1_addr
                                               << DEC() << " b=" << b
                                               << " blk=" << block
                                               << " is0=" << in0_subblock
                                               << " is1=" << in1_subblock
                                               << " w0=0x" << HEX() << v[0]
                                               << " w1=0x" << v[1]
                                               << " w2=0x" << v[2]
                                               << " w3=0x" << v[3]
                                               << " "
                                               << (any_nonzero ? "NONZERO" : "zero")
                                               << "]" << ENDL();
                                    }
                                }
                            ));
                        }
#endif

#ifdef SGLANG_TT_U18_PACK_PROBE
                        // U18 Phase 1 — large-budget PACK probe targeted at
                        // the W2 matmul under TRACE REPLAY.  U13 ran with
                        // SGLANG_TT_DISABLE_PREFILL_TRACE=1 (eager only).
                        // This probe runs WITHOUT that, so it observes PACK
                        // behavior during trace replay launches as well as
                        // the compile-run launch.  Tag includes ELF hash so
                        // we can filter to W2 in postprocessing.  Budget
                        // bumped to 2048 to cover ~64 decode steps * ~32
                        // subblock launches per launch.
                        {
                            constexpr uint32_t u18_elf_tag =
                                (in0_block_w * 1u) ^
                                (in0_num_subblocks * 131u) ^
                                (in1_num_subblocks * 17u) ^
                                (num_blocks * 7919u) ^
                                (out_subblock_h * 31u) ^
                                (out_subblock_w * 257u) ^
                                (batch * 65537u);
                            static uint32_t u18_out_budget = 2048;
                            static uint32_t u18_out_total = 0;
                            PACK((
                                {
                                    u18_out_total++;
                                    if (u18_out_budget > 0) {
                                        u18_out_budget--;
                                        ckernel::tensix_sync();
                                        uint32_t l1_addr = CB_WR_PTR(mm_out_cb_id);
                                        volatile tt_l1_ptr uint32_t* p =
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
                                        uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                                        bool any_nonzero =
                                            v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
                                        DPRINT << "[U18_PACK elf=0x" << HEX()
                                               << u18_elf_tag
                                               << " l1=0x" << l1_addr
                                               << DEC() << " tot=" << u18_out_total
                                               << " b=" << b
                                               << " blk=" << block
                                               << " is0=" << in0_subblock
                                               << " is1=" << in1_subblock
                                               << " w0=0x" << HEX() << v[0]
                                               << " w1=0x" << v[1]
                                               << " w2=0x" << v[2]
                                               << " w3=0x" << v[3]
                                               << " "
                                               << (any_nonzero ? "NONZERO" : "zero")
                                               << "]" << ENDL();
                                    }
                                }
                            ));
                        }
#endif

                        tile_regs_release();
                        if constexpr (untilize_out) {
                            pack_untilize_uninit(mm_out_cb_id);
                        }
                        mm_out_cb.push_back(out_subblock_num_tiles);

                    } else if (spill) {
                        tile_regs_commit();
                        // Move partial result to interm buffer
                        mm_partials_cb.reserve_back(out_subblock_num_tiles);
                        tile_regs_wait();

#ifdef PACKER_L1_ACC
                        if (block == 0) {  // no accumulation for first iteration
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        } else if (block == 1) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }
#endif

                        uint32_t start_dst_index = 0;
                        pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

#ifdef SGLANG_TT_PREFETCHER_PACK_PROBE
                        // U13 Part 1 — PACK-side L1 probe for the spill branch
                        // (writes to mm_partials_cb).  Same shape/structure as the
                        // mm_out_cb probe above.  Independent budget so we capture
                        // both code paths.
                        {
                            constexpr uint32_t u13p_elf_tag =
                                (in0_block_w * 1u) ^
                                (in0_num_subblocks * 131u) ^
                                (in1_num_subblocks * 17u) ^
                                (num_blocks * 7919u) ^
                                (out_subblock_h * 31u) ^
                                (out_subblock_w * 257u) ^
                                (batch * 65537u);
                            static uint32_t u13p_part_budget = 16;
                            PACK((
                                {
                                    if (u13p_part_budget > 0) {
                                        u13p_part_budget--;
                                        uint32_t l1_addr = CB_WR_PTR(mm_partials_cb_id);
                                        volatile tt_l1_ptr uint32_t* p =
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
                                        uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                                        bool any_nonzero =
                                            v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
                                        DPRINT << "[U13_PACK_PART elf=0x" << HEX()
                                               << u13p_elf_tag
                                               << " l1=0x" << l1_addr
                                               << DEC() << " b=" << b
                                               << " blk=" << block
                                               << " is0=" << in0_subblock
                                               << " is1=" << in1_subblock
                                               << " w0=0x" << HEX() << v[0]
                                               << " w1=0x" << v[1]
                                               << " w2=0x" << v[2]
                                               << " w3=0x" << v[3]
                                               << " "
                                               << (any_nonzero ? "NONZERO" : "zero")
                                               << "]" << ENDL();
                                    }
                                }
                            ));
                        }
#endif

                        tile_regs_release();
                        mm_partials_cb.push_back(out_subblock_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

#ifdef PACKER_L1_ACC

            // Last iteration does spill and reload to output buffer
            if (block < num_blocks - 2 && spill) {
                mm_partials_cb.wait_front(out_block_num_tiles);
                mm_partials_cb.pop_front(out_block_num_tiles);
            }
            if (block == num_blocks - 2 && spill) {
                enable_reload = true;
            }  // reload when last iteration
#else
            if constexpr (spill) {
                enable_reload = true;
            }
#endif

            input0_cb.pop_front(in0_block_num_tiles);
            if constexpr (in1_is_dram) {
                in1_cb.pop_front(in1_block_num_tiles);
            }
#ifdef ENABLE_GLOBAL_CB
    #ifndef SGLANG_TT_PREFETCHER_BYPASS_GCB_BLOCK
            // U10 Block C: per-block-end — applies the next-block pointer that
            // Block B calculated. Bypassed in lockstep with Block B (they're a
            // unit). When bypassed, the CB rd_ptr stays at its current value
            // for all blocks: the loop will read block 0 repeatedly. Output
            // will be wrong but in a DIFFERENT class than the baseline garbage
            // — if THAT signature matches the baseline garbage, the wrap
            // arithmetic is the source.
            curr_in1_block_index = next_in1_block_index;
            UNPACK((update_local_cb_rd_ptr(in1_cb_id, next_in1_rd_ptr_addr)));
    #endif
#endif
        }

#ifdef ENABLE_GLOBAL_CB
        // Release in1
        sync_buf.reserve_back(1);
        sync_buf.push_back(1);
    #ifndef SGLANG_TT_PREFETCHER_BYPASS_GCB_ADVANCE
        // U10 Block D: end-of-batch — resets rd_ptr to the saved start of
        // THIS matmul's view, then advances by ring_size blocks to land on
        // the NEXT matmul's data in GlobalCB. This is THE inter-matmul
        // handoff. If bypassed, the next matmul's Block A will capture the
        // current rd_ptr (which after the inner loop is somewhere mid-tensor
        // depending on how Block B/C ran) instead of the proper "next tensor"
        // address. The sync_buf push is NOT bypassed — the next matmul
        // depends on this sync handshake regardless of rd_ptr semantics.
        UNPACK((update_local_cb_rd_ptr(in1_cb_id, in1_rd_ptr_start_addr)));  // reset rd_ptr back to the initial addr
        UNPACK((update_rd_ptr_to_ring_index(
            in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)));  // update to next tensor addr
    #endif
#endif
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
        // U14 — END-OF-BATCH mm_out_cb L1 re-read probe.  By this point all
        // PACK writes for batch `b` have been issued AND all sync handshakes
        // have been performed.  We re-read the FIRST 16 bytes of mm_out_cb's
        // L1 region (its fifo_start_addr) and compare to what U13_PACK_OUT
        // recorded immediately after PACK.  Because mm_out_cb is allocated
        // with set_globally_allocated_address(*out_buffer), its L1 base IS
        // the output tensor's L1 buffer for this core.  If at kernel exit
        // these bytes are NONZERO under zero-weight injection, something
        // INSIDE the matmul kernel between PACK and exit stomped them
        // (most likely PACK reload + accumulation in the spill branch).
        // If they are STILL ZERO at kernel exit, the stomper lives OUTSIDE
        // the matmul kernel (consumer reader, intervening CCL kernel, or
        // an unrelated kernel sharing the same L1 region).  Per-ELF static
        // budget keeps log size bounded; CT ELF tag de-dupes per matmul
        // shape; tensix_sync() before the volatile read avoids the U13
        // race against in-flight L1 writes.
        {
            constexpr uint32_t u14_elf_tag =
                (in0_block_w * 1u) ^
                (in0_num_subblocks * 131u) ^
                (in1_num_subblocks * 17u) ^
                (num_blocks * 7919u) ^
                (out_subblock_h * 31u) ^
                (out_subblock_w * 257u) ^
                (batch * 65537u);
            // U18 Phase 3 — bumped from 16 to 2048 so we capture trace-replay
            // state, not just the compile-run state.  Combined with U17_PRE_RS,
            // this lets us see whether L1 at kernel exit (zero) becomes
            // NONZERO before RS reader runs (= P3 confirmed).
            static uint32_t u14_end_budget = 2048;
            static uint32_t u14_end_total = 0;
            PACK((
                {
                    u14_end_total++;
                    if (u14_end_budget > 0) {
                        u14_end_budget--;
                        ckernel::tensix_sync();
                        // get_local_cb_start_addr returns the value in SHIFTED
                        // CB units (matches fifo_wr_ptr / fifo_rd_ptr); convert
                        // to a byte address by left-shifting by cb_addr_shift,
                        // same as the CB_WR_PTR macro in dprint_tile.h.
                        uint32_t l1_start = get_local_cb_start_addr(mm_out_cb_id) << cb_addr_shift;
                        volatile tt_l1_ptr uint32_t* p =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_start);
                        uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                        bool any_nonzero =
                            v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
                        DPRINT << "[U14_END_OUT elf=0x" << HEX()
                               << u14_elf_tag
                               << " l1=0x" << l1_start
                               << DEC() << " tot=" << u14_end_total
                               << " b=" << b
                               << " w0=0x" << HEX() << v[0]
                               << " w1=0x" << v[1]
                               << " w2=0x" << v[2]
                               << " w3=0x" << v[3]
                               << " "
                               << (any_nonzero ? "NONZERO" : "zero")
                               << "]" << ENDL();
                    }
                }
            ));
        }
#endif

#ifdef SGLANG_TT_W2_RS_BARRIER_KERNEL
        // U16 (2026-05-25) — per-batch PACK fence.  When U16 is enabled at
        // the program-factory level, this kernel was compiled with
        // SGLANG_TT_W2_RS_BARRIER_KERNEL so PACK writes to mm_out_cb are
        // guaranteed flushed to L1 before the next dispatch ("matmul done"
        // signal) fires.  Without this fence, PACK writes can race against
        // the immediately-following reduce_scatter's noc_async_read of
        // mm_out_cb's L1 region (U14 / U15 evidence).
        PACK((ckernel::tensix_sync()));
#endif
    }
#ifdef SGLANG_TT_W2_RS_BARRIER_KERNEL
    // U16 — final kernel-exit fence across all PACK/UNPACK/MATH threads.
    // Ensures the matmul kernel does NOT return to the dispatcher (which
    // would advance the receiver-stream completion counter) until every
    // pending tensix op (including PACK->L1 writes) has retired.
    ckernel::tensix_sync();
#endif
#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
    // U29 Phase 2 v4 — proper producer-side sync between compute and
    // dataflow.  The in1 sender writer's noc_semaphore_inc CANNOT run
    // until ALL PACK→L1 writes for the entire matmul are retired.
    // Within the per-batch loop, dataflow already cb_sync.wait_fronts
    // on compute's per-batch sync_buf push.  But the FINAL tensix_sync
    // (this one) happens AFTER the loop exits in compute — at which
    // point dataflow's loop has also exited and is NOT waiting on
    // anything.  Result: dataflow's exit-time noc_semaphore_inc fires
    // BEFORE compute's final tensix_sync, racing PACK.
    //
    // Fix: push one MORE sync_buf entry AFTER the final tensix_sync.
    // Dataflow's exit code (in1 sender writer) waits on this push
    // before issuing noc_semaphore_inc.  The sync_cb has capacity 1
    // page (16 bytes) and is empty at this point (all per-batch
    // pushes have been popped), so this push fits.
    ckernel::tensix_sync();
    sync_buf.reserve_back(1);
    sync_buf.push_back(1);
#endif
}
