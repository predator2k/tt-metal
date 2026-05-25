// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include <cstdint>
#include <utility>

#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
// U14 — CONSUMER probe.  Reads the first 16 bytes pulled from the input
// tensor (the matmul output's L1 buffer) immediately after
// noc_async_read_barrier completes the NoC read.  If these bytes are
// nonzero under zero-weight injection, the L1 region was stomped between
// the matmul kernel's PACK (which U13 verified writes zero) and this read.
#include "api/debug/dprint.h"
#endif

#ifdef SGLANG_TT_U17_PROBE_RS_PRE
// U17 — Phase-0 PRE-RS probe.  Reads producer's L1 bytes at the very
// start of the RS reader, BEFORE any RS work fires.  Issues a single
// noc_async_read of the first 16 bytes from input_tensor_address +
// computed tile_id_start, drains the read with noc_async_read_barrier,
// and prints the bytes.  This discriminates:
//   - L1 zero at RS reader entry  -> stomper lives inside RS reader logic
//                                    or in a kernel that runs concurrently
//                                    with RS reader (very unlikely).
//   - L1 NONZERO at RS reader entry -> stomper runs BEFORE RS reader gets
//                                      its dispatch (i.e., some op between
//                                      W2 PACK completion and RS reader
//                                      start has stomped 0xa6700).
#include "api/debug/dprint.h"
#endif

#if defined(SGLANG_TT_U19_FORCE_ZERO) || defined(SGLANG_TT_U19_ADDR_DUMP)
// U19 — workaround / addr-dump in the reader kernel.
// FORCE_ZERO: after the first noc_async_read_barrier in the
//   "is_first_device_in_direction" branch, overwrites the local CB
//   bytes with ZERO.  Compute will then sum zero into the reduction —
//   if final output collapses to zero / a sane deterministic value,
//   confirms the stomp at L1 0xa6700 is the bug; otherwise something
//   else is also broken.
// ADDR_DUMP: prints intermediate_tensor_address and output_tensor_address
//   at kernel entry alongside U17's input_tensor_address dump.  Lets us
//   see if either intermediate or output is co-located at 0xa6700 across
//   iterations (i.e., L1 allocator reuse).
#include "api/debug/dprint.h"
#endif

// U19 — L1 data-cache invalidate ("fence") workaround.  Per the
// Blackhole bring-up guide: "Writing an address on one core and reading
// it from another only requires the reader to invalidate if the address
// was previously read."  L1 data cache is disabled by default but a
// stale-line race can still occur via the RISC-V write-buffer ordering.
// invalidate_l1_cache() is just a RISC-V fence on BH, which also
// orders all pending memory operations.  When set, the RS reader
// issues a fence before EVERY noc_async_read of producer L1.  If the
// 46% NONZERO race at L1 0xa6700 disappears, this is the bug.  No
// extra #include needed: invalidate_l1_cache() comes from
// dataflow_api.h already included above.

// U19 — FORCE_PRODUCER_ZERO workaround.  After the U17 PRE_RS probe
// confirms L1 0xa6700 is NONZERO on producer (2,7), do a NoC
// noc_async_write of zero bytes back to producer L1 0xa6700 to FORCE
// it to zero.  Then drain the write with noc_async_write_barrier.
// All subsequent RS reads of producer L1 0xa6700 will get zero (until
// PACK writes again, which it won't this iter).  Under zero weights,
// this should make the RS read deterministic zero — same as the
// canonical-no-prefetcher path — and the U11 garbage signature should
// appear consistently (not the racy variation we see today).  Under
// REAL weights this BREAKS the model (overwrites W2 output with zero)
// — diagnostic only.

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t output_batch_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t output_channel_num_pages = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(12);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(13);
constexpr uint32_t slice_C = get_compile_time_arg_val(14);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(15);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(16);
constexpr uint32_t fuse_op = get_compile_time_arg_val(17);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(18);
constexpr uint32_t dim = get_compile_time_arg_val(19);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fwd_bwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const bool is_forward = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_device_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_targets_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const bool do_final_reduction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 20;

#ifdef INPUT_IS_SHARDED
    constexpr uint32_t ct_offset_one = 7;

    using input_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx),       // Memory layout
        get_compile_time_arg_val(ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(ct_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + 6)>;  // pages_per_shard_y

    const auto [input_mapping_table, input_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<input_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<input_tensor_shard_info> input_tensor_addrgen = {
        .bank_base_address = input_tensor_address, .shard_array = input_mapping_table};

    arg_idx += input_rt_increment;
#else
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset_one = input_tensor_args.num_compile_time_args();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address);
#endif

#ifdef INTERMEDIATE_IS_SHARDED
    constexpr uint32_t ct_offset_two = 7;

    constexpr uint32_t inter_start_ct_idx = ct_idx + ct_offset_one;
    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(inter_start_ct_idx),       // Memory layout
        get_compile_time_arg_val(inter_start_ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(inter_start_ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(inter_start_ct_idx + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(inter_start_ct_idx + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(inter_start_ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(inter_start_ct_idx + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_tensor_addrgen = {
        .bank_base_address = intermediate_tensor_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;
#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx + ct_offset_one>();
    constexpr uint32_t ct_offset_two = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address);
#endif

#ifdef OUTPUT_IS_SHARDED
    constexpr uint32_t output_start_ct_idx = ct_idx + ct_offset_one + ct_offset_two;
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(output_start_ct_idx),       // Memory layout
        get_compile_time_arg_val(output_start_ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(output_start_ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(output_start_ct_idx + 3),   // The number of pages in each sharding row not including
                                                             // padding pages
        get_compile_time_arg_val(output_start_ct_idx + 4),   // This defines times when contiguous pages can't be
                                                             // calculated
        get_compile_time_arg_val(output_start_ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(output_start_ct_idx + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_tensor_addrgen = {
        .bank_base_address = output_tensor_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset_one + ct_offset_two>();
    auto output_tensor_addrgen = TensorAccessor(output_tensor_args, output_tensor_address);
#endif

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

#ifdef SGLANG_TT_U29_W2_RS_SIGNALER
    // U29 Phase 2 — consumer-side signal WAIT (relative-counter model).
    // Pair to the W2 in1_ring_all_gather producer-side
    // `noc_semaphore_inc` increment.  See
    // matmul_multicore_reuse_mcast_1d_program_factory.cpp (factory)
    // and reader_bmm_tile_layout_in1_ring_all_gather.cpp (producer
    // kernel).
    //
    // Cross-sub-device dispatch race (U28-β CONFIRMED): without an
    // in-kernel handshake, RS reader's noc_async_read of W2's
    // mm_out_cb L1 region can fire BEFORE W2's PACK has retired,
    // observing residual data at 0xa6700.
    //
    // Runtime args appended by the program factory AFTER any sharding
    // and fused-op args:
    //   [arg_idx]   num_producers (uint32, e.g. 32 for Qwen3-8B W2)
    //   [arg_idx+1] producer0_noc_x
    //   [arg_idx+2] producer0_noc_y
    //   ...         (interleaved x, y per producer)
    //
    // Per-replay synchronization model (RELATIVE):
    //   * ALL gathered matmuls on the prefetcher path increment the
    //     same producer-core L1 slot 0x90000 by 1 per dispatch.  Thus
    //     the absolute counter value drifts with the number of
    //     gathered matmuls per layer (W2, WO, FF1, FF2, FF3, WQKV ~ 6
    //     per layer), not 1:1 with RS-reader calls.
    //   * BUT, the trace-replay schedule is deterministic: each W2 is
    //     always dispatched immediately before its paired RS in the
    //     same layer.  So between RS_{K-1} and RS_K, AT LEAST ONE
    //     producer-side increment fires — specifically, W2 of layer K.
    //   * Therefore: track in per-RS-worker local L1 (`u29_prev_seen`)
    //     the producer-counter value observed at the END of the prior
    //     RS wait.  On entry, spin until producer's counter strictly
    //     EXCEEDS `u29_prev_seen` on every producer core; then update
    //     `u29_prev_seen` to the *minimum* observed value (conservative
    //     lower bound for next iteration).
    //
    // This is robust to all gathered matmuls sharing the same 0x90000
    // slot, and converges on the W2→RS race specifically because W2 is
    // the LAST gathered matmul to fire on the receiver cores before
    // the line RS dispatches on the worker cores.
    const uint32_t u29_num_producers = get_arg_val<uint32_t>(arg_idx++);
    uint32_t u29_producer_args_start = arg_idx;
    arg_idx += u29_num_producers * 2;  // skip past x,y pairs for tail args (none today)

    {
        static uint32_t u29_prev_seen = 0;

#ifdef SGLANG_TT_U29_SENTINEL
        // U29 Phase 3 — sentinel-mode handshake verification.
        //
        // Goal: prove or disprove that the chosen L1 sema slot is a
        // SAFE address (i.e. not stomped on by a tensor placement) AND
        // that the producer's noc-write actually lands there before
        // the RS reader reads it.
        //
        // The producer writes 0xDEADBEEF to the slot at end of W2 PACK
        // (see reader_bmm_tile_layout_in1_ring_all_gather.cpp).  The RS
        // reader reads from each producer's slot here and DPRINTs the
        // value.  If we consistently see 0xDEADBEEF on the FIRST forward
        // (and any value other than 0 after the second forward when the
        // sema is reset to 0 between forwards), the handshake works.
        // If we see any other value (e.g., looks like tensor data,
        // looks like the previous L1 contents), the slot is unsafe.
        if (u29_num_producers > 0) {
            cb_reserve_back(cb_input_id, 1);
            uint32_t u29_l1_scratch = get_write_ptr(cb_input_id);
            volatile tt_l1_ptr uint32_t* u29_scratch_p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u29_l1_scratch);

            static uint32_t u29_sentinel_log_budget = 64;
            // Sample only the FIRST producer core to keep DPRINT volume
            // manageable; one log per RS-reader-call is enough to confirm
            // the handshake.
            const uint32_t px = get_arg_val<uint32_t>(u29_producer_args_start + 0);
            const uint32_t py = get_arg_val<uint32_t>(u29_producer_args_start + 1);
            const uint64_t prod_sema_noc_addr =
                get_noc_addr(px, py, static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1));
            u29_scratch_p[0] = 0;
            noc_async_read(prod_sema_noc_addr, u29_l1_scratch, 4);
            noc_async_read_barrier();
            const uint32_t observed = u29_scratch_p[0];
            if (u29_sentinel_log_budget > 0) {
                u29_sentinel_log_budget--;
                DPRINT << "[U29_SENTINEL prod=(" << px << "," << py
                       << ") sema_addr=0x" << HEX()
                       << static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1)
                       << " val=0x" << observed
                       << DEC() << "]" << ENDL();
            }
        }
#else
#ifndef SGLANG_TT_U29_DISABLE_WAIT
        if (u29_num_producers > 0) {
            cb_reserve_back(cb_input_id, 1);
            uint32_t u29_l1_scratch = get_write_ptr(cb_input_id);
            volatile tt_l1_ptr uint32_t* u29_scratch_p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u29_l1_scratch);

            uint32_t u29_min_observed = 0xFFFFFFFFu;
            for (uint32_t i = 0; i < u29_num_producers; i++) {
                const uint32_t px = get_arg_val<uint32_t>(u29_producer_args_start + 2 * i);
                const uint32_t py = get_arg_val<uint32_t>(u29_producer_args_start + 2 * i + 1);
                const uint64_t prod_sema_noc_addr =
                    get_noc_addr(px, py, static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1));
                uint32_t cur_val = 0;
                // Spin until producer counter has advanced past
                // u29_prev_seen.  This guarantees the W2 PACK for THIS
                // layer's RS has retired (last gathered matmul before
                // this RS in deterministic trace order).
                do {
                    u29_scratch_p[0] = 0;
                    noc_async_read(prod_sema_noc_addr, u29_l1_scratch, 4);
                    noc_async_read_barrier();
                    cur_val = u29_scratch_p[0];
                } while (cur_val <= u29_prev_seen);
                if (cur_val < u29_min_observed) {
                    u29_min_observed = cur_val;
                }
            }
            // Conservative: bump prev_seen to the lowest observed value.
            // Any future wait must see counter > this min.
            u29_prev_seen = u29_min_observed;
        }
#endif  // SGLANG_TT_U29_DISABLE_WAIT

#ifdef SGLANG_TT_U29_DEBUG_DPRINT
        {
            static uint32_t u29_log_budget = 32;
            if (u29_log_budget > 0) {
                u29_log_budget--;
                DPRINT << "[U29_RS_READER_OK prev_seen=" << u29_prev_seen
                       << " num_producers=" << u29_num_producers
                       << " sema_addr=0x" << HEX()
                       << static_cast<uint32_t>(SGLANG_TT_U29_SEMA_L1)
                       << DEC() << "]" << ENDL();
            }
        }
#endif
#endif  // SGLANG_TT_U29_SENTINEL
    }
#endif

    /**
     * Intermediate buffer is double-sized (shape [2, *input_shape]) to accommodate forward and backward.
     * BWD indexes into second half of intermediate buffer.
     */
    const uint32_t intermediate_full_offset = is_forward ? 0 : input_num_pages;

    uint32_t chunk_count = 0;
    uint32_t fwd_sync_cnt = 0;
    uint32_t sem_target = 0;

#ifdef SGLANG_TT_U17_PROBE_RS_PRE
    // U17 Phase-0 — PRE-RS probe.  Read producer's first tile L1 bytes
    // immediately at RS reader entry, BEFORE any normal RS work fires.
    // Uses tile_id 0 (start_tiles_read=0 is the most common case for the
    // forward direction's first slice).  We read into a temporary L1
    // scratch slot (reuse cb_input_id's first reserve).  Per-launch
    // budget keeps log size bounded.
    {
        // U18 Phase 3 — bumped from 16 to 2048 so we capture trace-replay
        // state across many decode steps, paired with the bumped U14
        // CONSUMER_PROBE.
        static uint32_t u17_pre_budget = 2048;
        static uint32_t u17_pre_total = 0;
        u17_pre_total++;
        if (u17_pre_budget > 0) {
            u17_pre_budget--;
            cb_reserve_back(cb_input_id, 1);
            uint32_t l1_scratch = get_write_ptr(cb_input_id);
            uint32_t probe_tile_id = 0;  // first tile of input
            uint64_t probe_noc_addr = get_noc_addr(probe_tile_id, input_tensor_addrgen);
            // Zero-fill scratch first so a zero result is meaningful.
            volatile tt_l1_ptr uint32_t* z =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_scratch);
            z[0] = 0; z[1] = 0; z[2] = 0; z[3] = 0;
#ifdef SGLANG_TT_U19_INVALIDATE_CACHE
            // U19 — fence in U17 probe too.
            invalidate_l1_cache();
#endif
            // Issue a single read of page_size bytes from producer's L1.
            noc_async_read(probe_noc_addr, l1_scratch, page_size);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_scratch);
            uint32_t v[4] = { p[0], p[1], p[2], p[3] };
            bool any_nonzero = v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
            DPRINT << "[U17_PRE_RS in_addr=0x" << HEX() << input_tensor_address
                   << " noc=0x" << probe_noc_addr
                   << " w0=0x" << v[0]
                   << " w1=0x" << v[1]
                   << " w2=0x" << v[2]
                   << " w3=0x" << v[3]
                   << " " << DEC() << " tot=" << u17_pre_total
                   << " " << (any_nonzero ? "NONZERO" : "zero")
                   << "]" << ENDL();
#ifdef SGLANG_TT_U25_BYTE_HUNT
            // U25 Path B — byte-pattern identity tracing on device.
            // When the U17 probe sees NONZERO bytes at the cursed L1
            // 0xa6700, hunt for the source by:
            //   (1) scanning the SAME NoC core's L1 [0x10000, 0x1c0000]
            //       at 0x40 stride for matching 16-byte windows;
            //   (2) probing the SAME L1 address (0xa6700) on multiple
            //       OTHER tensix cores on chip 0 to see who else holds
            //       these bytes (mcast destinations?).
            // Per-RISC budget caps log size.
            // U25 v4 — disabled, was hanging.  Same-core L1 scan
            // already proved bytes don't exist anywhere else in
            // (2,7)'s L1 [0x10000, 0x1c0000].  Path B exhausted
            // for device-side hunting; pivoting to host-side
            // analysis.
            (void)any_nonzero;
#endif
#ifdef SGLANG_TT_U19_ADDR_DUMP
            // U19 — also dump intermediate & output addresses so we can
            // correlate every RS dispatch's full tensor-address triple
            // against W2 output address 0xa6700.  If intermediate or
            // output ever equal 0xa6700, the RS writer (which writes to
            // intermediate and output) is a candidate stomper.
            DPRINT << "[U19_ADDR_DUMP in=0x" << HEX() << input_tensor_address
                   << " inter=0x" << intermediate_tensor_address
                   << " out=0x" << output_tensor_address
                   << DEC() << "]" << ENDL();
#endif
#ifdef SGLANG_TT_U19_FORCE_PRODUCER_ZERO
            // U19 — FORCE producer L1 to zero via NoC write-back, then
            // READBACK to verify.  After the U17 probe shows producer L1
            // NONZERO, we overwrite via a NoC write of zero bytes, then
            // re-read to confirm.  If the readback ALSO shows NONZERO,
            // something is constantly stomping producer L1 (not a
            // historical artifact).  If readback shows zero, the write
            // worked and the bytes are coherent — meaning the original
            // U17 NONZERO read was racing against an in-flight stomper
            // that DIDN'T re-stomp after our write.
            {
                volatile tt_l1_ptr uint32_t* zlocal =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_scratch);
                for (uint32_t i = 0; i < page_size / 4; ++i) {
                    zlocal[i] = 0;
                }
                noc_async_write(l1_scratch, probe_noc_addr, page_size);
                noc_async_write_barrier();
                // Reuse same scratch; readback overwrites it.
                noc_async_read(probe_noc_addr, l1_scratch, page_size);
                noc_async_read_barrier();
                uint32_t r[4] = { zlocal[0], zlocal[1], zlocal[2], zlocal[3] };
                bool rback_nonzero = r[0] != 0 || r[1] != 0 || r[2] != 0 || r[3] != 0;
                DPRINT << "[U19_FORCED_ZERO_RBACK probe_noc=0x" << HEX()
                       << probe_noc_addr
                       << " r0=0x" << r[0]
                       << " r1=0x" << r[1]
                       << " r2=0x" << r[2]
                       << " r3=0x" << r[3]
                       << " " << DEC()
                       << (rback_nonzero ? "NONZERO" : "zero")
                       << "]" << ENDL();
            }
#endif
            // Do NOT push_back — leave scratch reserved-but-not-consumed,
            // we'll write over it in the normal flow.  Or rather: we just
            // reset the cb_input_id reservation by NOT pushing.  Safer is
            // to not reserve at all — but cb_reserve_back is required so
            // we get a valid l1_write_addr.  Since we never push_back, the
            // subsequent cb_reserve_back calls in the main loop will see
            // the same address and overwrite.
        }
    }
#endif

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        if (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        int slice_idx = is_forward ? ring_size - 1 : 0;
        uint32_t batch_offset = input_batch_num_pages * b;

        // Iterate over the slices in the direction we are going.
        // In forwards direction, count down from slice (ring_size -1) down to (my_chip_id+1), inclusive
        // In backwards direction, count up from slice 0 to (my_chip_id-1), inclusive
        // After doing all partial reductions and send, there's a final reduction step.
        // If we are not the first device in the direction, do the final reduction.
        // If this device has both FWD and BWD neighbors, the FWD reader will do final reduction first
        // and then signal the BWD reader to do its final reduction.
        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            chunk_count = 0;

            uint32_t input_tile_id_start;
            if constexpr (dim == 3) {
                input_tile_id_start = slice_idx * slice_Wt + batch_offset;
            } else if constexpr (dim == 2) {
                input_tile_id_start = slice_idx * slice_Ht * slice_Wt + batch_offset;
            } else if constexpr (dim == 1) {
                input_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt + batch_offset;
            } else {
                ASSERT(false);
            }
            uint32_t intermediate_tile_id_start = input_tile_id_start + intermediate_full_offset;

            if (is_first_device_in_direction) {
                // We have no incoming slices, so forward directly to writer
                uint32_t cb_in0 = cb_reader_output_id;
                for (uint32_t c = 0; c < slice_C; ++c) {
                    uint32_t input_pages_read_in_row = start_pages_read_in_row;
                    uint32_t input_row_offset = start_row_offset;

                    uint32_t tiles_read = start_tiles_read;
                    uint32_t tiles_to_read = start_tiles_to_read;

                    while (tiles_read < tiles_to_read) {
                        uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                        uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                        cb_reserve_back(cb_in0, tile_granularity);
                        uint32_t l1_write_addr = get_write_ptr(cb_in0);
#if defined(SGLANG_TT_PREFETCHER_CONSUMER_PROBE) || defined(SGLANG_TT_U19_FORCE_ZERO)
                        uint32_t u14_l1_base = l1_write_addr;
#endif
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
                        uint64_t u14_first_noc_addr = 0;
#endif
#ifdef SGLANG_TT_U19_INVALIDATE_CACHE
                        // U19 — invalidate this RISC's L1 cache lines (fence)
                        // BEFORE issuing the NoC read of producer L1.  If the
                        // producer's stale bytes are sitting in our local
                        // cache line for 0xa6700, fence forces a re-fetch.
                        invalidate_l1_cache();
#endif
                        for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                            uint32_t tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                            uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
                            if (j == 0) {
                                u14_first_noc_addr = noc_read_addr;
                            }
#endif
                            noc_async_read(noc_read_addr, l1_write_addr, page_size);
                            l1_write_addr += page_size;

                            input_pages_read_in_row++;
                            if (input_pages_read_in_row == slice_Wt) {
                                input_row_offset += input_tensor_Wt;
                                input_pages_read_in_row -= slice_Wt;
                            }
                        }
                        tiles_read += num_pages_to_read;

                        noc_async_read_barrier();
#ifdef SGLANG_TT_U19_FORCE_ZERO
                        // U19 Phase-1 — WORKAROUND TEST.  Overwrite every
                        // byte the reader just brought in with ZERO.  The
                        // reduction kernel then sums zero into the accumulator
                        // (and the writer scatters zero out), short-circuiting
                        // the stomped-L1 read.  Under zero-weight injection,
                        // the entire RS output should collapse to zero, and
                        // server output should become deterministic.  This
                        // PROVES the L1-0xa6700 stomp is the bug if the
                        // garbage signature disappears.
                        {
                            uint32_t bytes_just_read = num_pages_to_read * page_size;
                            uint32_t words = bytes_just_read >> 2;
                            volatile tt_l1_ptr uint32_t* z =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u14_l1_base);
                            for (uint32_t k = 0; k < words; ++k) {
                                z[k] = 0;
                            }
                        }
#endif
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
                        // U14 — read the first 16 bytes the consumer pulled
                        // from the matmul output's L1 buffer.  After
                        // noc_async_read_barrier, l1_write_addr region holds
                        // the bytes from the matmul output.  We read at
                        // u14_l1_base (the address the first page was written
                        // to).  Budget-limited per kernel launch; tagged with
                        // the input tensor address so we can correlate
                        // multiple matmul outputs across launches.
                        {
                            static uint32_t u14_first_budget = 16;
                            if (u14_first_budget > 0) {
                                u14_first_budget--;
                                volatile tt_l1_ptr uint32_t* p =
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u14_l1_base);
                                uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                                bool any_nonzero =
                                    v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
                                // NoC address layout (32-bit aliased view):
                                //   bits[31:24]=x_target_core
                                //   bits[23:16]=y_target_core
                                //   bits[15: 0]=l1_offset (low)
                                uint32_t noc_hi = (uint32_t)((u14_first_noc_addr >> 32) & 0xFFFFFFFFu);
                                uint32_t noc_lo = (uint32_t)(u14_first_noc_addr & 0xFFFFFFFFu);
                                DPRINT << "[U14_CONSUMER_FIRST in_addr=0x" << HEX()
                                       << input_tensor_address
                                       << " l1=0x" << u14_l1_base
                                       << " noc_hi=0x" << noc_hi
                                       << " noc_lo=0x" << noc_lo
                                       << " w0=0x" << v[0]
                                       << " w1=0x" << v[1]
                                       << " w2=0x" << v[2]
                                       << " w3=0x" << v[3]
                                       << " "
                                       << (any_nonzero ? "NONZERO" : "zero")
                                       << "]" << ENDL();
                            }
                        }
#endif
                        cb_push_back(cb_in0, tile_granularity);
                    }
                    input_tile_id_start += input_channel_num_pages;
                }
            } else {
                // I have incoming slices, so write my output to compute kernel and read intermediate input
                uint32_t cb_in0 = cb_input_id;
                for (uint32_t c = 0; c < slice_C; ++c) {
                    uint32_t input_pages_read_in_row = start_pages_read_in_row;
                    uint32_t input_row_offset = start_row_offset;

                    uint32_t intermediate_pages_read_in_row = input_pages_read_in_row;
                    uint32_t intermediate_row_offset = input_row_offset;

                    uint32_t tiles_read = start_tiles_read;
                    uint32_t tiles_to_read = start_tiles_to_read;

                    while (tiles_read < tiles_to_read) {
                        uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                        uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                        cb_reserve_back(cb_in0, tile_granularity);
                        uint32_t l1_write_addr = get_write_ptr(cb_in0);
#if defined(SGLANG_TT_PREFETCHER_CONSUMER_PROBE) || defined(SGLANG_TT_U19_FORCE_ZERO)
                        uint32_t u14_l1_base_in = l1_write_addr;
#endif
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
                        uint64_t u14_first_noc_addr_in = 0;
#endif
#ifdef SGLANG_TT_U19_INVALIDATE_CACHE
                        // U19 — fence before reading producer L1 (input branch).
                        invalidate_l1_cache();
#endif
                        for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                            uint32_t tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                            uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
                            if (j == 0) {
                                u14_first_noc_addr_in = noc_read_addr;
                            }
#endif
                            noc_async_read(noc_read_addr, l1_write_addr, page_size);
                            l1_write_addr += page_size;

                            input_pages_read_in_row++;
                            if (input_pages_read_in_row == slice_Wt) {
                                input_row_offset += input_tensor_Wt;
                                input_pages_read_in_row -= slice_Wt;
                            }
                        }
                        tiles_read += num_pages_to_read;

                        if (chunk_count % chunks_per_sync == 0) {
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                        }
                        chunk_count++;

                        // read the next intermediate slice out of intermediate buffer, and put it in intermediate CB
                        cb_reserve_back(cb_intermediate_id, tile_granularity);
                        l1_write_addr = get_write_ptr(cb_intermediate_id);
                        for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                            uint32_t tile_id =
                                intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                            uint64_t noc_read_addr = get_noc_addr(tile_id, intermediate_tensor_addrgen);
                            noc_async_read(noc_read_addr, l1_write_addr, page_size);
                            l1_write_addr += page_size;

                            intermediate_pages_read_in_row++;
                            if (intermediate_pages_read_in_row == slice_Wt) {
                                intermediate_row_offset += input_tensor_Wt;
                                intermediate_pages_read_in_row -= slice_Wt;
                            }
                        }

                        noc_async_read_barrier();
#ifdef SGLANG_TT_U19_FORCE_ZERO
                        // U19 Phase-1 — WORKAROUND TEST.  Zero both
                        // input and intermediate L1 bytes in the
                        // "not first device" branch.  See FORCE_ZERO
                        // comment in the "is_first_device" branch above.
                        {
                            uint32_t bytes_just_read = num_pages_to_read * page_size;
                            uint32_t words = bytes_just_read >> 2;
                            volatile tt_l1_ptr uint32_t* zi =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u14_l1_base_in);
                            for (uint32_t k = 0; k < words; ++k) {
                                zi[k] = 0;
                            }
                            // Also zero the intermediate buffer we just
                            // populated to keep the reduction summing zero.
                            uint32_t l1_intermediate_base =
                                l1_write_addr - bytes_just_read;
                            volatile tt_l1_ptr uint32_t* zr =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_intermediate_base);
                            for (uint32_t k = 0; k < words; ++k) {
                                zr[k] = 0;
                            }
                        }
#endif
#ifdef SGLANG_TT_PREFETCHER_CONSUMER_PROBE
                        // U14 — same probe, "not first device" branch.  After
                        // the barrier both the input and intermediate buffers
                        // have been populated.  Dump the input-side bytes (the
                        // matmul output we want to verify).
                        {
                            static uint32_t u14_in_budget = 16;
                            if (u14_in_budget > 0) {
                                u14_in_budget--;
                                volatile tt_l1_ptr uint32_t* p =
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u14_l1_base_in);
                                uint32_t v[4] = { p[0], p[1], p[2], p[3] };
                                bool any_nonzero =
                                    v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0;
                                DPRINT << "[U14_CONSUMER_IN in_addr=0x" << HEX()
                                       << input_tensor_address
                                       << " l1=0x" << u14_l1_base_in
                                       << " noc=0x" << (uint32_t)(u14_first_noc_addr_in & 0xFFFFFFFF)
                                       << " w0=0x" << v[0]
                                       << " w1=0x" << v[1]
                                       << " w2=0x" << v[2]
                                       << " w3=0x" << v[3]
                                       << " "
                                       << (any_nonzero ? "NONZERO" : "zero")
                                       << "]" << ENDL();
                            }
                        }
#endif
                        cb_push_back(cb_in0, tile_granularity);
                        cb_push_back(cb_intermediate_id, tile_granularity);
                    }
                    input_tile_id_start += input_channel_num_pages;
                    intermediate_tile_id_start += input_channel_num_pages;
                }
            }

            // Next slice idx
            if (is_forward) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }

        // Do the final reduction. Synchronize with other direction.
        if (do_final_reduction) {
            chunk_count = 0;

            uint32_t input_tile_id_start;
            if constexpr (dim == 3) {
                input_tile_id_start = my_chip_id * slice_Wt + batch_offset;
            } else if constexpr (dim == 2) {
                input_tile_id_start = my_chip_id * slice_Ht * slice_Wt + batch_offset;
            } else if constexpr (dim == 1) {
                input_tile_id_start = my_chip_id * slice_C * slice_Ht * slice_Wt + batch_offset;
            } else {
                ASSERT(false);
            }
            uint32_t input_pages_read_in_row = start_pages_read_in_row;
            uint32_t input_row_offset = start_row_offset;
            uint32_t input_stride_Wt = input_tensor_Wt;

            uint32_t intermediate_tile_id_start = input_tile_id_start + intermediate_full_offset;
            uint32_t intermediate_pages_read_in_row_per_channel = input_pages_read_in_row;
            uint32_t intermediate_row_offset_per_channel = input_row_offset;
            uint32_t intermediate_stride_Wt = input_tensor_Wt;

            uint32_t output_tile_id_start = b * output_batch_num_pages;
            uint32_t output_pages_read_in_row = input_pages_read_in_row;
            uint32_t output_row_offset = input_row_offset / input_tensor_Wt * slice_Wt;
            uint32_t output_stride_Wt = slice_Wt;

            /**
             * If two cores are doing final reduction, BWD core will accumulate output with
             * incoming BWD intermediate. Use output address generator.
             * If true, output += intermediate. Otherwise, output = input + intermediate
             */
            const bool accumulate_output = sync_with_other_direction && !is_forward;
            uint32_t tile_id_start;
            uint32_t stride_Wt;
            uint32_t channel_num_pages;
            if (accumulate_output) {
                tile_id_start = output_tile_id_start;
                stride_Wt = output_stride_Wt;
                channel_num_pages = output_channel_num_pages;
            } else {
                tile_id_start = input_tile_id_start;
                stride_Wt = input_stride_Wt;
                channel_num_pages = input_channel_num_pages;
            }

            uint32_t cb_in0 = cb_input_id;
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t pages_read_in_row;
                uint32_t row_offset;
                if (accumulate_output) {
                    pages_read_in_row = output_pages_read_in_row;
                    row_offset = output_row_offset;
                } else {
                    pages_read_in_row = input_pages_read_in_row;
                    row_offset = input_row_offset;
                }

                uint32_t intermediate_pages_read_in_row = intermediate_pages_read_in_row_per_channel;
                uint32_t intermediate_row_offset = intermediate_row_offset_per_channel;

                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    // Wait for FWD writer to signal that it has done its final reduction
                    if (accumulate_output) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_bwd_sem_addr), ++fwd_sync_cnt);
                    }

                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_reserve_back(cb_in0, tile_granularity);
                    uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                        uint64_t noc_read_addr = accumulate_output ? get_noc_addr(tile_id, output_tensor_addrgen)
                                                                   : get_noc_addr(tile_id, input_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;

                        pages_read_in_row++;
                        if (pages_read_in_row == slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row -= slice_Wt;
                        }
                    }
                    tiles_read += num_pages_to_read;

                    if (chunk_count % chunks_per_sync == 0) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                    }
                    chunk_count++;

                    // read the next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                    cb_reserve_back(cb_intermediate_id, tile_granularity);
                    l1_write_addr = get_write_ptr(cb_intermediate_id);
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t intermediate_tile_id =
                            intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                        uint64_t noc_read_addr = get_noc_addr(intermediate_tile_id, intermediate_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;

                        intermediate_pages_read_in_row++;
                        if (intermediate_pages_read_in_row == slice_Wt) {
                            intermediate_row_offset += intermediate_stride_Wt;
                            intermediate_pages_read_in_row -= slice_Wt;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                    cb_push_back(cb_intermediate_id, tile_granularity);
                }
                tile_id_start += channel_num_pages;
                intermediate_tile_id_start += input_channel_num_pages;
            }
        }
    }
    // Reset my output ready semaphore
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}
