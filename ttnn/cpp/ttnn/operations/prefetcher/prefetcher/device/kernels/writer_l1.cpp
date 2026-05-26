// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "api/debug/dprint.h"

uint32_t increment_arg_idx(uint32_t& arg_idx, uint32_t num_args = 1) {
    uint32_t old_arg_idx = arg_idx;
    arg_idx += num_args;
    return old_arg_idx;
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t local_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(7);
    constexpr bool skip_ptr_update = get_compile_time_arg_val(8);

    // Runtime args
    // Note: Coalesced sizes -> wrt to receiver cores, sizes -> wrt to dram reader cores
    uint32_t rt_args_idx = 0;
    const uint32_t* coalesced_page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* coalesced_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* single_tile_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_height_in_tiles =
        (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));  // Kt / num_blocks = in_block_h;

    uint32_t noc = noc_index;
#ifdef SGLANG_TT_U18_PREFETCHER_WRITE_PROBE
    // U18 Phase 3 — log GlobalCB destination addresses written by the
    // prefetcher writer_l1.  If any destination address lands at
    // 0xa6700 (the W2 output L1 address), we've found the stomper.
    static uint32_t u18_pref_write_budget = 256;
#endif
#ifdef SGLANG_TT_U21_PREFETCHER_ADDR_PROBE
    // U21 Probe C — log the prefetcher's per-receiver page-sent
    // semaphore L1 address (`aligned_pages_sent_ptr`).  When the
    // sender finishes writing pages, it calls noc_semaphore_inc on
    // this remote L1 address on EACH receiver core.  If any
    // receiver's aligned_pages_sent_ptr lands at 0xa6700, this is
    // the stomp source.  Also dump remote_cb config_ptr (where the
    // sender's update_remote_cb_config_in_l1 writes) and
    // receiver_noc_xy_ptr.  Per-RISC budget; runs once per receiver
    // per send.
    static uint32_t u21_addr_budget = 512;
    {
        auto& _u21_remote_cb = get_remote_sender_cb_interface(remote_cb_id);
        uint32_t _u21_aligned_pages_sent_ptr = _u21_remote_cb.aligned_pages_sent_ptr;
        uint32_t _u21_config_ptr = _u21_remote_cb.config_ptr;
        uint32_t _u21_receiver_noc_xy_ptr = _u21_remote_cb.receiver_noc_xy_ptr;
        uint32_t _u21_num_receivers = _u21_remote_cb.num_receivers;
        uint32_t _u21_fifo_start = _u21_remote_cb.fifo_start_addr;
        if (u21_addr_budget > 0) {
            u21_addr_budget--;
            DPRINT << "[U21_PREF_ADDR aligned_pages_sent_ptr=0x" << HEX()
                   << _u21_aligned_pages_sent_ptr
                   << " config_ptr=0x" << _u21_config_ptr
                   << " receiver_noc_xy_ptr=0x" << _u21_receiver_noc_xy_ptr
                   << " fifo_start=0x" << _u21_fifo_start
                   << DEC() << " num_receivers=" << _u21_num_receivers
                   << "]" << ENDL();
            // Also dump the per-receiver pages_sent slots (stride 2*L1_ALIGNMENT).
            // L1_ALIGNMENT on blackhole = 16; we step by 32 bytes per receiver.
            for (uint32_t _u21_i = 0; _u21_i < _u21_num_receivers && _u21_i < 16; _u21_i++) {
                uint32_t _u21_per_recv_ptr =
                    _u21_aligned_pages_sent_ptr + _u21_i * 2 * L1_ALIGNMENT;
                DPRINT << "[U21_PREF_RECV_PSENT i=" << _u21_i
                       << " psent_ptr=0x" << HEX() << _u21_per_recv_ptr
                       << DEC() << "]" << ENDL();
            }
        }
    }
#endif
    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_coalesced_page_size = coalesced_page_sizes[t];
            uint32_t curr_coalesced_num_pages = coalesced_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_single_tile_sizes = single_tile_sizes[t];
            uint32_t curr_block_height_in_tiles = block_height_in_tiles[t];
            uint32_t curr_block_size = curr_block_num_tiles * curr_single_tile_sizes;
            uint32_t curr_block_size_per_receiver = curr_block_size / num_receivers;

            experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);

            for (uint32_t block = 0; block < num_blocks; ++block) {
                {
                    cb_wait_front(local_cb_id, max_block_num_tiles);
                    experimental::remote_cb_reserve_back(remote_cb_id, 1);
                    uint32_t local_cb_addr = get_read_ptr(local_cb_id);
#ifdef SGLANG_TT_U18_PREFETCHER_WRITE_PROBE
                    // Read the fifo_wr_ptr that the upcoming write will
                    // target — same value the kernel uses internally for
                    // dest_addr.
                    {
                        auto& _u18_remote_cb =
                            get_remote_sender_cb_interface(remote_cb_id);
                        uint32_t _u18_wr_ptr = _u18_remote_cb.fifo_wr_ptr;
                        uint32_t _u18_start = _u18_remote_cb.fifo_start_addr;
                        if (u18_pref_write_budget > 0) {
                            u18_pref_write_budget--;
                            DPRINT << "[U18_PREF_WR layer=" << layer
                                   << " t=" << t
                                   << " blk=" << block
                                   << " wr_ptr=0x" << HEX() << _u18_wr_ptr
                                   << " fifo_start=0x" << _u18_start
                                   << " local_cb=0x" << local_cb_addr
                                   << DEC() << "]" << ENDL();
                        }
                    }
#endif
#ifdef SGLANG_TT_U37_PROD_BYTES
                    // U37 — ground-truth byte-level probe at the PRODUCER.
                    // Dump the first 16 bytes of local_cb_addr (= the SOURCE
                    // bytes being NoC-written to each receiver's GCB region)
                    // for each (layer, t, block) we care about.  Gated to
                    // (layer==0, block==0) to cap the log to one dump per
                    // tensor per receiver-write.  These bytes are what each
                    // receiver's L1 SHOULD see at its rd_ptr for that
                    // (tensor, ring slot) pair.  Cross-correlate with
                    // U37_READ from the matmul compute kernel to find:
                    //   match     → bytes correctly delivered
                    //   divergent → producer's intended bytes != consumer's
                    //                read bytes (post-producer L1 stomp OR
                    //                NoC dst skew)
                    {
                        static uint32_t u37_pb_budget = 64;
                        if (layer == 0 && block == 0 && u37_pb_budget > 0) {
                            u37_pb_budget--;
                            auto& _u37_remote_cb =
                                get_remote_sender_cb_interface(remote_cb_id);
                            uint32_t _u37_wr_ptr = _u37_remote_cb.fifo_wr_ptr;
                            volatile uint32_t* _u37_src =
                                (volatile uint32_t*)local_cb_addr;
                            uint32_t _u37_w0 = _u37_src[0];
                            uint32_t _u37_w1 = _u37_src[1];
                            uint32_t _u37_w2 = _u37_src[2];
                            uint32_t _u37_w3 = _u37_src[3];
                            DPRINT << "[U37_PROD layer=" << layer
                                   << " t=" << t
                                   << " blk=" << block
                                   << " local_cb=0x" << HEX() << local_cb_addr
                                   << " wr_ptr=0x" << _u37_wr_ptr
                                   << " w=[0x" << _u37_w0
                                   << " 0x" << _u37_w1
                                   << " 0x" << _u37_w2
                                   << " 0x" << _u37_w3 << "]"
                                   << DEC() << " bsz=" << curr_block_size_per_receiver
                                   << "]" << ENDL();
                        }
                    }
#endif
                    experimental::remote_cb_push_back_and_write_pages<skip_ptr_update>(
                        remote_cb_id,
                        local_cb_addr,
                        1,
                        curr_block_height_in_tiles,
                        curr_coalesced_num_pages,
                        curr_coalesced_page_size,
                        noc);
                    noc_async_posted_writes_flushed();
                    cb_pop_front(local_cb_id, max_block_num_tiles);
                }
            }

            if (t == num_tensors - 1) {
                experimental::remote_cb_sender_barrier(remote_cb_id);
            }
        }
    }

#ifdef SGLANG_TT_U25_RCB_PROBE
    // U25 Path A — log the prefetcher's call to
    // update_remote_cb_config_in_l1.  Writes
    // remote_cb_interface.fifo_rd_ptr (== fifo_wr_ptr for sender — both
    // structs alias) to LOCAL L1 at
    // `config_ptr + offsetof(RemoteReceiverCBInterface, fifo_rd_ptr)`.
    // Note: U21 already showed prefetcher's config_ptr = 0x17f640;
    // re-verify under U25 to confirm the destination.
    {
        auto& _u25_rcb = get_remote_sender_cb_interface(remote_cb_id);
        uint32_t _u25_config_ptr = _u25_rcb.config_ptr;
        uint32_t _u25_dest_addr =
            _u25_config_ptr + offsetof(RemoteReceiverCBInterface, fifo_rd_ptr);
        uint32_t _u25_value = _u25_rcb.fifo_wr_ptr;
        bool _u25_hits_stomp =
            (_u25_dest_addr >= 0xa6000 && _u25_dest_addr < 0xa7000);
        static uint32_t _u25_pref_budget = 256;
        if (_u25_pref_budget > 0) {
            _u25_pref_budget--;
            DPRINT << "[U25_RCB_PREF cb=" << remote_cb_id
                   << " config_ptr=0x" << HEX() << _u25_config_ptr
                   << " dest=0x" << _u25_dest_addr
                   << " value=0x" << _u25_value
                   << DEC()
                   << " hits_stomp=" << (uint32_t)_u25_hits_stomp
                   << "]" << ENDL();
        }
        if (_u25_hits_stomp) {
            DPRINT << "[U25_RCB_PREF_STOMP_HIT cb=" << remote_cb_id
                   << " config_ptr=0x" << HEX() << _u25_config_ptr
                   << " dest=0x" << _u25_dest_addr
                   << " value=0x" << _u25_value
                   << "]" << ENDL();
        }
    }
#endif
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
    // reset noc counters here because we didn't properly update ptrs for better perf.
    if (noc_mode == DM_DEDICATED_NOC) {
        ncrisc_noc_counters_init();
    } else {
        dynamic_noc_local_state_init();
    }
    // signal reader can exit, since reader cannot exit early due to the ongoing traffic on the same noc.
    cb_reserve_back(sync_cb_id, 1);
    cb_push_back(sync_cb_id, 1);
}
