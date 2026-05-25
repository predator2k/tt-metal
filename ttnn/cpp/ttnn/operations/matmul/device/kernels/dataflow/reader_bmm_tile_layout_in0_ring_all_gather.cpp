// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/debug/dprint.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

void kernel_main() {
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);

    // All Gather specific
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE) {
        return;
    }
#ifdef SGLANG_TT_U20_DATAFLOW_PROBE
    // U20 ENTRY probe — read local L1 0xa6700 BEFORE this kernel does
    // any work.  Bounded budget; per-kernel-instance running totals.
    // If ENTRY = zero across many entries but EXIT = NONZERO, this
    // kernel is the L1 0xa6700 stomper.
    {
        static uint32_t u20_in0_entry_budget = 2048;
        static uint32_t u20_in0_entry_total = 0;
        u20_in0_entry_total++;
        if (u20_in0_entry_budget > 0) {
            u20_in0_entry_budget--;
            volatile tt_l1_ptr uint32_t* l1p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0xa6700);
            uint32_t v0 = l1p[0], v1 = l1p[1], v2 = l1p[2], v3 = l1p[3];
            bool nz = v0 != 0 || v1 != 0 || v2 != 0 || v3 != 0;
            DPRINT << "[U20_ENTRY in0_ring_ag l1=0xa6700"
                   << " w0=0x" << HEX() << v0
                   << " w1=0x" << v1
                   << " w2=0x" << v2
                   << " w3=0x" << v3
                   << " " << DEC() << "tot=" << u20_in0_entry_total
                   << " " << (nz ? "NONZERO" : "zero")
                   << "]" << ENDL();
        }
    }
#endif
    bool is_hop_core = core_type == (uint32_t)CORE_TYPE::HOP_CORE;

    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc_id = get_arg_val<uint32_t>(rt_args_idx++);
    bool end_of_hop = (bool)get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* unpadded_in0_shard_widths_in_tiles = nullptr;
    if (!is_hop_core) {
        unpadded_in0_shard_widths_in_tiles = (uint32_t*)get_arg_addr(rt_args_idx);
        rt_args_idx += ring_size;
    }

    experimental::Noc noc_obj(noc_id);
    experimental::Semaphore<> signal_sem(get_compile_time_arg_val(4));

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_id_in2 = get_named_compile_time_arg_val("cb_in2");

    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in2(cb_id_in2);

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles * in0_single_tile_size_bytes;

    // Reserving/pushing the local shard is done in compute
    cb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);

    uint32_t local_shard_read_addr = cb_in0.get_read_ptr();
    uint32_t l1_write_addr_in0 = cb_in2.get_write_ptr();

    uint32_t hop_core_offset = static_cast<uint32_t>(is_hop_core);
#ifdef SGLANG_TT_U20_DATAFLOW_PROBE
    // U20 ADDR probe — dump cb_in0 / cb_in2 L1 addresses so we can see
    // whether the in0 sender's NoC writes target (next_core, 0xa6700).
    // If cb_in2.get_write_ptr() == 0xa6700 on any core, then the in0
    // ring all-gather mcasts INTO the receiver's W2 output L1, which
    // would be the stomp source.
    {
        static uint32_t u20_addr_in0_budget = 256;
        if (u20_addr_in0_budget > 0) {
            u20_addr_in0_budget--;
            DPRINT << "[U20_ADDR in0_ring_ag"
                   << " local_in0=0x" << HEX() << local_shard_read_addr
                   << " cb_in2_wr=0x" << l1_write_addr_in0
                   << " shard_size_bytes=0x" << shard_size_bytes
                   << " ring_size=" << DEC() << ring_size
                   << " next_x=" << next_core_noc_x
                   << " next_y=" << next_core_noc_y
                   << "]" << ENDL();
        }
    }
#endif

    for (uint32_t shard_cnt = hop_core_offset; shard_cnt < ring_size; shard_cnt++) {
        uint32_t curr_ring_idx = (ring_idx + shard_cnt) % ring_size;
        bool skip_send = !is_hop_core && unpadded_in0_shard_widths_in_tiles[curr_ring_idx] == 0;

        uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * (shard_cnt - hop_core_offset);
        uint32_t curr_shard_read_addr =
            shard_cnt == 0 ? local_shard_read_addr : l1_write_addr_in0 + shard_size_bytes * (shard_cnt - 1);

        // Wait for signal from previous core that data has been added to this core's in0
        signal_sem.wait_min(shard_cnt);

        // Send data to next core
        if (shard_cnt < ring_size - 1 || is_hop_core) {  // Skip sending the last shard
            if (!skip_send) {
                experimental::UnicastEndpoint dst_ep;
                noc_obj.async_write(
                    experimental::CoreLocalMem<uint32_t>(curr_shard_read_addr),
                    dst_ep,
                    shard_size_bytes,
                    {},
                    {.noc_x = next_core_noc_x, .noc_y = next_core_noc_y, .addr = curr_shard_write_addr});
            }

            // Signal the next core that data is ready
            signal_sem.up(noc_obj, next_core_noc_x, next_core_noc_y, 1);
        }

        // Do stuff for matmul fusion here
        if (shard_cnt > 0) {
            cb_in2.push_back(shard_size_in_tiles);
        }
    }

    if (!is_hop_core) {
        for (uint32_t b = 0; b < batch - 1; ++b) {  // for rest batches, not need to gather in0 anymore
            cb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);
            cb_in2.push_back((ring_size - 1) * shard_size_in_tiles);
        }
    }
    noc_obj.async_atomic_barrier();
#ifdef SGLANG_TT_U20_DATAFLOW_PROBE
    // U20 EXIT probe — read local L1 0xa6700 AFTER this kernel finishes
    // all its NoC writes (atomic barrier ensures cross-core writes
    // landed remotely; local L1 here is the SAME core's view).
    // tensix_sync() ensures any in-flight writes from this RISC to
    // local L1 are visible before we read.
    {
        static uint32_t u20_in0_exit_budget = 2048;
        static uint32_t u20_in0_exit_total = 0;
        u20_in0_exit_total++;
        if (u20_in0_exit_budget > 0) {
            u20_in0_exit_budget--;
            // Drain any pending NoC ops to ensure local L1 view is fresh.
            noc_async_read_barrier();
            noc_async_write_barrier();
            volatile tt_l1_ptr uint32_t* l1p =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0xa6700);
            uint32_t v0 = l1p[0], v1 = l1p[1], v2 = l1p[2], v3 = l1p[3];
            bool nz = v0 != 0 || v1 != 0 || v2 != 0 || v3 != 0;
            DPRINT << "[U20_EXIT in0_ring_ag l1=0xa6700"
                   << " w0=0x" << HEX() << v0
                   << " w1=0x" << v1
                   << " w2=0x" << v2
                   << " w3=0x" << v3
                   << " " << DEC() << "tot=" << u20_in0_exit_total
                   << " " << (nz ? "NONZERO" : "zero")
                   << "]" << ENDL();
        }
    }
#endif
}
