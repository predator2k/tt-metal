// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t total_size_bytes = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t num_chunks = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t chunk_size_bytes = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t remainder_chunk_size_bytes = get_arg_val<uint32_t>(i);
    i += 1;
    // U23 fix: per-core src/dst L1 base addresses passed as runtime args.
    // For uniform sharded buffers these equal the CB base on every core
    // (preserves legacy behavior).  For per-core-allocated buffers these
    // are the ACTUAL per-core L1 base addresses, which is what we need to
    // compute the correct backward-copy ranges.
    //
    // A value of 0 means "fall back to the CB base" -- preserves any
    // caller that has not been updated to pass per-core overrides (e.g.,
    // if this kernel is ever invoked from outside the move_sharded
    // factory with the older 4-arg layout).
    uint32_t src_addr_override = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t dst_addr_override = get_arg_val<uint32_t>(i);
    i += 1;
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);

    uint32_t src_cb_base_addr = src_addr_override != 0 ? src_addr_override : get_read_ptr(src_cb_id);
    uint32_t dst_cb_base_addr = dst_addr_override != 0 ? dst_addr_override : get_write_ptr(dst_cb_id);

    // Copy from top of src cb to top of dst cb (backwards)
    uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
    uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
    for (uint32_t i = 0; i < num_chunks; i += 1) {
        src_cb_addr -= chunk_size_bytes;
        dst_cb_addr -= chunk_size_bytes;
        noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, chunk_size_bytes);
        noc_async_read_barrier();
    }
    if (remainder_chunk_size_bytes > 0) {
        src_cb_addr -= remainder_chunk_size_bytes;
        dst_cb_addr -= remainder_chunk_size_bytes;
        noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, remainder_chunk_size_bytes);
        noc_async_read_barrier();
    }
}
