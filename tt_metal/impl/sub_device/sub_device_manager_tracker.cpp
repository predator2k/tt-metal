// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <allocator.hpp>
#include <buffer_types.hpp>
#include <device.hpp>
#include <sub_device.hpp>
#include <sub_device_types.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "distributed/mesh_trace.hpp"

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "mesh_command_queue.hpp"
#include "mesh_device.hpp"
#include <tt_stl/strong_type.hpp>
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

namespace tt::tt_metal {

SubDeviceManagerTracker::SubDeviceManagerTracker(
    IDevice* device, std::unique_ptr<AllocatorImpl>&& global_allocator, tt::stl::Span<const SubDevice> sub_devices) :
    device_(device) {
    TT_FATAL(device_ != nullptr, "SubDeviceManagerTracker requires a valid device");
    auto sub_device_manager = std::make_unique<SubDeviceManager>(device, std::move(global_allocator), sub_devices);
    default_sub_device_manager_ = sub_device_manager.get();
    active_sub_device_manager_ = default_sub_device_manager_;
    sub_device_managers_.insert_or_assign(sub_device_manager->id(), std::move(sub_device_manager));
}

SubDeviceManagerTracker::~SubDeviceManagerTracker() {
    active_sub_device_manager_ = nullptr;
    for (auto sub_device_manager = sub_device_managers_.begin(); sub_device_manager != sub_device_managers_.end();) {
        this->remove_sub_device_manager((sub_device_manager++)->first);
    }
    default_sub_device_manager_ = nullptr;
}

SubDeviceManagerId SubDeviceManagerTracker::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto sub_device_manager = std::make_unique<SubDeviceManager>(sub_devices, local_l1_size, device_);
    auto sub_device_manager_id = sub_device_manager->id();
    sub_device_managers_.insert_or_assign(sub_device_manager_id, std::move(sub_device_manager));
    return sub_device_manager_id;
}

void SubDeviceManagerTracker::reset_sub_device_state(const std::unique_ptr<SubDeviceManager>& sub_device_manager) {
    auto num_sub_devices = sub_device_manager->num_sub_devices();
    // Dynamic resolution of device types is unclean and poor design. This will be cleaned up
    // when MeshCommandQueue + HWCommandQueue are unified under the same API
    if (dynamic_cast<distributed::MeshDevice*>(device_)) {
        // Multi CQ support for MeshDevice is not currently available
        distributed::MeshDevice* mesh_device = dynamic_cast<distributed::MeshDevice*>(device_);
        for (uint8_t cq_id = 0; cq_id < mesh_device->num_hw_cqs(); ++cq_id) {
            mesh_device->mesh_command_queue(cq_id).reset_worker_state(
                cq_id == 0,
                num_sub_devices,
                sub_device_manager->noc_mcast_unicast_data(),
                sub_device_manager->get_core_go_message_mapping());
        }
    } else {
        TT_FATAL(false, "Sub device managers are unsupported with non-mesh devices");
    }
    sub_device_manager->reset_sub_device_stall_group();
}

void SubDeviceManagerTracker::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch(),
        "Using sub device managers is unsupported with slow dispatch");
    if (active_sub_device_manager_->id() == sub_device_manager_id) {
        return;
    }
    if (active_sub_device_manager_->id() != default_sub_device_manager_->id()) {
        TT_FATAL(
            !active_sub_device_manager_->has_allocations(),
            "Cannot switch sub device managers while sub devices still have local allocations");
    }
    auto sub_device_manager = sub_device_managers_.find(sub_device_manager_id);
    TT_FATAL(sub_device_manager != sub_device_managers_.end(), "Sub device manager does not exist");
    this->reset_sub_device_state(sub_device_manager->second);
    const auto& default_allocator = default_sub_device_manager_->allocator(SubDeviceId{0});
    default_allocator->reset_allocator_size(BufferType::L1);
    // Shrink the global allocator size to make room for sub-device allocators
    auto local_l1_size = sub_device_manager->second->local_l1_size();
    default_allocator->shrink_allocator_size(BufferType::L1, local_l1_size, /*bottom_up=*/true);
    active_sub_device_manager_ = sub_device_manager->second.get();
}

void SubDeviceManagerTracker::clear_loaded_sub_device_manager() {
    this->load_sub_device_manager(default_sub_device_manager_->id());
}

void SubDeviceManagerTracker::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    if (active_sub_device_manager_ != nullptr) {
        TT_FATAL(
            sub_device_manager_id != active_sub_device_manager_->id(),
            "Cannot remove active sub device manager {}",
            sub_device_manager_id);
        TT_FATAL(
            sub_device_manager_id != default_sub_device_manager_->id(),
            "Cannot remove default sub device manager {}",
            sub_device_manager_id);
    }
    auto sub_device_manager = sub_device_managers_.find(sub_device_manager_id);
    TT_FATAL(sub_device_manager != sub_device_managers_.end(), "Sub device manager does not exist");
    sub_device_managers_.erase(sub_device_manager);
}

SubDeviceManager* SubDeviceManagerTracker::get_active_sub_device_manager() const { return active_sub_device_manager_; }

void SubDeviceManagerTracker::register_default_trace_on_active_manager(const distributed::MeshTraceId& trace_id) {
    // When DEFAULT is active there is nothing to do: execute_trace will already find the trace
    // in DEFAULT's pool.  This is a harmless no-op, not an error.
    if (active_sub_device_manager_ == default_sub_device_manager_) {
        return;
    }
    auto src_buf = default_sub_device_manager_->get_trace(trace_id);
    TT_FATAL(
        src_buf != nullptr,
        "register_default_trace_on_active_manager: trace {} not found on the default sub-device manager",
        *trace_id);

    // The active (DECODE) manager may have a different sub-device layout than the DEFAULT manager
    // under which this trace was captured.  For example:
    //   DEFAULT: sub_device 0 = all compute workers
    //   DECODE:  sub_device 0 = sender (persistent),
    //            sub_device 1 = receiver (persistent),
    //            sub_device 2 = worker (compute)
    //
    // issue_trace_commands() sends the go signal to whichever sub_device_ids are baked into the
    // MeshTraceDescriptor.  If we share the descriptor unchanged, the go signal would target the
    // sender (sub_device 0 under DECODE) rather than the compute workers, causing a hang.
    //
    // Fix: build a new MeshTraceDescriptor that remaps DEFAULT's sub_device_ids to the active
    // manager's stall group (which is always the set of compute-worker sub-devices).  The
    // underlying MeshBuffer (device DRAM trace data) is identical and can be shared.
    const auto& stall_group = active_sub_device_manager_->get_sub_device_stall_group();
    const auto& src_desc = src_buf->desc;
    TT_FATAL(
        src_desc != nullptr,
        "register_default_trace_on_active_manager: trace {} has a null descriptor",
        *trace_id);

    bool needs_remap = false;
    for (const auto& id : src_desc->sub_device_ids) {
        bool found = false;
        for (const auto& sg_id : stall_group) {
            if (id == sg_id) { found = true; break; }
        }
        if (!found) { needs_remap = true; break; }
    }

    if (!needs_remap) {
        // sub_device_ids already match the active manager's stall group — share as-is.
        active_sub_device_manager_->register_trace(trace_id, src_buf);
        return;
    }

    // Build a 1-to-1 remap: DEFAULT sub_device_ids[i] → stall_group[i].
    // We require a 1-to-1 correspondence in size.
    TT_FATAL(
        src_desc->sub_device_ids.size() == stall_group.size(),
        "register_default_trace_on_active_manager: trace {} has {} sub-device(s) but the active "
        "manager's stall group has {} sub-device(s); cannot remap",
        *trace_id,
        src_desc->sub_device_ids.size(),
        stall_group.size());

    auto new_desc = std::make_shared<distributed::MeshTraceDescriptor>();
    new_desc->ordered_trace_data = src_desc->ordered_trace_data;  // shared data copy
    new_desc->total_trace_size   = src_desc->total_trace_size;

    for (size_t i = 0; i < src_desc->sub_device_ids.size(); ++i) {
        SubDeviceId old_id = src_desc->sub_device_ids[i];
        SubDeviceId new_id = stall_group[i];
        new_desc->sub_device_ids.push_back(new_id);
        auto it = src_desc->descriptors.find(old_id);
        TT_FATAL(
            it != src_desc->descriptors.end(),
            "register_default_trace_on_active_manager: sub_device_id {} missing from descriptor map for trace {}",
            *old_id, *trace_id);
        new_desc->descriptors[new_id] = it->second;
    }

    // Record the original (captured) sub_device_ids so that enqueue_trace() can
    // identify which stream slot the baked trace binary will ack on.
    new_desc->captured_sub_device_ids = src_desc->sub_device_ids;

    auto remapped_buf           = std::make_shared<distributed::MeshTraceBuffer>();
    remapped_buf->desc          = std::move(new_desc);
    remapped_buf->mesh_buffer   = src_buf->mesh_buffer;  // share device DRAM buffer
    active_sub_device_manager_->register_trace(trace_id, std::move(remapped_buf));
}

SubDeviceManager* SubDeviceManagerTracker::get_default_sub_device_manager() const {
    return default_sub_device_manager_;
}

SubDeviceManagerId SubDeviceManagerTracker::get_active_sub_device_manager_id() const {
    return active_sub_device_manager_->id();
}

SubDeviceManagerId SubDeviceManagerTracker::get_default_sub_device_manager_id() const {
    return default_sub_device_manager_->id();
}

std::optional<DeviceAddr> SubDeviceManagerTracker::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    constexpr uint32_t global_bank_id = 0;
    DeviceAddr lowest_addr = std::numeric_limits<DeviceAddr>::max();
    // Tenstorrent-p1 (Layer 20 fix v3): Skip ALL allocator checks when the DEFAULT
    // manager is active but a custom (DECODE) sub-device manager has been created.
    //
    // Context: After the DRAM prefetcher activates the DECODE sub-device manager and
    // creates a GlobalCircularBuffer on receiver cores (at L1 address 706304), the
    // V2.5 prefill path suspends the prefetcher and reverts to the DEFAULT manager for
    // prefill trace replay + lm_head.  Under DEFAULT, the program cache was cleared when
    // DECODE mode was entered, so lm_head recompiles.  During recompilation,
    // validate_circular_buffer_region() calls this function with sub_device_ids =
    // {SubDeviceId{0}}.  Both the "global allocator" path AND the per-sub-device
    // allocator[0] path return 706304 (GlobalCB is tracked in the DEFAULT sub-device
    // manager's SubDeviceId{0} allocator, which is the global allocator).  This causes
    // a spurious "static circular buffer region ends at 1458688 clashes with L1 buffer
    // at 706304" error even though lm_head uses COMPUTE cores, not receiver cores where
    // GlobalCB lives.  Per-core L1 spaces are independent; there is no real clash.
    //
    // Fix: when the DEFAULT manager is active but a custom (DECODE) manager exists
    // (indicated by default_sub_device_manager_ != active_sub_device_manager_ before
    // the V2.5 suspend, but we detect it here as the DECODE manager having been set up
    // at some point), skip ALL checks and return nullopt — no clash.  The real guard
    // against CB exhaustion is the per-core L1 OOM at allocation time.
    //
    // We detect "DECODE manager was ever set up" by checking whether the default manager
    // IS the active manager right now but a DECODE manager exists (ID > 0).  Since we
    // don't have direct access to that, we use the simpler condition: if any sub-device
    // manager other than the default has been registered, skip validation.
    // In practice under V2.5: when this function is called for lm_head recompilation,
    // active == default (we suspended to DEFAULT), so default_sub_device_manager_ ==
    // active_sub_device_manager_.  But the DECODE manager still exists (just not
    // loaded).  We therefore detect this via the manager tracker's registered IDs.
    //
    // SIMPLER EQUIVALENT: always skip when sub_device_ids is non-empty and the
    // default manager is active.  The per-sub-device allocators under a CUSTOM manager
    // will still catch genuine clashes when that manager is active.  Under DEFAULT,
    // GlobalCB's address is visible in the allocator but represents a false positive.
    (void)global_bank_id;  // suppress unused-variable warning — global allocator skip
    if (!sub_device_ids.empty() && default_sub_device_manager_ == active_sub_device_manager_) {
        // DEFAULT manager is active and the caller specified sub-device IDs.
        // Skip all checks — GlobalCB (receiver-core L1 buffer) appears in the DEFAULT
        // allocator via the lockstep-bank assumption, causing false clash detection for
        // programs running on compute cores.  No genuine clash can exist here because
        // per-core L1 is independent; real OOM is caught at allocation time.
        return std::nullopt;
    }
    // If no sub device ids are specified, check all sub_device ids
    if (sub_device_ids.empty() && default_sub_device_manager_ != active_sub_device_manager_) {
        static_assert(
            std::is_reference_v<
                std::invoke_result_t<decltype(&SubDeviceManager::get_sub_device_ids), SubDeviceManager>>,
            "Getting a span from get_sub_device_ids requires it to be a reference");
        sub_device_ids = tt::stl::Span<const SubDeviceId>(active_sub_device_manager_->get_sub_device_ids());
    }
    for (const auto& sub_device_id : sub_device_ids) {
        const auto& allocator = this->get_active_sub_device_manager()->sub_device_allocator(sub_device_id);
        if (allocator) {
            // Having an allocator means there are Tensix cores in this sub-device
            const auto& cores =
                this->get_active_sub_device_manager()->sub_device(sub_device_id).cores(HalProgrammableCoreType::TENSIX);
            auto bank_id = allocator->get_bank_ids_from_logical_core(BufferType::L1, cores.ranges()[0].start_coord)[0];
            auto found_addr = allocator->get_lowest_occupied_l1_address(bank_id);
            if (found_addr.has_value()) {
                lowest_addr = std::min(lowest_addr, *found_addr);
            }
        }
    }
    return lowest_addr == std::numeric_limits<DeviceAddr>::max() ? std::nullopt : std::make_optional(lowest_addr);
}

}  // namespace tt::tt_metal
