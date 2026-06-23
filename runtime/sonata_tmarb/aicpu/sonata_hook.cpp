// sonata_hook.cpp — Implementation of the Sonata schedule hook.
//
// Validates the flat schedule blob (magic, version, bounds) and forwards
// it to aicpu_entry for device-side interpretation.
//
// Design: the hook is stateless except for a global enabled flag.
// Each process_schedule call is independent; no persistent state survives
// across calls. This keeps the hook safe for multi-threaded runtimes.

#include "sonata_hook.h"
#include "flat_schedule.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

// Global enable flag. Set by SONATA_ENABLED env var at init time.
static bool g_sonata_enabled = false;

// Forward declaration: device-side entry point (aicpu_executor.cpp)
extern "C" int aicpu_entry(void* prebuilt_arena, uint64_t arena_size,
                           void* sm_ptr, uint64_t sm_size,
                           void* gm_heap, uint64_t heap_size,
                           int32_t aic_count, int32_t aiv_count,
                           int32_t task_window_size,
                           const FlatSchedule* flat_sched,
                           const void* tensor_registry,
                           int32_t tensor_registry_size);

// ── Helpers ──

static bool validate_schedule(const void* blob, size_t blob_size) {
    if (blob == nullptr || blob_size < sizeof(FlatSchedule)) {
        return false;
    }
    auto* sched = static_cast<const FlatSchedule*>(blob);
    if (sched->magic != 0x534F4E41) {  // "SONA"
        return false;
    }
    if (sched->version != 1) {
        return false;
    }
    if (sched->num_regions < 0 || sched->total_tasks < 0 ||
        sched->total_args < 0 || sched->total_deps < 0) {
        return false;
    }
    // Verify total size covers all arrays
    size_t expected = sizeof(FlatSchedule)
        + static_cast<size_t>(sched->num_regions) * sizeof(FlatRegion)
        + static_cast<size_t>(sched->total_tasks) * sizeof(FlatTask)
        + static_cast<size_t>(sched->total_args) * sizeof(FlatArg)
        + static_cast<size_t>(sched->total_deps) * sizeof(FlatDep);
    if (blob_size < expected) {
        return false;
    }
    return true;
}

// ── Lifecycle ──

extern "C" int sonata_hook_init(void) {
    const char* env = std::getenv("SONATA_ENABLED");
    if (env != nullptr && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0)) {
        g_sonata_enabled = true;
    }
    return SONATA_HOOK_OK;
}

extern "C" int sonata_hook_fini(void) {
    g_sonata_enabled = false;
    return SONATA_HOOK_OK;
}

// ── Process schedule ──

extern "C" int sonata_hook_process_schedule(const void* blob, size_t blob_size) {
    if (!g_sonata_enabled) {
        return SONATA_HOOK_DISABLED;
    }
    if (!validate_schedule(blob, blob_size)) {
        return SONATA_HOOK_ERROR;
    }

    auto* sched = const_cast<FlatSchedule*>(static_cast<const FlatSchedule*>(blob));

    // Call device-side interpreter.
    // In a real NPU runtime, the host-side framework would have already:
    //   1. Uploaded AIC/AIV kernel binaries to device memory
    //   2. Built the tensor_registry from orch_args
    //   3. Allocated prebuilt_arena, SM, GM heap
    //
    // For now, aicpu_entry is called with null resource pointers.
    // This is sufficient for the simulator path and fail-open tests.
    // The real host-side integration (runtime_maker.cpp) will fill these in
    // when the simpler runtime framework calls prepare/bind/validate.
    int rc = aicpu_entry(
        nullptr, 0,   // prebuilt_arena
        nullptr, 0,   // sm_ptr
        nullptr, 0,   // gm_heap
        0, 0,          // aic_count, aiv_count
        0,             // task_window_size
        sched,
        nullptr,       // tensor_registry
        0              // tensor_registry_size
    );

    if (rc == 0) {
        return SONATA_HOOK_OK;
    }
    return SONATA_HOOK_ERROR;
}

// ── Introspection ──

extern "C" int sonata_hook_info(const void* blob, size_t blob_size,
                                SonataScheduleInfo* out) {
    if (out == nullptr) {
        return SONATA_HOOK_ERROR;
    }
    if (!validate_schedule(blob, blob_size)) {
        return SONATA_HOOK_ERROR;
    }
    auto* sched = static_cast<const FlatSchedule*>(blob);
    out->num_regions = sched->num_regions;
    out->total_tasks = sched->total_tasks;
    out->total_args  = sched->total_args;
    out->total_deps  = sched->total_deps;
    std::memcpy(out->fingerprint, sched->fingerprint, sizeof(out->fingerprint));
    return SONATA_HOOK_OK;
}
