// Sonata TMARB Interpreter — host-side runtime maker
//
// Receives a pre-serialized flat schedule binary from Python, allocates
// GM shared memory + heap, and initializes the PTO2 runtime structures.
//
// Exports the three symbols required by simpler's runtime framework:
//   prepare_callable_impl, bind_callable_to_runtime_impl, validate_runtime_impl

#include <cstdint>
#include <cstring>

#include "flat_schedule.h"   // FlatSchedule, FlatRegion, etc.
#include "runtime.h"          // TMARB Runtime struct
#include "device_arena.h"    // DeviceArena
#include "pto_runtime2.h"    // runtime_reserve_layout, etc.
#include "pto_shared_memory.h"

// ── prepare_callable_impl ──
//
// Called once per callable. Allocates the prebuilt arena that will hold
// the PTO2Runtime, and uploads the flat schedule binary to GM.

extern "C" int prepare_callable_impl(
    void* /*callable*/,
    void* /*upload_fn*/,
    void* artifacts_out
) {
    // Not yet implemented — the interpreter path (aicpu_entry) handles
    // all initialization on-device. Return an explicit error code so
    // callers get a clear failure rather than a silent success.
    return -10;
}

// ── bind_callable_to_runtime_impl ──
//
// Called before each run. Creates the PTO2Runtime from the prebuilt
// arena, initializes SM and heap, and passes control to the AICPU.

extern "C" int bind_callable_to_runtime_impl(
    void* runtime,
    void* /*orch_args*/,
    void* prebuilt_arena,
    void* /*arg_directions*/,
    int /*arg_count*/
) {
    // Not yet implemented — the interpreter path (aicpu_entry) handles
    // all runtime initialization on-device.
    return -11;
}

// ── validate_runtime_impl ──
//
// Called after device execution completes. Copies output tensors
// back to host and frees device allocations.

extern "C" int validate_runtime_impl(void* runtime) {
    // Not yet implemented — schedule outputs are consumed via the
    // standard output tensor mechanism in simpler's runtime framework.
    return -12;
}
