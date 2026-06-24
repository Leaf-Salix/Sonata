// sonata_hook.h — Minimal C interface for Sonata schedule hook.
//
// This header is self-contained (no pypto or C++ dependencies) so that
// upstream PyPTO's runtime runner can optionally call into a Sonata
// schedule consumer without introducing a hard dependency.
//
// ## Ownership
//
// The caller owns the schedule blob passed to process_schedule;
// the hook does not take ownership and must not retain a pointer beyond
// the call.
//
// ## Fail-open semantics
//
// Every failure mode degrades gracefully to the original PyPTO execution
// path. The hook never crashes or blocks execution:
//
//   SONATA_HOOK_DISABLED (3)  — SONATA_ENABLED not set or hook compiled out
//   SONATA_HOOK_SKIP    (1)  — No valid schedule available
//   SONATA_HOOK_ERROR   (2)  — Schedule corrupt, CRC mismatch, or version
//                              mismatch; original path runs
//   SONATA_HOOK_OK      (0)  — Schedule processed, interpreter ran
//
// ## Thread safety
//
// The hook is safe for single-threaded callers.  The global enable flag
// uses std::atomic<bool> (C++ build) for basic concurrent read/write safety.
// Concurrent calls to process_schedule() from multiple threads are NOT
// supported; the caller must serialize access.
//
// ## Environment variables
//
//   SONATA_ENABLED=1       — Enable the hook (checked at init() time)
//   SONATA_SCHEDULE_PATH   — Path to sonata_schedule.bin (host-side only;
//                            read by runtime_maker.cpp, not by the hook)
//
// ## Binary format versions
//
//   v1 (initial)     — 88-byte FlatSchedule header + struct arrays
//   v2 (current)     — +4-byte CRC-32 at offset 88 (before arrays)
//   Accepts both v1 and v2 on input; produces v2 on output.
//
// ## Schedule info (sonata_hook_info)
//
// Reads the header fields from a binary blob without processing the
// schedule.  Returns HOOK_ERROR when the blob is invalid.
// The fingerprint field is a 64-byte null-terminated identifier that
// uniquely identifies the schedule contract.

#ifndef SONATA_HOOK_H
#define SONATA_HOOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes

#define SONATA_HOOK_OK        0   // Schedule processed, interpreter ran
#define SONATA_HOOK_SKIP      1   // No schedule available; original path
#define SONATA_HOOK_ERROR     2   // Schedule invalid / read error; original path
#define SONATA_HOOK_DISABLED  3   // Hook compiled out or globally disabled

// Schedule info returned by sonata_hook_info()

typedef struct {
    int32_t  num_regions;
    int32_t  total_tasks;
    int32_t  total_args;
    int32_t  total_deps;
    char     fingerprint[64];
} SonataScheduleInfo;

// ── Lifecycle ──

// Initialize hook.  Call once at process start.
// Reads SONATA_ENABLED env var; returns DISABLED when not set.
// Returns SONATA_HOOK_OK, SONATA_HOOK_DISABLED.
int sonata_hook_init(void);

// Process a schedule binary blob.  Called once per compile output.
// The blob must be a valid FlatSchedule (magic=0x534F4E41)
// with version 1 (no CRC) or version 2 (v2 with CRC-32).
// Returns SONATA_HOOK_OK / SONATA_HOOK_SKIP / SONATA_HOOK_ERROR.
int sonata_hook_process_schedule(const void* blob, size_t blob_size);

// Finalize hook.  Call once at process exit.
// Resets the global enable flag.
// Returns SONATA_HOOK_OK.
int sonata_hook_fini(void);

// ── Introspection ──

// Read info from a schedule blob without processing it.
// Returns SONATA_HOOK_OK on success, SONATA_HOOK_ERROR for invalid blob
// or null output pointer.
int sonata_hook_info(const void* blob, size_t blob_size, SonataScheduleInfo* out);

#ifdef __cplusplus
}
#endif

#endif // SONATA_HOOK_H
