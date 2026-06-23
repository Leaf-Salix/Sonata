// sonata_hook.h — Minimal C interface for Sonata schedule hook.
//
// This header is self-contained (no pypto or C++ dependencies) so that
// upstream PyPTO's runtime runner can optionally call into a Sonata
// schedule consumer without introducing a hard dependency.
//
// Ownership: the caller owns the schedule blob passed to process_schedule;
// the hook does not take ownership and must not retain a pointer beyond
// the call.

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

// Lifecycle

// Initialize hook. Call once at process start.
// Returns SONATA_HOOK_OK, SONATA_HOOK_DISABLED.
int sonata_hook_init(void);

// Process a schedule binary blob. Called once per compile output.
// The blob must be a valid FlatSchedule (magic=0x534F4E41, version=1).
// Returns SONATA_HOOK_OK / SONATA_HOOK_SKIP / SONATA_HOOK_ERROR.
int sonata_hook_process_schedule(const void* blob, size_t blob_size);

// Finalize hook. Call once at process exit.
// Returns SONATA_HOOK_OK.
int sonata_hook_fini(void);

// Introspection: read info from a schedule blob without processing it.
// Returns SONATA_HOOK_OK / SONATA_HOOK_ERROR.
int sonata_hook_info(const void* blob, size_t blob_size, SonataScheduleInfo* out);

#ifdef __cplusplus
}
#endif

#endif // SONATA_HOOK_H
