// flat_schedule.h — Flat binary schedule format for on-device interpreter.
//
// Memory layout (packed):
//   FlatSchedule      — 88-byte header
//   (v2 only) CRC-32  — 4-byte checksum over payload (arrays + string table)
//   FlatRegion[N]     — N region descriptors
//   FlatTask[T]       — T task descriptors
//   FlatArg[A]        — A arg bindings
//   FlatDep[D]        — D dependency edges
//   (optional) string table — uint16(length) + UTF-8 entries for
//                  kernel_identity (T entries) and arg_identity (A entries).
//                  Appended by Python to_binary; from_binary falls back
//                  to generated names when the string table is absent.
//
// Version history:
//   1 — initial flat binary (magic + counts + fingerprint + arrays)
//   2 — +4-byte CRC-32 of payload at offset 88 (right after header)

#ifndef SONATA_TMARB_RUNTIME_H
#define SONATA_TMARB_RUNTIME_H

#include <cstdint>
#include <cstddef>

// Current binary format version. Bump when the wire layout changes.
static constexpr int32_t BINARY_FORMAT_VERSION = 2;

// FlatSchedule magic identifier ("SONA" = 0x534F4E41).
static constexpr uint32_t FLAT_SCHEDULE_MAGIC = 0x534F4E41;

// Maximum number of dependency edges per task.
static constexpr int32_t MAX_DEPS_PER_TASK = 256;

#pragma pack(push, 1)

struct FlatArg {
    int32_t runtime_slot;   // tensor map index; -1 if not bound
    int16_t direction;      // 0=input, 1=output, 2=inout, 3=scalar, 4=nodep, 5=outputexisting
};

struct FlatTask {
    int32_t task_id;
    int32_t func_id;
    int16_t core_type;      // 0=aic, 1=aiv, 2=mixed
    int16_t num_args;
    int32_t arg_base;       // offset into arg blob (index of first arg)
};

struct FlatDep {
    int32_t producer;       // task index within region
    int32_t consumer;       // task index within region
};

struct FlatRegion {
    int32_t kind;           // 0=static, 1=dynamic
    int32_t scope_mode;     // 0=auto, 1=manual
    int32_t task_start;
    int32_t num_tasks;
    int32_t dep_start;
    int32_t num_deps;
};

struct FlatSchedule {
    int32_t magic;          // 0x534F4E41 ("SONA")
    int32_t version;        // BINARY_FORMAT_VERSION
    int32_t num_regions;
    int32_t total_tasks;
    int32_t total_args;     // total arg count across all tasks
    int32_t total_deps;     // total dep count across all regions
    char    fingerprint[64];
};

#pragma pack(pop)

#endif
