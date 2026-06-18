// flat_schedule.h — Flat binary schedule format for on-device interpreter.
//
// Memory layout (packed):
//   FlatSchedule      — 88-byte header
//   FlatRegion[N]     — N region descriptors
//   FlatTask[T]       — T task descriptors
//   FlatArg[A]        — A arg bindings
//   FlatDep[D]        — D dependency edges

#ifndef SONATA_TMARB_RUNTIME_H
#define SONATA_TMARB_RUNTIME_H

#include <cstdint>
#include <cstddef>

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
    int32_t version;        // 1
    int32_t num_regions;
    int32_t total_tasks;
    int32_t total_args;     // total arg count across all tasks
    int32_t total_deps;     // total dep count across all regions
    char    fingerprint[64];
};

#pragma pack(pop)

#endif
