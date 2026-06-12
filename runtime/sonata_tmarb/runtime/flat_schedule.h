#ifndef SONATA_TMARB_RUNTIME_H
#define SONATA_TMARB_RUNTIME_H

#include <cstdint>
#include <cstddef>

// ── Flat schedule binary format ──
//
// Produced by sonata.schedule.serialize_to_binary().
// Read by aicpu_executor.cpp on device and host runtime_maker.cpp.
// All structs are packed for consistent layout across host/device.

#pragma pack(push, 1)

struct FlatArg {
    int32_t runtime_slot;
    int16_t direction;  // 0=input, 1=output, 2=inout, 3=nodep, 4=scalar
};

struct FlatTask {
    int32_t task_id;
    int32_t func_id;
    int16_t core_type;  // 0=aic, 1=aiv, 2=mixed
    int16_t num_args;
    int32_t dep_start;
    int32_t dep_count;
};

struct FlatDep {
    int32_t producer;
    int32_t consumer;
};

struct FlatRegion {
    int32_t kind;       // 0=static, 1=dynamic
    int32_t scope_mode; // 0=auto, 1=manual
    int32_t task_start;
    int32_t num_tasks;
    int32_t dep_start;
    int32_t num_deps;
};

struct FlatSchedule {
    int32_t magic;         // 0x534F4E41 = "SONA"
    int32_t num_regions;
    char fingerprint[64];
    // Followed by: FlatRegion[num_regions], FlatTask[], FlatArg[], FlatDep[]
};

#pragma pack(pop)

#endif
