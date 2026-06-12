#ifndef SONATA_TMARB_RUNTIME_H
#define SONATA_TMARB_RUNTIME_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

#include "runtime.h"          // TMARB Runtime struct
#include "pto_shared_memory.h"
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"

namespace sonata { namespace tmarb {

constexpr int32_t INVALID_FUNC_ID = -1;

// ── Schedule data (deserialized from sonata_schedule.json) ──

struct ScheduledArg {
    std::string arg_identity;
    int32_t runtime_slot = -1;
    std::string direction = "input";
};

struct ScheduledTask {
    int32_t task_id = 0;
    int32_t func_id = INVALID_FUNC_ID;
    std::string core_type = "aic";
    std::vector<ScheduledArg> args;
};

struct ScheduleDep {
    int32_t producer = 0;
    int32_t consumer = 0;
    std::string kind = "data";
};

struct ScheduledRegion {
    std::string region_id;
    std::string kind = "static";
    std::string scope_mode = "auto";
    std::vector<ScheduledTask> tasks;
    std::vector<ScheduleDep> deps;
};

// ── Shared memory layout for schedule data ──

struct SonataScheduleHeader {
    uint32_t num_regions;
    uint32_t total_tasks;
    uint32_t total_deps;
    uint64_t schedule_data_offset;  // offset from this header to serialized data
    char fingerprint[64];
};

// ── Host-side schedule storage (allocated in GM, read by AICPU) ──

struct SonataRuntime {
    SonataScheduleHeader* header = nullptr;  // points into GM shared memory
    void* schedule_data = nullptr;           // raw serialized schedule bytes
    void* sm_ptr = nullptr;
    size_t sm_size = 0;
    void* gm_heap = nullptr;
    size_t heap_size = 0;
};

}}  // namespace sonata::tmarb

#endif
