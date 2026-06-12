#ifndef SONATA_TMARB_RUNTIME_H
#define SONATA_TMARB_RUNTIME_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <map>

#include "runtime.h"  // TMARB Runtime struct — provides ring buffers, TensorMap, scheduler
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_orchestrator.h"
#include "pto_constants.h"

namespace sonata {
namespace tmarb {

// ── Schedule data structures (deserialized from sonata_schedule.json) ──

struct ScheduledArg {
    std::string arg_identity;
    int32_t runtime_slot;
    std::string direction;  // "input", "output", "inout", etc.
};

struct ScheduledTask {
    int32_t task_id;
    int32_t func_id;  // -1 = unbound
    std::string core_type;  // "aic", "aiv", "mixed"
    std::vector<ScheduledArg> args;
};

struct ScheduleDep {
    int32_t producer;
    int32_t consumer;
    std::string kind;
};

struct ScheduledRegion {
    std::string region_id;
    std::string kind;        // "static", "dynamic"
    std::string scope_mode;  // "auto", "manual"
    std::vector<ScheduledTask> tasks;
    std::vector<ScheduleDep> deps;
};

// ── Sonata TMARB runtime state ──

struct SonataRuntime {
    // Schedule data (loaded once by host, read by AICPU)
    std::vector<ScheduledRegion> regions;
    std::string fingerprint;

    // TMARB runtime infrastructure (shared memory, tensormap, allocator)
    void* sm_ptr;
    size_t sm_size;
    void* gm_heap;
    size_t heap_size;
    PTO2Runtime* rt;  // Created via runtime_create_from_sm()

    // Orchestrator state (used by interpreter loop)
    PTO2OrchestratorState* orch;
    int32_t current_region_idx;
    bool orchestration_done;
};

}  // namespace tmarb
}  // namespace sonata

#endif  // SONATA_TMARB_RUNTIME_H
