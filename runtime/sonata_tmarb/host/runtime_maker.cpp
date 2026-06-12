// Sonata TMARB — host-side runtime_maker
//
// Reads sonata_schedule.json, parses regions/tasks/deps, initializes
// the shared memory runtime, and passes control to the AICPU executor.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "runtime.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"

using json = nlohmann::json;
using namespace sonata::tmarb;

// ── JSON parsing helpers ──

static ScheduledArg parse_arg(const json& j) {
    ScheduledArg arg;
    arg.arg_identity = j.value("arg_identity", "");
    arg.runtime_slot = j.value("runtime_slot", -1);
    arg.direction = j.value("direction", "input");
    return arg;
}

static ScheduledTask parse_task(const json& j) {
    ScheduledTask task;
    task.task_id = j.value("task_id", 0);
    task.func_id = j.value("func_id", -1);
    task.core_type = j.value("core_type", "aic");
    for (const auto& a : j.value("args", json::array())) {
        task.args.push_back(parse_arg(a));
    }
    return task;
}

static ScheduleDep parse_dep(const json& j) {
    ScheduleDep dep;
    dep.producer = j.value("producer", 0);
    dep.consumer = j.value("consumer", 0);
    dep.kind = j.value("kind", "data");
    return dep;
}

static ScheduledRegion parse_region(const json& j) {
    ScheduledRegion region;
    region.region_id = j.value("region_id", "");
    region.kind = j.value("kind", "static");
    region.scope_mode = j.value("scope_mode", "auto");
    for (const auto& t : j.value("tasks", json::array())) {
        region.tasks.push_back(parse_task(t));
    }
    for (const auto& d : j.value("deps", json::array())) {
        region.deps.push_back(parse_dep(d));
    }
    return region;
}

static bool parse_schedule(const std::string& path, SonataRuntime& sr) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    json j;
    try {
        f >> j;
    } catch (...) {
        return false;
    }

    sr.fingerprint = j.value("fingerprint", "");
    for (const auto& r : j.value("regions", json::array())) {
        sr.regions.push_back(parse_region(r));
    }
    return !sr.regions.empty();
}

// ── Host-side prepare (called once per callable) ──
//
// Reads the schedule, allocates shared memory, and creates the PTO2Runtime.

extern "C" int runtime_prepare(SonataRuntime* sr, const std::string& schedule_path,
                                void* (*sm_alloc)(size_t), void* (*heap_alloc)(size_t)) {
    if (!parse_schedule(schedule_path, *sr)) {
        std::cerr << "[sonata_tmarb] failed to parse " << schedule_path << std::endl;
        return -1;
    }

    // Allocate shared memory and GM heap for the PTO2 runtime
    size_t sm_size = PTO2SharedMemoryHandle::calculate_size(PTO2_TASK_WINDOW_SIZE);
    sr->sm_ptr = sm_alloc(sm_size);
    sr->sm_size = sm_size;
    sr->heap_size = PTO2_HEAP_SIZE;
    sr->gm_heap = heap_alloc(sr->heap_size);
    sr->rt = nullptr;
    sr->orch = nullptr;
    sr->current_region_idx = 0;
    sr->orchestration_done = false;

    return 0;
}

// ── Host-side bind (called every run) ──
//
// Called before device execution. Sets up the shared memory runtime.

extern "C" int runtime_bind(SonataRuntime* sr) {
    if (sr->rt != nullptr) return 0;  // Already initialized

    // The PTO2Runtime is created on-device by aicpu_executor.
    // Host-side binding writes schedule ptr into shared memory for the device.
    return 0;
}
