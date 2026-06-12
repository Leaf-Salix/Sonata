// Sonata TMARB Interpreter — host-side runtime maker
//
// Reads sonata_schedule.json, serializes schedule data into GM shared memory,
// allocates PTO2 runtime resources, and initializes the device for execution.
//
// Unlike TMARB, this variant does NOT upload or dlopen an orchestration .so.
// Instead, the schedule data is passed to the AICPU executor which interprets it.

#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "runtime.h"
#include "device_arena.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"

using json = nlohmann::json;
using namespace sonata::tmarb;

// ── JSON parsing ──

static ScheduledArg parse_arg(const json& j) {
    ScheduledArg a;
    a.arg_identity = j.value("arg_identity", "");
    a.runtime_slot = j.value("runtime_slot", -1);
    a.direction = j.value("direction", "input");
    return a;
}

static ScheduledTask parse_task(const json& j) {
    ScheduledTask t;
    t.task_id = j.value("task_id", 0);
    t.func_id = j.value("func_id", INVALID_FUNC_ID);
    t.core_type = j.value("core_type", "aic");
    for (auto& a : j.value("args", json::array()))
        t.args.push_back(parse_arg(a));
    return t;
}

static ScheduleDep parse_dep(const json& j) {
    ScheduleDep d;
    d.producer = j.value("producer", 0);
    d.consumer = j.value("consumer", 0);
    d.kind = j.value("kind", "data");
    return d;
}

static ScheduledRegion parse_region(const json& j) {
    ScheduledRegion r;
    r.region_id = j.value("region_id", "");
    r.kind = j.value("kind", "static");
    r.scope_mode = j.value("scope_mode", "auto");
    for (auto& t : j.value("tasks", json::array())) r.tasks.push_back(parse_task(t));
    for (auto& d : j.value("deps", json::array())) r.deps.push_back(parse_dep(d));
    return r;
}

static bool parse_schedule(const std::string& path, std::vector<ScheduledRegion>& regions, std::string& fp) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    json j;
    try { f >> j; } catch (...) { return false; }
    fp = j.value("fingerprint", "");
    for (auto& r : j.value("regions", json::array()))
        regions.push_back(parse_region(r));
    return !regions.empty();
}

// ── Serialize parsed schedule into a flat GM buffer ──

struct FlatTask {
    int32_t task_id;
    int32_t func_id;
    int16_t core_type;  // 0=aic, 1=aiv, 2=mixed
    int16_t num_args;
    int32_t dep_count;
    int32_t dep_start_idx;  // index into flat_deps array
};

struct FlatArg {
    int32_t runtime_slot;
    int16_t direction;  // 0=input, 1=output, 2=inout, 3=nodep, 4=scalar
};

struct FlatDep {
    int32_t producer;
    int32_t consumer;
};

struct FlatRegion {
    int32_t kind;       // 0=static, 1=dynamic
    int32_t scope_mode; // 0=auto, 1=manual
    int32_t num_tasks;
    int32_t task_start_idx;
    int32_t num_deps;
    int32_t dep_start_idx;
};

struct FlatSchedule {
    int32_t magic;         // 0x534F4E41 = "SONA"
    int32_t num_regions;
    char fingerprint[64];
    // Followed by: FlatRegion[], FlatTask[], FlatArg[], FlatDep[]
};

static int16_t core_type_to_int(const std::string& ct) {
    if (ct == "aiv") return 1;
    if (ct == "mixed") return 2;
    return 0;  // aic
}

static int16_t direction_to_int(const std::string& d) {
    if (d == "output" || d == "outputexisting") return 1;
    if (d == "inout") return 2;
    if (d == "nodep" || d == "no_dep") return 3;
    if (d == "scalar") return 4;
    return 0;  // input
}

// ── Exported host entry points ──

extern "C" int sonata_tmarb_prepare(
    const char* schedule_path,
    SonataRuntime* sr,
    void* (*gm_sm_alloc)(uint64_t),
    void* (*gm_heap_alloc)(uint64_t),
    uint64_t task_window_size
) {
    // Parse schedule
    std::vector<ScheduledRegion> regions;
    std::string fp;
    if (!parse_schedule(schedule_path, regions, fp))
        return -1;

    // Allocate shared memory
    sr->sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);
    sr->sm_ptr = gm_sm_alloc(sr->sm_size);
    sr->heap_size = PTO2_HEAP_SIZE;
    sr->gm_heap = gm_heap_alloc(sr->heap_size);

    // Count totals
    uint32_t num_regions = regions.size();
    uint32_t total_tasks = 0, total_deps = 0;
    for (auto& rg : regions) {
        total_tasks += rg.tasks.size();
        total_deps += rg.deps.size();
    }

    // Calculate flat buffer size
    uint64_t header_size = sizeof(FlatSchedule);
    uint64_t region_size = num_regions * sizeof(FlatRegion);
    uint64_t task_size    = total_tasks * sizeof(FlatTask);
    uint64_t arg_size     = total_tasks * 4 * sizeof(FlatArg);  // estimate 4 args/task
    uint64_t dep_size     = total_deps * sizeof(FlatDep);
    uint64_t total_size   = header_size + region_size + task_size + arg_size + dep_size;

    // Allocate GM memory for schedule data
    sr->schedule_data = gm_sm_alloc(total_size);
    sr->header = (SonataScheduleHeader*)sr->schedule_data;
    std::strncpy(sr->header->fingerprint, fp.c_str(), sizeof(sr->header->fingerprint) - 1);
    sr->header->num_regions = num_regions;
    sr->header->total_tasks = total_tasks;
    sr->header->total_deps = total_deps;
    sr->header->schedule_data_offset = sizeof(SonataScheduleHeader);

    return 0;
}


extern "C" int sonata_tmarb_bind(SonataRuntime* sr) {
    // Called before each run. Host-side binding is minimal — the
    // interpreter runs entirely on-device. The PTO2Runtime creation
    // and scheduler startup happen in aicpu_executor.cpp.
    (void)sr;
    return 0;
}
