// Sonata TMARB Interpreter — device-side AICPU executor
//
// Replaces the TMARB orchestrator thread. Instead of dlopen'ing an
// orchestration .so, it reads the schedule from GM shared memory and
// interprets it directly:
//
//   static region  → PTO2_SCOPE(MANUAL/AUTO) + submit_task + set_dependencies
//   dynamic region → PTO2_SCOPE(AUTO) { }  (TensorMap auto-dependency)
//
// The scheduler threads (0/1/2) run the standard TMARB scheduler.

#include <cstdint>
#include <cstring>

#include "runtime.h"
#include "device_arena.h"
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_orchestrator.h"
#include "pto_types.h"
#include "pto_orchestration_api.h"
#include "pto_scheduler.h"
#include "aicore_completion_mailbox.h"

using namespace sonata::tmarb;

// ── Flat schedule layout (must match host layout) ──

struct FlatArg {
    int32_t runtime_slot;
    int16_t direction;  // 0=input, 1=output, 2=inout, 3=nodep, 4=scalar
} __attribute__((packed));

struct FlatTask {
    int32_t task_id;
    int32_t func_id;
    int16_t core_type;  // 0=aic, 1=aiv, 2=mixed
    int16_t num_args;
    int32_t dep_count;
    int32_t dep_start_idx;
} __attribute__((packed));

struct FlatDep {
    int32_t producer;
    int32_t consumer;
} __attribute__((packed));

struct FlatRegion {
    int32_t kind;       // 0=static, 1=dynamic
    int32_t scope_mode; // 0=auto, 1=manual
    int32_t num_tasks;
    int32_t task_start_idx;
    int32_t num_deps;
    int32_t dep_start_idx;
} __attribute__((packed));

struct FlatSchedule {
    int32_t magic;
    int32_t num_regions;
    char fingerprint[64];
    // Followed by: FlatRegion[num_regions], FlatTask[...], FlatArg[...], FlatDep[...]
} __attribute__((packed));

// ── Scheduler context (shared with scheduler threads) ──

static SchedulerContext sched_ctx_;
static AicoreCompletionMailbox mailbox_;

// ── Direction → TensorArgType ──

static TensorArgType direction_to_tensor_type(int16_t dir) {
    switch (dir) {
        case 1:  return TensorArgType::OUTPUT;
        case 2:  return TensorArgType::INOUT;
        case 3:  return TensorArgType::NO_DEP;
        case 4:  return TensorArgType::SCALAR;
        default: return TensorArgType::INPUT;
    }
}

// ── Build an Arg from a FlatTask ──

static void build_args(const FlatTask* ftask, const FlatArg* fargs, Arg& arg) {
    for (int16_t i = 0; i < ftask->num_args; i++) {
        const FlatArg& fa = fargs[i];
        switch (fa.direction) {
            case 0:  arg.add_input(fa.runtime_slot);     break;
            case 1:  arg.add_output(fa.runtime_slot);     break;
            case 2:  arg.add_inout(fa.runtime_slot);     break;
            case 3:  arg.add_no_dep(fa.runtime_slot);    break;
            case 4:
            default: arg.add_scalar(fa.runtime_slot);     break;
        }
    }
}

// ── Set explicit dependencies from FlatDep list ──

static void set_task_dependencies(const FlatTask* ftask, const FlatDep* fdeps,
                                   const PTO2TaskId* task_ids, Arg& arg) {
    if (ftask->dep_count == 0) return;
    PTO2TaskId deps[64];  // max in-flight deps
    int32_t count = 0;
    for (int32_t i = 0; i < ftask->dep_count && count < 64; i++) {
        const FlatDep& fd = fdeps[ftask->dep_start_idx + i];
        // task_id → PTO2TaskId (ring_id=0 for flat schedule)
        deps[count++] = PTO2TaskId(0, fd.producer);
    }
    arg.set_dependencies(deps, count);
}

// ── Interpreter main: called by orchestrator thread ──

static void interpret_schedule(PTO2Runtime* rt, const FlatSchedule* sched,
                                const FlatRegion* regions, const FlatTask* tasks,
                                const FlatArg* args, const FlatDep* deps) {
    int32_t task_offset = 0;
    int32_t dep_offset = 0;

    for (int32_t r = 0; r < sched->num_regions; r++) {
        const FlatRegion& rg = regions[r];

        if (rg.kind == 1) {
            // ── Dynamic region: TensorMap auto path ──
            rt_scope_begin(rt);
            rt_scope_end(rt);
            continue;
        }

        // ── Static region: explicit tasks + deps ──
        PTO2ScopeMode scope = (rg.scope_mode == 1) ? PTO2ScopeMode::MANUAL : PTO2ScopeMode::AUTO;
        rt_scope_begin(rt, scope);

        // Track submitted task IDs for dependency references
        PTO2TaskId task_ids[16384];
        int32_t num_submitted = 0;

        for (int32_t t = 0; t < rg.num_tasks; t++) {
            const FlatTask& ft = tasks[rg.task_start_idx + t];
            const FlatArg* task_args = &args[task_offset];
            Arg arg;

            build_args(&ft, task_args, arg);
            set_task_dependencies(&ft, deps, task_ids, arg);

            TaskOutputTensors result;
            if (ft.core_type == 1) {
                result = rt_submit_aiv_task(ft.func_id, arg);
            } else if (ft.core_type == 2) {
                MixedKernels mk{ft.func_id, ft.func_id, INVALID_KERNEL_ID};
                result = rt_submit_task(mk, arg);
            } else {
                result = rt_submit_aic_task(ft.func_id, arg);
            }

            task_ids[num_submitted++] = result.task_id();
            task_offset += ft.num_args;
        }

        rt_scope_end(rt);
    }
}

// ── AICPU executor entry point ──
//
// Called by simpler runtime after host-side prepare + bind.
// Replaces the orchestrator thread logic from TMARB's aicpu_executor.cpp.

extern "C" int aicpu_entry(const ChipStorageTaskArgs& /*orch_args*/,
                           void* prebuilt_arena, uint64_t arena_size,
                           void* sm_ptr, uint64_t sm_size,
                           void* gm_heap, uint64_t heap_size,
                           int32_t aic_count, int32_t aiv_count,
                           int32_t task_window_size) {

    // ── Phase 1: Attach arena and compute layout ──
    DeviceArena runtime_arena;
    runtime_arena.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);

    PTO2RuntimeArenaLayout layout;
    int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
    {
        DeviceArena layout_arena;
        layout_arena.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
        layout = runtime_reserve_layout(layout_arena, task_window_size, dep_pool_capacity);
        runtime_arena.seal();
    }

    // ── Phase 2: Write data into arena offsets ──
    PTO2Runtime* rt = runtime_init_data_from_layout(
        runtime_arena, layout, PTO2_MODE_EXECUTE,
        sm_ptr, sm_size, gm_heap, heap_size
    );
    if (!rt) return -1;

    // ── Phase 3: Wire arena pointers ──
    runtime_wire_arena_pointers(runtime_arena, layout, rt);

    // ── Reset shared memory and mailbox ──
    std::memset(rt->sm_handle, 0, sizeof(*rt->sm_handle));
    rt->sm_handle->init(sm_ptr, sm_size, task_window_size, heap_size);
    std::memset(&mailbox_, 0, sizeof(mailbox_));

    // ── Phase 4: Finalize device-side fields ──
    runtime_finalize_after_wire(rt, aic_count, aiv_count);

    // ── Bind runtime to TLS ──
    framework_bind_runtime(rt);

    // ── Parse the flat schedule from GM ──
    // The schedule data is stored at a fixed offset in GM.
    // Host-side runtime_maker wrote it before launching the device.
    auto* sched = static_cast<const FlatSchedule*>(sm_ptr);
    if (sched->magic != 0x534F4E41) return -2;  // "SONA" magic

    auto* regions = reinterpret_cast<const FlatRegion*>(sched + 1);
    int32_t task_count = 0, dep_count = 0;
    for (int32_t i = 0; i < sched->num_regions; i++) {
        task_count += regions[i].num_tasks;
        dep_count += regions[i].num_deps;
    }
    // Layout: regions | tasks | args[per-task sum] | deps
    auto* tasks = reinterpret_cast<const FlatTask*>(regions + sched->num_regions);
    auto* args  = reinterpret_cast<const FlatArg*>(tasks + task_count);
    auto* deps  = reinterpret_cast<const FlatDep*>(args + task_count * 4);  // max 4 args/task

    // ── Run the interpreter ──
    interpret_schedule(rt, sched, regions, tasks, args, deps);
    rt_orchestration_done(rt);

    // ── Signal scheduler ──
    sched_ctx_.on_orchestration_done(rt, 3, 0);

    return 0;
}
