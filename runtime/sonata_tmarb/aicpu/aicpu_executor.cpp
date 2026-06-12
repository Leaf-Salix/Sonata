// Sonata TMARB Interpreter — device-side AICPU executor
//
// Replaces the TMARB orchestrator thread. Reads the schedule from a flat
// binary blob in GM and interprets it directly.
//
//   static region  → PTO2_SCOPE + submit_task + set_dependencies
//   dynamic region → PTO2_SCOPE(AUTO) { }  (TensorMap auto-dependency)

#include <cstdint>
#include <cstring>

#include "flat_schedule.h"       // FlatSchedule, FlatRegion, FlatTask, FlatArg, FlatDep
#include "runtime.h"              // TMARB Runtime struct
#include "utils/device_arena.h"  // DeviceArena
#include "pto_runtime2.h"        // runtime_reserve_layout, runtime_init_data_from_layout, etc.
#include "pto_runtime2_types.h"  // PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE, etc.
#include "pto_shared_memory.h"   // PTO2SharedMemoryHandle, PTO2SharedMemoryHeader
#include "pto_types.h"           // Arg, TensorArgType, PTO2ScopeMode, MixedKernels, TaskOutputTensors
#include "pto_orchestration_api.h"  // rt_submit_aic_task, rt_submit_task, rt_scope_begin, rt_scope_end

using namespace pto;

// ── Build Arg from FlatTask ──

static void build_arg(const FlatTask* ftask, const FlatArg* fargs, Arg& arg) {
    for (int16_t i = 0; i < ftask->num_args; i++) {
        const FlatArg& fa = fargs[i];
        switch (fa.direction) {
            case 1:  arg.add_output(fa.runtime_slot);    break;
            case 2:  arg.add_inout(fa.runtime_slot);     break;
            case 3:  arg.add_no_dep(fa.runtime_slot);    break;
            case 4:  arg.add_scalar(fa.runtime_slot);    break;
            default: arg.add_input(fa.runtime_slot);     break;
        }
    }
}

// ── Set dependencies from FlatDep list ──

static void set_deps(const FlatTask* ftask, const FlatDep* fdeps,
                     const PTO2TaskId* task_ids, int32_t num_submitted, Arg& arg) {
    if (ftask->dep_count == 0 || task_ids == nullptr) return;
    PTO2TaskId deps[64];
    uint32_t count = 0;
    for (int32_t i = 0; i < ftask->dep_count && count < 64; i++) {
        const FlatDep& fd = fdeps[ftask->dep_start + i];
        if (fd.producer >= 0 && fd.producer < num_submitted) {
            deps[count++] = task_ids[fd.producer];
        }
    }
    if (count > 0) {
        arg.set_dependencies(deps, count);
    }
}

// ── Main interpreter loop ──

static void interpret_schedule(PTO2Runtime* rt, const FlatSchedule* sched,
                                const FlatRegion* regions, const FlatTask* tasks,
                                const FlatArg* args, const FlatDep* fdeps) {

    for (int32_t r = 0; r < sched->num_regions; r++) {
        const FlatRegion& rg = regions[r];

        if (rg.kind == 1) {
            rt->pending_scope_mode = PTO2ScopeMode::AUTO;
            rt_scope_begin(rt);
            rt_scope_end(rt);
            continue;
        }

        // Static region — explicit tasks + deps
        rt->pending_scope_mode = (rg.scope_mode == 1) ? PTO2ScopeMode::MANUAL : PTO2ScopeMode::AUTO;
        rt_scope_begin(rt);

        // Capture PTO2TaskId for dependency resolution
        PTO2TaskId task_ids[16384];
        int32_t num_submitted = 0;
        int32_t arg_cursor = 0;  // running offset into flat arg array

        for (int32_t t = 0; t < rg.num_tasks; t++) {
            const FlatTask& ft = tasks[rg.task_start + t];
            Arg submit_arg;

            build_arg(&ft, &args[arg_cursor], submit_arg);
            set_deps(&ft, fdeps, task_ids, num_submitted, submit_arg);

            TaskOutputTensors result;
            if (ft.core_type == 1) {
                result = rt_submit_aiv_task(ft.func_id, submit_arg);
            } else if (ft.core_type == 2) {
                MixedKernels mk;
                mk.aic_kernel_id = ft.func_id;
                mk.aiv0_kernel_id = INVALID_KERNEL_ID;
                mk.aiv1_kernel_id = INVALID_KERNEL_ID;
                result = rt_submit_task(mk, submit_arg);
            } else {
                result = rt_submit_aic_task(ft.func_id, submit_arg);
            }

            task_ids[num_submitted++] = result.task_id();
            arg_cursor += ft.num_args;
        }

        rt_scope_end(rt);
    }
}

// ── Device-side entry point ──
//
// Called by simpler runtime after host-side bind. Initializes the
// PTO2Runtime from the prebuilt arena, parses the flat schedule,
// and runs the interpreter loop.

extern "C" int aicpu_entry(void* prebuilt_arena, uint64_t arena_size,
                           void* sm_ptr, uint64_t sm_size,
                           void* gm_heap, uint64_t heap_size,
                           int32_t aic_count, int32_t aiv_count,
                           int32_t task_window_size,
                           const FlatSchedule* flat_sched) {

    if (flat_sched == nullptr || flat_sched->magic != 0x534F4E41) {
        return -1;
    }

    // ── Initialize PTO2Runtime from prebuilt arena ──
    DeviceArena runtime_arena;
    runtime_arena.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);

    PTO2RuntimeArenaLayout layout;
    {
        DeviceArena layout_arena;
        layout_arena.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
        layout = runtime_reserve_layout(layout_arena, task_window_size, PTO2_DEP_LIST_POOL_SIZE);
    }

    PTO2Runtime* rt = runtime_init_data_from_layout(
        runtime_arena, layout, PTO2_MODE_EXECUTE,
        sm_ptr, sm_size, gm_heap, heap_size
    );
    if (!rt) return -2;

    runtime_wire_arena_pointers(runtime_arena, layout, rt);

    // Reset SM and mailbox
    std::memset(rt->sm_handle, 0, sizeof(*rt->sm_handle));
    rt->sm_handle->init(sm_ptr, sm_size, task_window_size, heap_size);
    std::memset(rt->aicore_mailbox, 0, sizeof(*rt->aicore_mailbox));

    runtime_finalize_after_wire(rt, aic_count, aiv_count);
    framework_bind_runtime(rt);

    // ── Parse flat schedule ──
    auto* regions = reinterpret_cast<const FlatRegion*>(flat_sched + 1);
    int32_t total_tasks = 0;
    for (int32_t i = 0; i < flat_sched->num_regions; i++) {
        total_tasks += regions[i].num_tasks;
    }

    auto* tasks = reinterpret_cast<const FlatTask*>(regions + flat_sched->num_regions);
    auto* args  = reinterpret_cast<const FlatArg*>(tasks + total_tasks);
    auto* deps  = reinterpret_cast<const FlatDep*>(args + total_tasks * 4);

    // ── Run interpreter ──
    interpret_schedule(rt, flat_sched, regions, tasks, args, deps);
    rt_orchestration_done(rt);

    return 0;
}
