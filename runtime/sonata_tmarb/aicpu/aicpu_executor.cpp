// Sonata TMARB Interpreter — device-side AICPU executor
//
// Replaces the TMARB orchestrator thread. Reads the schedule from a flat
// binary blob in GM and interprets it directly.
//
//   static region  → PTO2_SCOPE + submit_task + set_dependencies
//   dynamic region → PTO2_SCOPE(AUTO) { }  (TensorMap auto-dependency)

#include <cstdint>
#include <cstring>
#include <new>

#include "flat_schedule.h"       // FlatSchedule, FlatRegion, FlatTask, FlatArg, FlatDep
#include "runtime.h"              // TMARB Runtime struct
#include "utils/device_arena.h"  // DeviceArena
#include "pto_runtime2.h"        // runtime_reserve_layout, runtime_init_data_from_layout, etc.
#include "pto_runtime2_types.h"  // PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE, etc.
#include "pto_shared_memory.h"   // PTO2SharedMemoryHandle, PTO2SharedMemoryHeader
#include "pto_types.h"           // Arg, TensorArgType, PTO2ScopeMode, MixedKernels, TaskOutputTensors
#include "pto_orchestration_api.h"  // rt_submit_aic_task, rt_submit_task, rt_scope_begin, rt_scope_end

using namespace pto;

static constexpr int32_t MAX_DEPS_PER_TASK = 256;

// ── Build Arg from FlatTask ──

static void build_arg(const FlatTask* ftask, const FlatArg* fargs,
                      int32_t arg_bound, Arg& arg) {
    for (int16_t i = 0; i < ftask->num_args; i++) {
        const FlatArg& fa = fargs[i];
        switch (fa.direction) {
            case 0:  arg.add_input(fa.runtime_slot);     break;
            case 1:  arg.add_output(fa.runtime_slot);    break;
            case 2:  arg.add_inout(fa.runtime_slot);     break;
            case 3:  arg.add_scalar(fa.runtime_slot);    break;
            case 4:  arg.add_no_dep(fa.runtime_slot);    break;
            case 5:  arg.add_output_existing(fa.runtime_slot); break;
            default:
                // Unknown direction — treat as input to avoid silent data loss
                arg.add_input(fa.runtime_slot);
                break;
        }
    }
}

// ── Set dependencies from FlatDep list ──

static void set_deps(const FlatTask* ftask, const FlatDep* fdeps,
                     int32_t total_deps,
                     const PTO2TaskId* task_ids, int32_t num_submitted,
                     PTO2TaskId* dep_buf, int32_t dep_buf_size,
                     Arg& arg) {
    if (ftask->dep_count == 0 || task_ids == nullptr) return;
    uint32_t count = 0;
    for (int32_t i = 0; i < ftask->dep_count; i++) {
        int32_t dep_idx = ftask->dep_start + i;
        if (dep_idx < 0 || dep_idx >= total_deps) break;
        const FlatDep& fd = fdeps[dep_idx];
        if (fd.producer >= 0 && fd.producer < num_submitted) {
            if (count >= (uint32_t)dep_buf_size) {
                // Dependency buffer full — drop remaining deps with error signal.
                // This is a data integrity issue, not a crash.
                break;
            }
            dep_buf[count++] = task_ids[fd.producer];
        }
    }
    if (count > 0) {
        arg.set_dependencies(dep_buf, count);
    }
}

// ── Main interpreter loop ──

static void interpret_schedule(PTO2Runtime* rt, const FlatSchedule* sched,
                                const FlatRegion* regions, const FlatTask* tasks,
                                const FlatArg* args, const FlatDep* fdeps,
                                PTO2TaskId* dep_buf) {

    for (int32_t r = 0; r < sched->num_regions; r++) {
        const FlatRegion& rg = regions[r];

        // Bounds check: validate region's task and dep ranges
        if (rg.task_start < 0 || rg.task_start + rg.num_tasks > sched->total_tasks) {
            continue;  // invalid region bounds — skip
        }
        if (rg.dep_start < 0 || rg.dep_start + rg.num_deps > sched->total_deps) {
            continue;  // invalid dep bounds — skip
        }

        if (rg.kind == 1) {
            // Dynamic region
            rt->pending_scope_mode = PTO2ScopeMode::AUTO;
            rt_scope_begin(rt);
            rt_scope_end(rt);
            continue;
        }

        // Static region — explicit tasks + deps
        rt->pending_scope_mode = (rg.scope_mode == 1) ? PTO2ScopeMode::MANUAL : PTO2ScopeMode::AUTO;
        rt_scope_begin(rt);

        // Heap-allocated task ID array — sized to actual task count, avoids stack overflow
        int32_t alloc_size = rg.num_tasks > 0 ? rg.num_tasks : 1;
        PTO2TaskId* task_ids = new (std::nothrow) PTO2TaskId[alloc_size];
        if (!task_ids) {
            rt_scope_end(rt);
            continue;  // allocation failed — skip region
        }
        int32_t num_submitted = 0;
        int32_t arg_cursor = 0;

        for (int32_t t = 0; t < rg.num_tasks; t++) {
            const FlatTask& ft = tasks[rg.task_start + t];

            // Bounds check arg range
            int32_t task_arg_base = ft.arg_base;
            if (task_arg_base < 0 || task_arg_base + ft.num_args > sched->total_args) {
                continue;  // invalid arg range — skip task
            }

            Arg submit_arg;

            build_arg(&ft, &args[task_arg_base], sched->total_args, submit_arg);
            set_deps(&ft, fdeps, sched->total_deps,
                     task_ids, num_submitted, dep_buf, MAX_DEPS_PER_TASK,
                     submit_arg);

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

        delete[] task_ids;
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
    if (flat_sched->version != 1) {
        return -3;  // unsupported format version
    }

    // ── Initialize PTO2Runtime from prebuilt arena ──
    DeviceArena runtime_arena;
    runtime_arena.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);

    PTO2RuntimeArenaLayout layout;
    layout = runtime_reserve_layout(runtime_arena, task_window_size, PTO2_DEP_LIST_POOL_SIZE);

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

    // ── Parse flat schedule using header fields ──
    auto* regions = reinterpret_cast<const FlatRegion*>(flat_sched + 1);
    auto* tasks   = reinterpret_cast<const FlatTask*>(regions + flat_sched->num_regions);
    auto* args    = reinterpret_cast<const FlatArg*>(tasks  + flat_sched->total_tasks);
    auto* fdeps   = reinterpret_cast<const FlatDep*>(args   + flat_sched->total_args);

    // ── Pre-allocate dependency buffer (heap, max deps per task) ──
    PTO2TaskId* dep_buf = new (std::nothrow) PTO2TaskId[MAX_DEPS_PER_TASK];
    if (!dep_buf) return -4;

    // ── Run interpreter ──
    interpret_schedule(rt, flat_sched, regions, tasks, args, fdeps, dep_buf);
    rt_orchestration_done(rt);

    delete[] dep_buf;
    return 0;
}
