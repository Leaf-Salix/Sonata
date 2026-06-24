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
#include "tensor.h"              // Tensor, TensorCreateInfo
#include "pto_types.h"           // Arg, TensorArgType, PTO2ScopeMode, MixedKernels, TaskOutputTensors
extern "C" PTO2Runtime *framework_current_runtime(void);
extern "C" void framework_bind_runtime(PTO2Runtime *rt);

static inline PTO2Runtime *sonata_current_runtime() { return framework_current_runtime(); }

static inline TaskOutputTensors sonata_rt_submit_task(const MixedKernels &mk, const Arg &args) {
    PTO2Runtime *rt = sonata_current_runtime();
    if (rt->ops->is_fatal(rt)) return TaskOutputTensors{};
    return rt->ops->submit_task(rt, mk, args);
}

static inline TaskOutputTensors sonata_rt_submit_aic_task(int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    mk.aiv0_kernel_id = INVALID_KERNEL_ID;
    mk.aiv1_kernel_id = INVALID_KERNEL_ID;
    return sonata_rt_submit_task(mk, args);
}

static inline TaskOutputTensors sonata_rt_submit_aiv_task(int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aic_kernel_id = INVALID_KERNEL_ID;
    mk.aiv0_kernel_id = kernel_id;
    mk.aiv1_kernel_id = INVALID_KERNEL_ID;
    return sonata_rt_submit_task(mk, args);
}

static inline void sonata_rt_scope_begin(PTO2Runtime *rt) {
    if (rt->ops->is_fatal(rt)) return;
    rt->ops->scope_begin(rt);
}

static inline void sonata_rt_scope_end(PTO2Runtime *rt) {
    if (rt->ops->is_fatal(rt)) return;
    rt->ops->scope_end(rt);
}

static constexpr int32_t MAX_DEPS_PER_TASK = 256;

// ── Build Arg from FlatTask ──
//
// Looks up Tensor objects from the registry by runtime_slot index.
// The caller (host-side framework) populates the registry from orch_args
// before invoking aicpu_entry. If registry is nullptr, tensor-dependent
// directions are skipped (scalar-only tasks still work).

static void build_arg(const FlatTask* ftask, const FlatArg* fargs,
                      const Tensor* tensor_registry, int32_t registry_size,
                      Arg& arg) {
    for (int16_t i = 0; i < ftask->num_args; i++) {
        const FlatArg& fa = fargs[i];
        int32_t slot = fa.runtime_slot;
        switch (fa.direction) {
            case 0:  // input
                if (tensor_registry && slot >= 0 && slot < registry_size) {
                    arg.add_input(tensor_registry[slot]);
                }
                break;
            case 1:  // output
            case 5:  // outputexisting — same runtime behavior as output
                if (tensor_registry && slot >= 0 && slot < registry_size) {
                    arg.add_output(tensor_registry[slot]);
                }
                break;
            case 2:  // inout
                if (tensor_registry && slot >= 0 && slot < registry_size) {
                    arg.add_inout(tensor_registry[slot]);
                }
                break;
            case 3:  // scalar — slot value stored directly as scalar arg
                arg.add_scalar(slot);
                break;
            case 4:  // nodep
                if (tensor_registry && slot >= 0 && slot < registry_size) {
                    arg.add_no_dep(tensor_registry[slot]);
                }
                break;
            default:
                // Unknown direction — skip to avoid silent misbinding
                break;
        }
    }
}

// ── Set dependencies from FlatDep list ──

static void set_deps(int32_t task_index_in_region,
                     const FlatDep* fdeps, int32_t total_deps,
                     int32_t dep_start, int32_t num_deps,
                     const PTO2TaskId* task_ids, int32_t num_submitted,
                     PTO2TaskId* dep_buf, int32_t dep_buf_size,
                     Arg& arg) {
    if (num_deps == 0 || task_ids == nullptr) return;
    uint32_t count = 0;
    for (int32_t i = 0; i < num_deps; i++) {
        int32_t dep_idx = dep_start + i;
        if (dep_idx < 0 || dep_idx >= total_deps) break;
        const FlatDep& fd = fdeps[dep_idx];
        if (fd.consumer != task_index_in_region) continue;
        if (fd.producer >= 0 && fd.producer < num_submitted) {
            if (count >= (uint32_t)dep_buf_size) {
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
                                PTO2TaskId* dep_buf,
                                const Tensor* tensor_registry, int32_t registry_size) {

    for (int32_t r = 0; r < sched->num_regions; r++) {
        const FlatRegion& rg = regions[r];

        // Bounds check: validate region's task and dep ranges (overflow-safe)
        if (rg.task_start < 0 || rg.num_tasks > sched->total_tasks - rg.task_start) {
            continue;  // invalid region bounds — skip
        }
        if (rg.dep_start < 0 || rg.num_deps > sched->total_deps - rg.dep_start) {
            continue;  // invalid dep bounds — skip
        }

        if (rg.kind == 1) {
            // Dynamic region
            rt->pending_scope_mode = PTO2ScopeMode::AUTO;
            sonata_rt_scope_begin(rt);
            sonata_rt_scope_end(rt);
            continue;
        }

        // Static region — explicit tasks + deps
        rt->pending_scope_mode = (rg.scope_mode == 1) ? PTO2ScopeMode::MANUAL : PTO2ScopeMode::AUTO;
        sonata_rt_scope_begin(rt);

        // Heap-allocated task ID array — sized to actual task count, avoids stack overflow
        int32_t alloc_size = rg.num_tasks > 0 ? rg.num_tasks : 1;
        PTO2TaskId* task_ids = new (std::nothrow) PTO2TaskId[alloc_size];
        if (!task_ids) {
            sonata_rt_scope_end(rt);
            continue;  // allocation failed — skip region
        }
        int32_t num_submitted = 0;

        for (int32_t t = 0; t < rg.num_tasks; t++) {
            const FlatTask& ft = tasks[rg.task_start + t];

            // Bounds check arg range (overflow-safe)
            int32_t task_arg_base = ft.arg_base;
            if (task_arg_base < 0 || ft.num_args > sched->total_args - task_arg_base) {
                continue;  // invalid arg range — skip task
            }

            Arg submit_arg;

            build_arg(&ft, &args[task_arg_base],
                      tensor_registry, registry_size, submit_arg);
            set_deps(t, fdeps, sched->total_deps,
                     rg.dep_start, rg.num_deps,
                     task_ids, num_submitted, dep_buf, MAX_DEPS_PER_TASK,
                     submit_arg);

            TaskOutputTensors result;
            if (ft.core_type == 1) {
                result = sonata_rt_submit_aiv_task(ft.func_id, submit_arg);
            } else if (ft.core_type == 2) {
                MixedKernels mk;
                mk.aic_kernel_id = ft.func_id;
                mk.aiv0_kernel_id = INVALID_KERNEL_ID;
                mk.aiv1_kernel_id = INVALID_KERNEL_ID;
                result = sonata_rt_submit_task(mk, submit_arg);
            } else {
                result = sonata_rt_submit_aic_task(ft.func_id, submit_arg);
            }

            task_ids[num_submitted++] = result.task_id();
        }

        delete[] task_ids;
        sonata_rt_scope_end(rt);
    }
}

// ── Device-side entry point ──
//
// Called by simpler runtime after host-side bind. Initializes the
// PTO2Runtime from the prebuilt arena, parses the flat schedule,
// and runs the interpreter loop.
//
// tensor_registry: array of Tensor objects populated from orch_args by
// the host-side framework. Indexed by FlatArg.runtime_slot. May be
// nullptr if the host-side caller has not yet been updated to provide
// tensor bindings (scalar-only tasks still work).

extern "C" int aicpu_entry(void* prebuilt_arena, uint64_t /*arena_size*/,
                           void* sm_ptr, uint64_t sm_size,
                           void* gm_heap, uint64_t heap_size,
                           int32_t aic_count, int32_t aiv_count,
                           int32_t task_window_size,
                           const FlatSchedule* flat_sched,
                           const Tensor* tensor_registry,
                           int32_t tensor_registry_size) {

    if (flat_sched == nullptr || flat_sched->magic != 0x534F4E41) {
        return -1;
    }
    if (flat_sched->version != 1 && flat_sched->version != BINARY_FORMAT_VERSION) {
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
    //
    // v1: header (88 bytes) → directly to arrays
    // v2: header (88 bytes) + CRC-32 (4 bytes) → skip CRC before arrays
    const auto* raw = reinterpret_cast<const uint8_t*>(flat_sched);
    int32_t payload_skip = (flat_sched->version >= 2) ? 4 : 0;
    auto* regions = reinterpret_cast<const FlatRegion*>(raw + sizeof(FlatSchedule) + payload_skip);
    auto* tasks   = reinterpret_cast<const FlatTask*>(regions + flat_sched->num_regions);
    auto* args    = reinterpret_cast<const FlatArg*>(tasks  + flat_sched->total_tasks);
    auto* fdeps   = reinterpret_cast<const FlatDep*>(args   + flat_sched->total_args);

    // ── Pre-allocate dependency buffer (heap, max deps per task) ──
    PTO2TaskId* dep_buf = new (std::nothrow) PTO2TaskId[MAX_DEPS_PER_TASK];
    if (!dep_buf) return -4;

    // ── Run interpreter ──
    interpret_schedule(rt, flat_sched, regions, tasks, args, fdeps, dep_buf,
                       tensor_registry, tensor_registry_size);
    rt_orchestration_done(rt);

    delete[] dep_buf;
    return 0;
}
