// sonata_orchestrate.cpp — AICPU entry point for Sonata schedule branch.
//
// Called unconditionally by the patched aicpu_executor.cpp orchestrator
// thread.  Reads the schedule address from PTO2Runtime::total_cycles
// (stashed by host before copy_to_device) and executes it.  Falls back
// to TMARB when no schedule is available.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#include "flat_schedule.h"
#include "runtime.h"
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_types.h"
#include "tensor.h"

extern "C" PTO2Runtime *framework_current_runtime(void);

// ── Static helpers ──

static void build_arg(const FlatTask* ftask, const FlatArg* fargs,
                      const Tensor* tensor_registry, int32_t registry_size,
                      Arg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>& arg) {
    for (int16_t i = 0; i < ftask->num_args; i++) {
        const FlatArg& fa = fargs[i];
        int32_t slot = fa.runtime_slot;
        bool slot_valid = (tensor_registry && slot >= 0 && slot < registry_size);
        if (!slot_valid && fa.direction != 3) {
            LOG_WARN("sonata: task=%d arg=%d dir=%d slot=%d out of range [0,%d)",
                     ftask->func_id, i, fa.direction, slot, registry_size);
        }
        switch (fa.direction) {
            case 0:  // input
                if (slot_valid) { arg.add_input(tensor_registry[slot]); }
                break;
            case 1:  // output
            case 5:  // outputexisting
                if (slot_valid) { arg.add_output(tensor_registry[slot]); }
                break;
            case 2:  // inout
                if (slot_valid) { arg.add_inout(tensor_registry[slot]); }
                break;
            case 3:  // scalar
                arg.add_scalar(slot);
                break;
            case 4:  // nodep
                if (slot_valid) { arg.add_no_dep(tensor_registry[slot]); }
                break;
            default:
                break;
        }
    }
}

static void set_deps(int32_t task_index_in_region,
                     const FlatDep* fdeps, int32_t total_deps,
                     int32_t dep_start, int32_t num_deps,
                     const PTO2TaskId* task_ids, int32_t num_submitted,
                     PTO2TaskId* dep_buf, int32_t dep_buf_size,
                     Arg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>& arg) {
    if (num_deps == 0 || task_ids == nullptr) return;
    uint32_t count = 0;
    for (int32_t i = 0; i < num_deps; i++) {
        int32_t dep_idx = dep_start + i;
        if (dep_idx < 0 || dep_idx >= total_deps) break;
        const FlatDep& fd = fdeps[dep_idx];
        if (fd.consumer != task_index_in_region) continue;
        if (fd.producer >= 0 && fd.producer < num_submitted) {
            if (count >= (uint32_t)dep_buf_size) break;
            dep_buf[count++] = task_ids[fd.producer];
        }
    }
    if (count > 0) {
        arg.set_dependencies(dep_buf, count);
    }
}

static void interpret_schedule(PTO2Runtime* rt, const FlatSchedule* sched,
                               const FlatRegion* regions,
                               const FlatTask* tasks,
                               const FlatArg* args, const FlatDep* fdeps,
                               PTO2TaskId* dep_buf,
                               const Tensor* tensor_registry,
                               int32_t registry_size) {
    using SubmitArg = Arg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>;

    for (int32_t r = 0; r < sched->num_regions; r++) {
        const FlatRegion& rg = regions[r];
        if (rg.task_start < 0 || rg.num_tasks > sched->total_tasks - rg.task_start) continue;
        if (rg.dep_start < 0 || rg.num_deps > sched->total_deps - rg.dep_start) continue;
        if (rt->ops->is_fatal(rt)) return;

        if (rg.kind == 1) {
            rt->pending_scope_mode = PTO2ScopeMode::AUTO;
            rt->ops->scope_begin(rt);
            rt->ops->scope_end(rt);
            LOG_WARN("sonata: dynamic region %d via AUTO scope", r);
            continue;
        }

        rt->pending_scope_mode = (rg.scope_mode == 1) ? PTO2ScopeMode::MANUAL : PTO2ScopeMode::AUTO;
        rt->ops->scope_begin(rt);

        int32_t alloc_size = rg.num_tasks > 0 ? rg.num_tasks : 1;
        auto* task_ids = new (std::nothrow) PTO2TaskId[alloc_size];
        if (!task_ids) { rt->ops->scope_end(rt); continue; }
        int32_t num_submitted = 0;

        for (int32_t t = 0; t < rg.num_tasks; t++) {
            const FlatTask& ft = tasks[rg.task_start + t];
            int32_t task_arg_base = ft.arg_base;
            if (task_arg_base < 0 || ft.num_args > sched->total_args - task_arg_base) continue;

            SubmitArg submit_arg;
            build_arg(&ft, &args[task_arg_base],
                      tensor_registry, registry_size, submit_arg);
            set_deps(t, fdeps, sched->total_deps,
                     rg.dep_start, rg.num_deps,
                     task_ids, num_submitted, dep_buf, MAX_DEPS_PER_TASK,
                     submit_arg);

            MixedKernels mk;
            if (ft.core_type == 1) {
                mk.aiv0_kernel_id = ft.func_id;
                mk.aic_kernel_id = INVALID_KERNEL_ID;
            } else if (ft.core_type == 2) {
                mk.aic_kernel_id = ft.func_id;
                mk.aiv0_kernel_id = INVALID_KERNEL_ID;
            } else {
                mk.aic_kernel_id = ft.func_id;
                mk.aiv0_kernel_id = INVALID_KERNEL_ID;
            }
            mk.aiv1_kernel_id = INVALID_KERNEL_ID;

            if (rt->ops->is_fatal(rt)) { rt->ops->scope_end(rt); break; }
            auto result = rt->ops->submit_task(rt, mk, submit_arg);
            task_ids[num_submitted++] = result.task_id();
        }
        delete[] task_ids;
        rt->ops->scope_end(rt);
    }
}

// ── Public entry point ──
// Returns true if a schedule was executed, false to fall back to TMARB.

extern "C" bool sonata_orchestrate_with_schedule(
    PTO2Runtime* rt,
    Runtime* runtime
) {
    if (rt == nullptr || runtime == nullptr) {
        return false;
    }

    // Read sonata data offset from total_cycles field.
    // On the NPU path, total_cycles contains the offset from rt (PTO2Runtime)
    // to the sonata data block within the same prebuilt arena.
    // The prebuilt arena is fully AICPU-accessible after attach().
    // Layout at (uint8_t*)rt + total_cycles:
    //   offset 0:                   int32_t tensor_count
    //   offset 4:                   Tensor[tensor_count]
    //   offset 4 + tensor_data:     FlatSchedule
    int64_t sonata_offset = rt->total_cycles;
    if (sonata_offset == 0) {
        return false;
    }

    const uint8_t* base = reinterpret_cast<const uint8_t*>(rt) + sonata_offset;

    // Tensor registry (at offset 0: count, offset 4: Tensor array)
    int32_t registry_size = 0;
    const Tensor* tensor_registry = nullptr;
    std::memcpy(&registry_size, base, sizeof(registry_size));
    if (registry_size > 0 && registry_size <= MAX_TENSOR_ARGS) {
        tensor_registry = reinterpret_cast<const Tensor*>(base + sizeof(int32_t));
    }

    // FlatSchedule (after registry: count + tensors)
    auto* sched = reinterpret_cast<const FlatSchedule*>(
        base + sizeof(int32_t) + static_cast<size_t>(registry_size) * sizeof(Tensor));
    if (sched->magic != FLAT_SCHEDULE_MAGIC) {
        LOG_ERROR("sonata: bad schedule magic 0x%08x", sched->magic);
        return false;
    }

    // FlatSchedule array parsing (relative to sched pointer)
    int32_t payload_skip = (sched->version >= 2) ? 4 : 0;
    const uint8_t* sched_raw = reinterpret_cast<const uint8_t*>(sched);
    auto* regions = reinterpret_cast<const FlatRegion*>(sched_raw + sizeof(FlatSchedule) + payload_skip);
    auto* tasks   = reinterpret_cast<const FlatTask*>(regions + sched->num_regions);
    auto* args    = reinterpret_cast<const FlatArg*>(tasks  + sched->total_tasks);
    auto* fdeps   = reinterpret_cast<const FlatDep*>(args   + sched->total_args);

    auto* dep_buf = new (std::nothrow) PTO2TaskId[MAX_DEPS_PER_TASK];
    if (!dep_buf) { return false; }

    interpret_schedule(rt, sched, regions, tasks, args, fdeps, dep_buf,
                       tensor_registry, registry_size);

    delete[] dep_buf;

    LOG_INFO_V0("sonata: done (%d regions, %d tasks)",
                sched->num_regions, sched->total_tasks);
    return true;
}
