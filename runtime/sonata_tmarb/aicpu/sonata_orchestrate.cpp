// sonata_orchestrate.cpp — FlatSchedule bridge for NPU dual-path execution.
//
// This file provides sonata_orchestrate_with_schedule(), the entry point
// that the upstream AICPU orchestrator thread calls when a Sonata schedule
// binary is available (detected via Runtime::sonata_sched_addr_).
//
// It reuses interpret_schedule() from the standalone interpreter to dispatch
// tasks according to the FlatSchedule, using the PTO2Runtime and tensor
// registry already set up by the TMARB runtime.
//
// See ADR-002 for the full architecture.

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

// ── Forward declarations of interpreter-internal functions ──
//
// These are shared with aicpu_executor_standalone.cpp.  Both files define
// the same static helpers after the rename (B1); the linker keeps one copy
// since neither is exported.

static void build_arg(const FlatTask* ftask, const FlatArg* fargs,
                      const Tensor* tensor_registry, int32_t registry_size,
                      Arg& arg) {
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
                     Arg& arg) {
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

    static constexpr int32_t MAX_DEPS_PER_TASK = 256;

    for (int32_t r = 0; r < sched->num_regions; r++) {
        const FlatRegion& rg = regions[r];

        // Bounds checks
        if (rg.task_start < 0 || rg.num_tasks > sched->total_tasks - rg.task_start) continue;
        if (rg.dep_start < 0 || rg.num_deps > sched->total_deps - rg.dep_start) continue;

        // Single is_fatal check before any scope or task operations.
        if (rt->ops->is_fatal(rt)) return;

        if (rg.kind == 1) {
            // Dynamic region — AUTO scope (TMARB runtime discovers tasks)
            rt->pending_scope_mode = PTO2ScopeMode::AUTO;
            rt->ops->scope_begin(rt);
            rt->ops->scope_end(rt);
            LOG_WARN("sonata: dynamic region %d via AUTO scope", r);
            continue;
        }

        // Static region — explicit tasks + deps
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

            Arg submit_arg;
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

            if (rt->ops->is_fatal(rt)) break;
            auto result = rt->ops->submit_task(rt, mk, submit_arg);
            task_ids[num_submitted++] = result.task_id();
        }

        delete[] task_ids;
        if (!rt->ops->is_fatal(rt)) rt->ops->scope_end(rt);
    }
}

// ── Public entry point (called by upstream orchestrator thread) ──

extern "C" void sonata_orchestrate_with_schedule(
    PTO2Runtime* rt,
    Runtime* runtime,
    uint64_t sched_addr,
    uint64_t sched_size
) {
    if (rt == nullptr || runtime == nullptr || sched_addr == 0 || sched_size == 0) {
        LOG_ERROR("sonata_orchestrate_with_schedule: invalid args");
        return;
    }

    // Reinterpret the device-address as pointer (works on both sim and NPU:
    // sim → host virtual address; NPU → AICPU-accessible HBM address).
    auto* raw = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(sched_addr));
    if (raw == nullptr) {
        LOG_ERROR("sonata_orchestrate_with_schedule: null raw pointer");
        return;
    }
    if (sched_size < sizeof(FlatSchedule)) {
        LOG_ERROR("sonata_orchestrate_with_schedule: blob too small (%zu)", sched_size);
        return;
    }

    auto* sched = reinterpret_cast<const FlatSchedule*>(raw);
    if (sched->magic != FLAT_SCHEDULE_MAGIC) {
        LOG_ERROR("sonata_orchestrate_with_schedule: bad magic 0x%08x", sched->magic);
        return;
    }

    // Parse FlatSchedule arrays from the raw binary.
    int32_t payload_skip = (sched->version >= 2) ? 4 : 0;
    auto* regions = reinterpret_cast<const FlatRegion*>(raw + sizeof(FlatSchedule) + payload_skip);
    auto* tasks   = reinterpret_cast<const FlatTask*>(regions + sched->num_regions);
    auto* args    = reinterpret_cast<const FlatArg*>(tasks  + sched->total_tasks);
    auto* fdeps   = reinterpret_cast<const FlatDep*>(args   + sched->total_args);

    // Extract tensor registry from the Runtime's ChipStorageTaskArgs.
    const auto& orch_args = runtime->get_orch_args();
    int32_t registry_size = orch_args.tensor_count();
    const Tensor* tensor_registry = (registry_size > 0) ? orch_args.tensor_data() : nullptr;

    // Pre-allocate dependency buffer.
    static constexpr int32_t MAX_DEPS_PER_TASK = 256;
    auto* dep_buf = new (std::nothrow) PTO2TaskId[MAX_DEPS_PER_TASK];
    if (!dep_buf) {
        LOG_ERROR("sonata_orchestrate_with_schedule: dep_buf alloc failed");
        return;
    }

    // Run the schedule interpreter using the pre-existing PTO2Runtime.
    interpret_schedule(rt, sched, regions, tasks, args, fdeps, dep_buf,
                       tensor_registry, registry_size);

    delete[] dep_buf;

    LOG_INFO_V0("sonata_orchestrate_with_schedule: done (%d regions, %d tasks)",
                sched->num_regions, sched->total_tasks);
}
