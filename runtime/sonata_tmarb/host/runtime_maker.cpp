// Sonata TMARB Interpreter — host-side runtime maker
//
// Thin wrapper over simpler's runtime framework. Stages kernel binaries
// (same as upstream TMARB) and replaces device-side orchestration with
// the schedule-driven interpreter.
//
// Exports the three symbols required by simpler's runtime framework:
//   prepare_callable_impl, bind_callable_to_runtime_impl, validate_runtime_impl
//
// The interpreter reads a pre-serialized flat_schedule binary from disk
// (written by Sonata's compile hook as sonata_schedule.bin) and feeds it
// to aicpu_entry() for device-side interpretation.

#include <sys/time.h>

#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "flat_schedule.h"
#include "sonata_hook.h"

// Upstream TMARB headers (resolved via build_config include_dirs + platform cmake)
#include "callable.h"
#include "prepare_callable_common.h"
#include "runtime.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "common/unified_log.h"
#include "utils/device_arena.h"

static int64_t _now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

// ── prepare_callable_impl ──
//
// Same as upstream TMARB: stage kernel binaries + orch SO into the callable.
// The orchestration SO binary is stored as the callable's binary_data(),
// which in the sonata_tmarb case is the flat_schedule binary embedded at
// compile time.

extern "C" int
prepare_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out) {
    if (callable == nullptr || upload_fn == nullptr || out == nullptr) {
        LOG_ERROR("prepare_callable_impl: null argument");
        return -1;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO_V0("Sonata prepare: registering %d kernel(s)", callable->child_count());
    if (upload_and_collect_child_addrs(callable, upload_fn, &out->kernel_addrs) != 0) {
        LOG_ERROR("Failed to upload ChipCallable buffer");
        return -1;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
            return -1;
        }
    }

    const uint8_t *orch_so = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();

    if (orch_so == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration binary is required (carries the flat_schedule)");
        return -1;
    }

    out->orch_so_data = orch_so;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();
    LOG_INFO_V0("Sonata prepare: orch binary staged (%zu bytes)", orch_so_size);
    return 0;
}

// ── bind_callable_to_runtime_impl ──
//
// Per-run binding. Sets up device memory (tensor H2D, GM heap, SM) and
// then invokes the schedule interpreter instead of the TMARB orchestrator.
//
// The flat_schedule binary is extracted from the callable's orch_so_data
// (staged by prepare_callable_impl) and passed directly to aicpu_entry.

// Forward declaration: device-side entry point
struct Tensor;
extern "C" int aicpu_entry(void* prebuilt_arena, uint64_t arena_size,
                           void* sm_ptr, uint64_t sm_size,
                           void* gm_heap, uint64_t heap_size,
                           int32_t aic_count, int32_t aiv_count,
                           int32_t task_window_size,
                           const FlatSchedule* flat_sched,
                           const Tensor* tensor_registry,
                           int32_t tensor_registry_size);

extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count, uint64_t ring_task_window, uint64_t ring_heap, uint64_t ring_dep_pool
) {
    if (runtime == nullptr) {
        LOG_ERROR("bind_callable_to_runtime_impl: runtime is null");
        return -1;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("bind_callable_to_runtime_impl: orch_args is null");
        return -1;
    }
    if (host_orch_func_ptr != nullptr) {
        LOG_ERROR("sonata_tmarb does not accept a host_orch_func_ptr");
        return -1;
    }

    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO_V0("Sonata bind: %d tensors + %d scalars", tensor_count, scalar_count);

    int64_t t_total_start = _now_ms();

    // ── Stage tensors to device (same as upstream) ──
    ChipStorageTaskArgs device_args;
    for (int i = 0; i < tensor_count; i++) {
        Tensor t = orch_args->tensor(i);
        if (t.is_child_memory()) {
            device_args.add_tensor(t);
            continue;
        }
        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());

        void *dev_ptr = runtime->host_api.device_malloc(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return -1;
        }
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        int rc;
        if (is_pure_output && runtime->host_api.device_memset != nullptr) {
            rc = runtime->host_api.device_memset(dev_ptr, 0, size);
        } else {
            rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
        }
        if (rc != 0) {
            LOG_ERROR("Failed to stage tensor %d to device", i);
            runtime->host_api.device_free(dev_ptr);
            return -1;
        }
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back});
        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }

    // ── Set up GM heap + SM (same as upstream) ──
    uint64_t eff_heap_size = ring_heap ? ring_heap : PTO2_HEAP_SIZE;
    uint64_t eff_task_window_size = ring_task_window ? ring_task_window : PTO2_TASK_WINDOW_SIZE;
    uint64_t total_heap_size = eff_heap_size * PTO2_MAX_RING_DEPTH;
    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(eff_task_window_size);
    int32_t eff_dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;

    DeviceArena host_arena;
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, eff_task_window_size, eff_dep_pool_capacity);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena");
        return -1;
    }

    if (runtime->host_api.setup_static_arena(total_heap_size, sm_size, layout.arena_size) != 0) {
        LOG_ERROR("Failed to setup static arena");
        return -1;
    }

    void *gm_heap = runtime->host_api.acquire_pooled_gm_heap();
    void *sm_ptr = runtime->host_api.acquire_pooled_gm_sm();
    void *runtime_arena_dev = runtime->host_api.acquire_pooled_runtime_arena();
    if (gm_heap == nullptr || sm_ptr == nullptr || runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled resources");
        return -1;
    }

    runtime->set_gm_heap(gm_heap);
    runtime->set_gm_sm_ptr(sm_ptr);
    runtime->set_orch_args(device_args);

    // ── Build prebuilt runtime arena image ──
    PTO2Runtime *rt = runtime_init_data_from_layout(
        host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size, gm_heap, eff_heap_size);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return -1;
    }
    runtime_wire_arena_pointers(host_arena, layout, rt);
    rt->prebuilt_layout = layout;

    int rc_upload = runtime->host_api.copy_to_device(runtime_arena_dev, host_arena.base(), layout.arena_size);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to upload prebuilt arena (rc=%d)", rc_upload);
        return -1;
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);

    // ── Extract flat_schedule from callable's orch binary ──
    // The orch_so_data contains the flat_schedule binary that was uploaded
    // by prepare_callable_impl. Copy it to a host-side buffer for validation.
    //
    // NOTE: The callable artifacts are passed via the Runtime's callable_
    // member (set by the framework before bind_callable is called).
    // For now, we rely on SONATA_SCHEDULE_PATH env var or a well-known
    // work_dir path as a fallback.
    //
    // TODO: Extract flat_schedule from callable_.orch_so_data when
    // the framework passes callable artifacts through the Runtime.
    const char *schedule_path_env = std::getenv("SONATA_SCHEDULE_PATH");
    if (schedule_path_env == nullptr) {
        LOG_ERROR("SONATA_SCHEDULE_PATH not set — cannot locate sonata_schedule.bin");
        return -1;
    }

    FILE *f = std::fopen(schedule_path_env, "rb");
    if (f == nullptr) {
        LOG_ERROR("Cannot open schedule file: %s", schedule_path_env);
        return -1;
    }
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (file_size < static_cast<long>(sizeof(FlatSchedule))) {
        LOG_ERROR("Schedule file too small: %ld bytes", file_size);
        std::fclose(f);
        return -1;
    }

    auto *sched_buf = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(file_size)));
    if (sched_buf == nullptr) {
        LOG_ERROR("Failed to allocate schedule buffer");
        std::fclose(f);
        return -1;
    }
    size_t read = std::fread(sched_buf, 1, static_cast<size_t>(file_size), f);
    std::fclose(f);
    if (read != static_cast<size_t>(file_size)) {
        LOG_ERROR("Short read: %zu of %ld bytes", read, file_size);
        std::free(sched_buf);
        return -1;
    }

    auto *flat_sched = reinterpret_cast<FlatSchedule*>(sched_buf);
    if (flat_sched->magic != 0x534F4E41) {
        LOG_ERROR("Bad schedule magic: 0x%08x", flat_sched->magic);
        std::free(sched_buf);
        return -1;
    }

    // ── Invoke the interpreter (replaces TMARB orchestrator) ──
    int interp_rc = aicpu_entry(
        runtime_arena_dev, layout.arena_size,
        sm_ptr, sm_size,
        gm_heap, eff_heap_size,
        0, 0,  // aic_count, aiv_count (unused by interpreter)
        static_cast<int32_t>(eff_task_window_size),
        flat_sched,
        nullptr,  // tensor_registry — TODO: build from device_args
        0         // tensor_registry_size
    );

    std::free(sched_buf);

    int64_t t_total_end = _now_ms();
    LOG_INFO_V0("Sonata bind total: %" PRId64 "ms (interp rc=%d)", t_total_end - t_total_start, interp_rc);

    if (interp_rc != 0) {
        LOG_ERROR("Interpreter failed with rc=%d", interp_rc);
        return -1;
    }

    return 0;
}

// ── validate_runtime_impl ──
//
// Copy output tensors back from device to host and free device allocations.
// Same as upstream TMARB.

extern "C" int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("validate_runtime_impl: runtime is null");
        return -1;
    }

    for (auto &tp : runtime->tensor_pairs_) {
        if (tp.needs_copy_back) {
            int rc = runtime->host_api.copy_from_device(tp.host_ptr, tp.dev_ptr, tp.size);
            if (rc != 0) {
                LOG_WARN("D2H copy failed for tensor at %p", tp.dev_ptr);
            }
        }
        runtime->host_api.device_free(tp.dev_ptr);
    }
    runtime->tensor_pairs_.clear();

    return 0;
}
