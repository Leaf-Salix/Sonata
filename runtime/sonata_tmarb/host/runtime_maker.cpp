// Sonata TMARB Interpreter — host-side runtime maker
//
// Thin wrapper over simpler's runtime framework. Stages kernel binaries
// (same as upstream TMARB) and replaces device-side orchestration with
// the schedule-driven interpreter.
//
// Exports the three symbols required by simpler's runtime framework:
//   prepare_callable_impl, bind_callable_to_runtime_impl, validate_runtime_impl
//
// The interpreter reads a pre-serialized flat_schedule binary embedded in
// the ChipCallable's binary_data() by the compile-time hook (sonata_hook).
// prepare_callable_impl stashes a host copy; bind_callable_to_runtime_impl
// reads it out and passes it to aicpu_execute.
//
// Fallback: SONATA_SCHEDULE_PATH env var (v0.28 compatibility path).

#include <sys/time.h>

#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits.h>

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

// Sanity cap: 64 MiB max for a schedule binary (prevents OOM).
static constexpr size_t MAX_SCHEDULE_SIZE = 64UL * 1024UL * 1024UL;

// ── Host-side schedule buffer ──
//
// prepare_callable_impl stashes the schedule binary from the callable's
// binary_data() into this static buffer.  bind_callable_to_runtime_impl
// reads it back, avoiding a second file-system read (v0.28 env-var path)
// or a deep dive into the framework's callable-artifact lifecycle.
//
// Thread-safety: the simpler framework serialises prepare + bind per
// Worker, so no locking is needed.  Multiple Workers each go through
// their own prepare→bind→validate sequence, but since prepare overwrites
// the buffer, only one active Worker is expected at a time (L2 Worker).
//
// The env-var fallback (SONATA_SCHEDULE_PATH) is preserved for debugging
// and cross-language binary validation.

static uint8_t *g_schedule_buf = nullptr;
static size_t   g_schedule_size = 0;

static void _clear_schedule_buf() {
    std::free(g_schedule_buf);
    g_schedule_buf = nullptr;
    g_schedule_size = 0;
}

static bool _set_schedule_buf(const uint8_t *data, size_t size) {
    _clear_schedule_buf();
    if (data == nullptr || size == 0) return false;
    if (size > MAX_SCHEDULE_SIZE) return false;
    g_schedule_buf = static_cast<uint8_t *>(std::malloc(size));
    if (g_schedule_buf == nullptr) return false;
    std::memcpy(g_schedule_buf, data, size);
    g_schedule_size = size;
    return true;
}

// ── Find stashed schedule ──
//
// First tries the host-side static buffer (stashed by prepare_callable_impl).
// Falls back to SONATA_SCHEDULE_PATH env var (v0.28 compatibility path).
// The returned pointer is owned by the schedule-buffer; caller must NOT free it.

static const FlatSchedule *_find_stashed_schedule(size_t *out_size) {
    *out_size = 0;

    if (g_schedule_buf != nullptr && g_schedule_size >= sizeof(FlatSchedule)) {
        auto *fs = reinterpret_cast<const FlatSchedule *>(g_schedule_buf);
        if (fs->magic == FLAT_SCHEDULE_MAGIC) {
            *out_size = g_schedule_size;
            LOG_INFO_V0("Sonata: using stashed schedule (%zu bytes)", g_schedule_size);
            return fs;
        }
        LOG_WARN("Sonata: stashed schedule bad magic 0x%08x, clearing", fs->magic);
        _clear_schedule_buf();
    }

    // env var fallback
    const char *path = std::getenv("SONATA_SCHEDULE_PATH");
    if (path == nullptr) {
        LOG_ERROR("SONATA_SCHEDULE_PATH not set and no stashed schedule");
        return nullptr;
    }

    char resolved[PATH_MAX];
    if (realpath(path, resolved) == nullptr) {
        LOG_ERROR("Cannot resolve schedule path: %s", path);
        return nullptr;
    }

    FILE *f = std::fopen(resolved, "rb");
    if (f == nullptr) {
        LOG_ERROR("Cannot open schedule: %s", resolved);
        return nullptr;
    }
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (file_size < static_cast<long>(sizeof(FlatSchedule))) {
        LOG_ERROR("Schedule file too small: %ld bytes (need %zu)", file_size, sizeof(FlatSchedule));
        std::fclose(f);
        return nullptr;
    }
    if (file_size > static_cast<long>(MAX_SCHEDULE_SIZE)) {
        LOG_ERROR("Schedule file too large: %ld bytes", file_size);
        std::fclose(f);
        return nullptr;
    }

    _clear_schedule_buf();
    auto *buf = static_cast<uint8_t *>(std::malloc(static_cast<size_t>(file_size)));
    if (buf == nullptr) { std::fclose(f); return nullptr; }
    if (std::fread(buf, 1, static_cast<size_t>(file_size), f) != static_cast<size_t>(file_size)) {
        LOG_ERROR("Short read from schedule file");
        std::free(buf); std::fclose(f);
        return nullptr;
    }
    std::fclose(f);
    g_schedule_buf = buf;
    g_schedule_size = static_cast<size_t>(file_size);

    auto *fs = reinterpret_cast<const FlatSchedule *>(g_schedule_buf);
    if (fs->magic != FLAT_SCHEDULE_MAGIC) {
        LOG_ERROR("Bad schedule magic in env-var file: 0x%08x", fs->magic);
        _clear_schedule_buf();
        return nullptr;
    }
    LOG_INFO_V0("Sonata: loaded schedule from env var (%s, %zu bytes)", resolved, g_schedule_size);
    return fs;
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
        _clear_schedule_buf();
        return -1;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
            _clear_schedule_buf();
            return -1;
        }
    }

    const uint8_t *orch_so = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();

    if (orch_so == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration binary is required (carries the flat_schedule)");
        _clear_schedule_buf();
        return -1;
    }

    out->orch_so_data = orch_so;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();

    // Stash a host-side copy of the schedule binary.
    if (!_set_schedule_buf(orch_so, orch_so_size)) {
        LOG_WARN("Sonata prepare: failed to stash schedule buffer (%zu bytes)", orch_so_size);
    }

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
//
// aicpu_entry 通过 dlsym 从 aicpu_kernel.so 动态解析，避免 host_runtime.so
// 链接 TMARB 运行时函数（scheduler / platform_regs 等）。aicpu_kernel.so
// 已经包含完整的 TMARB 运行时 + 平台代码，且由 ChipWorker 在 host_runtime.so
// 之前加载。dlsym(RTLD_DEFAULT, "aicpu_entry") 可在当前进程空间中找到它。
//
// 这仅在模拟环境（a2a3sim）中有效，在该环境所有 .so 运行在同一进程空间。
// Onboard / 真机环境需要不同机制（跨芯片调用）。

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
    // Track whether bind has succeeded. If not, TensorCleanupGuard frees
    // any tensor device allocations on error exit.
    bool bind_succeeded = false;
    struct TensorCleanupGuard {
        Runtime *r;
        bool *success;
        ~TensorCleanupGuard() noexcept {
            if (!r || *success) return;
            for (auto &tp : r->tensor_pairs_) {
                r->host_api.device_free(tp.dev_ptr);
            }
            r->tensor_pairs_.clear();
        }
    } tensor_cleanup{runtime, &bind_succeeded};
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

    // ── Store runtime objects for the aicpu side ──
    // GM heap, SM, and orch_args are needed by aicpu_execute's runtime init.
    // The prebuilt arena offset is stored for consistency (aicpu doesn't
    // read it — it does its own runtime_init_data_from_layout — but the
    // upstream TMARB path expects the correct offset here).
    runtime->set_gm_heap(gm_heap);
    runtime->set_gm_sm_ptr(sm_ptr);
    runtime->set_orch_args(device_args);
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);

    // ── Extract flat_schedule from callable's orch binary ──
    // The orch_so_data contains the flat_schedule binary that was uploaded
    // by prepare_callable_impl and stashed via _set_schedule_buf.
    // Fall back to SONATA_SCHEDULE_PATH env var when stash is missing.

    size_t flat_sched_size = 0;
    const FlatSchedule *flat_sched = _find_stashed_schedule(&flat_sched_size);

    // ── NPU path: upload schedule to device memory ──
    // Check BEFORE the schedule-null return and validation so TMARB can run
    // as a fallback when no schedule exists.  When schedule IS available under
    // NPU mode, upload it to device memory and set Runtime fields for the
    // AICPU orchestrator to consume.
    const char *rt_mode = std::getenv("SONATA_RUNTIME_MODE");
    if (rt_mode != nullptr && strcmp(rt_mode, "npu") == 0) {
        if (flat_sched != nullptr && flat_sched_size >= sizeof(FlatSchedule)) {
            LOG_INFO_V0("Sonata: NPU — uploading schedule (%zu bytes)", flat_sched_size);
            void *sched_dev = runtime->host_api.device_malloc(flat_sched_size);
            if (sched_dev != nullptr) {
                int rc = runtime->host_api.copy_to_device(sched_dev, flat_sched, flat_sched_size);
                if (rc == 0) {
                    runtime->set_sonata_schedule(reinterpret_cast<uint64_t>(sched_dev), flat_sched_size);
                    LOG_INFO_V0("Sonata: schedule uploaded to 0x%llx",
                                (unsigned long long)(uintptr_t)sched_dev);
                } else {
                    runtime->host_api.device_free(sched_dev);
                    LOG_WARN("Sonata: H2D copy failed (rc=%d), TMARB fallback", rc);
                }
            } else {
                LOG_WARN("Sonata: device_malloc failed, TMARB fallback");
            }
        } else {
            LOG_INFO_V0("Sonata: NPU mode, no schedule — TMARB orchestrator");
        }
        return 0;
    }

    // ── Sim path: validate schedule + dlsym ──
    // Only reachable when SONATA_RUNTIME_MODE is not set or is "sim".
    if (flat_sched == nullptr) {
        LOG_ERROR("No schedule binary available");
        return -1;
    }

    // Validate header fields for the sim path (version, overflow-safe bounds).
    if (flat_sched->version != 1 && flat_sched->version != BINARY_FORMAT_VERSION) {
        LOG_ERROR("Unsupported schedule version: %d", flat_sched->version);
        return -1;
    }
    if (flat_sched->num_regions < 0 || flat_sched->total_tasks < 0 ||
        flat_sched->total_args < 0 || flat_sched->total_deps < 0) {
        LOG_ERROR("Schedule has negative field counts");
        return -1;
    }
    size_t expected_size = sizeof(FlatSchedule);
    expected_size += static_cast<size_t>(flat_sched->num_regions) * sizeof(FlatRegion);
    expected_size += static_cast<size_t>(flat_sched->total_tasks) * sizeof(FlatTask);
    expected_size += static_cast<size_t>(flat_sched->total_args) * sizeof(FlatArg);
    expected_size += static_cast<size_t>(flat_sched->total_deps) * sizeof(FlatDep);
    if (flat_sched->version >= 2) expected_size += 4;
    if (expected_size > flat_sched_size || expected_size < sizeof(FlatSchedule)) {
        LOG_ERROR("Schedule header fields overflow or exceed blob size");
        return -1;
    }

    using AicpuEntryFn = int (*)(void*, uint64_t, void*, uint64_t, void*, uint64_t,
                                  int32_t, int32_t, int32_t,
                                  const FlatSchedule*, const void*, int32_t);
    AicpuEntryFn aicpu_exec_fn = nullptr;
    const char *aicpu_path = std::getenv("SONATA_AICPU_PATH");
    if (aicpu_path != nullptr) {
        // Validate path via realpath() to prevent injection.
        char resolved_aicpu[PATH_MAX];
        if (realpath(aicpu_path, resolved_aicpu) == nullptr) {
            LOG_WARN("Cannot resolve aicpu_kernel path: %s", aicpu_path);
        } else {
            void *aicpu_handle = dlopen(resolved_aicpu, RTLD_LAZY | RTLD_GLOBAL);
            if (aicpu_handle != nullptr) {
                aicpu_exec_fn = reinterpret_cast<AicpuEntryFn>(dlsym(aicpu_handle, "sonata_standalone_interpreter"));
                if (aicpu_exec_fn == nullptr) {
                    aicpu_exec_fn = reinterpret_cast<AicpuEntryFn>(dlsym(aicpu_handle, "aicpu_entry"));
                }
            } else {
                LOG_WARN("dlopen(%s) failed: %s", aicpu_path, dlerror());
            }
        }
    }
    // Last-resort fallback: RTLD_DEFAULT (Linux with RTLD_GLOBAL lib loading)
    if (aicpu_exec_fn == nullptr) {
        aicpu_exec_fn = reinterpret_cast<AicpuEntryFn>(dlsym(RTLD_DEFAULT, "aicpu_execute"));
    }
    if (aicpu_exec_fn == nullptr) {
        LOG_ERROR("dlsym(aicpu_execute) failed: "
                  "set SONATA_AICPU_PATH to the aicpu_kernel.so path");
        return -1;
    }
    int interp_rc = aicpu_exec_fn(
        runtime_arena_dev, layout.arena_size,
        sm_ptr, sm_size,
        gm_heap, eff_heap_size,
        0, 0,  // aic_count, aiv_count (unused by interpreter)
        static_cast<int32_t>(eff_task_window_size),
        flat_sched,
        device_args.tensor_data(),
        tensor_count
    );

    int64_t t_total_end = _now_ms();
    LOG_INFO_V0("Sonata bind total: %" PRId64 "ms (interp rc=%d)", t_total_end - t_total_start, interp_rc);

    if (interp_rc != 0) {
        LOG_ERROR("Interpreter failed with rc=%d", interp_rc);
        return -1;
    }

    bind_succeeded = true;
    return 0;
}

// ── validate_runtime_impl ──
//
// Copy output tensors back from device to host and free device allocations.
// Same as upstream TMARB. The host-side schedule buffer is NOT released
// here — run_prepared may be called multiple times per prepare_callable,
// and the buffer must survive across runs. buffer is freed on the next
// prepare_callable_impl call (via _set_schedule_buf → _clear_schedule_buf)
// or when the host process exits.

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
