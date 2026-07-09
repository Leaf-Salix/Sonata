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

#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <limits.h>
#include <string>

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

// ── Ring config helpers (mirroring upstream TMARB) ──
static bool is_power_of_2_u64(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

template <typename T>
static std::string format_ring_array(const T (&values)[PTO2_MAX_RING_DEPTH]) {
    std::string out = "[";
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; ++r) {
        if (r != 0) out += ", ";
        out += std::to_string(values[r]);
    }
    out += "]";
    return out;
}

static std::string trim_copy(const std::string &input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) ++begin;
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;
    return input.substr(begin, end - begin);
}

static bool parse_uint_token(const char *name, const std::string &raw, uint64_t min_val, uint64_t max_val,
                              bool require_power_of_2, uint64_t *out) {
    std::string token = trim_copy(raw);
    if (token.empty()) { LOG_WARN("%s has empty value, ignored", name); return false; }
    char *endptr = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(token.c_str(), &endptr, 10);
    if (errno == ERANGE || endptr == token.c_str() || *endptr != '\0') {
        LOG_WARN("%s=%s invalid integer, ignored", name, token.c_str()); return false;
    }
    uint64_t val = static_cast<uint64_t>(parsed);
    if (val < min_val || val > max_val) {
        LOG_WARN("%s=%s out of range [%" PRIu64 ", %" PRIu64 "], ignored", name, token.c_str(), min_val, max_val);
        return false;
    }
    if (require_power_of_2 && !is_power_of_2_u64(val)) {
        LOG_WARN("%s=%s not a power of 2, ignored", name, token.c_str()); return false;
    }
    *out = val;
    return true;
}

static void apply_env_ring_values(const char *name, uint64_t min_val, uint64_t max_val, bool require_power_of_2,
                                   uint64_t out[PTO2_MAX_RING_DEPTH]) {
    const char *env = std::getenv(name);
    if (!env) return;
    std::string text(env);
    if (text.find(',') == std::string::npos) {
        uint64_t value = 0;
        if (!parse_uint_token(name, text, min_val, max_val, require_power_of_2, &value)) return;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) out[r] = value;
        return;
    }
    uint64_t parsed[PTO2_MAX_RING_DEPTH]{};
    size_t pos = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        size_t comma = text.find(',', pos);
        std::string token = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!parse_uint_token(name, token, min_val, max_val, require_power_of_2, &parsed[r])) return;
        if (comma == std::string::npos) {
            if (r != PTO2_MAX_RING_DEPTH - 1) {
                LOG_WARN("%s: expected %d comma-separated values, got fewer", name, PTO2_MAX_RING_DEPTH);
                return;
            }
            pos = text.size();
        } else { pos = comma + 1; }
    }
    if (pos < text.size() || (!text.empty() && text.back() == ',')) {
        LOG_WARN("%s: expected %d comma-separated values, got more", name, PTO2_MAX_RING_DEPTH);
        return;
    }
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) out[r] = parsed[r];
}

static uint64_t read_ring_override(const uint64_t *base, int idx) {
    if (base == nullptr) return 0;
    uint64_t value;
    std::memcpy(&value, base + idx, sizeof(value));
    return value;
}

static bool resolve_ring_config(const uint64_t *ring_task_window, const uint64_t *ring_heap,
                                 const uint64_t *ring_dep_pool,
                                 uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
                                 uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH],
                                 int32_t eff_dep_pool_capacities[PTO2_MAX_RING_DEPTH]) {
    uint64_t dep_pool_values[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        eff_task_window_sizes[r] = PTO2_TASK_WINDOW_SIZE;
        eff_heap_sizes[r] = PTO2_HEAP_SIZE;
        dep_pool_values[r] = PTO2_DEP_LIST_POOL_SIZE;
    }
    apply_env_ring_values("PTO2_RING_TASK_WINDOW", 4, static_cast<uint64_t>(INT32_MAX), true, eff_task_window_sizes);
    apply_env_ring_values("PTO2_RING_HEAP", 1024, std::numeric_limits<uint64_t>::max(), false, eff_heap_sizes);
    apply_env_ring_values("PTO2_RING_DEP_POOL", 4, static_cast<uint64_t>(INT32_MAX), false, dep_pool_values);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        uint64_t val = read_ring_override(ring_task_window, r);
        if (val != 0) eff_task_window_sizes[r] = val;
        val = read_ring_override(ring_heap, r);
        if (val != 0) eff_heap_sizes[r] = val;
        val = read_ring_override(ring_dep_pool, r);
        if (val != 0) dep_pool_values[r] = val;
        if (eff_task_window_sizes[r] < 4 || eff_task_window_sizes[r] > static_cast<uint64_t>(INT32_MAX) ||
            !is_power_of_2_u64(eff_task_window_sizes[r])) {
            LOG_ERROR("ring_task_window[%d]=%" PRIu64 " must be power of 2 in [4, INT32_MAX]",
                      r, eff_task_window_sizes[r]);
            return false;
        }
        if (eff_heap_sizes[r] < 1024) {
            LOG_ERROR("ring_heap[%d]=%" PRIu64 " must be >= 1024", r, eff_heap_sizes[r]);
            return false;
        }
        if (dep_pool_values[r] < 4 || dep_pool_values[r] > static_cast<uint64_t>(INT32_MAX)) {
            LOG_ERROR("ring_dep_pool[%d]=%" PRIu64 " must be in [4, INT32_MAX]", r, dep_pool_values[r]);
            return false;
        }
        eff_dep_pool_capacities[r] = static_cast<int32_t>(dep_pool_values[r]);
    }
    return true;
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
    *out_size = g_schedule_size;
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

    // Stash a host-side copy ONLY if the binary looks like a FlatSchedule
    // (magic = 0x534F4E41).  Standard compile_and_assemble puts an orchestration
    // ELF in binary_data(), which is not a valid schedule.  The env-var fallback
    // (SONATA_SCHEDULE_PATH) is the primary delivery mechanism for now.
    if (orch_so_size >= sizeof(FlatSchedule)) {
        auto *hdr = reinterpret_cast<const FlatSchedule *>(orch_so);
        if (hdr->magic == FLAT_SCHEDULE_MAGIC) {
            if (!_set_schedule_buf(orch_so, orch_so_size)) {
                LOG_WARN("Sonata prepare: failed to stash schedule (%zu bytes)", orch_so_size);
            }
        }
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
    int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
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

    // ── Resolve per-ring config (mirroring upstream TMARB) ──
    uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH];
    int32_t eff_dep_pool_capacities[PTO2_MAX_RING_DEPTH];
    if (!resolve_ring_config(ring_task_window, ring_heap, ring_dep_pool,
                              eff_task_window_sizes, eff_heap_sizes, eff_dep_pool_capacities)) {
        return -1;
    }
    const std::string tw_log = format_ring_array(eff_task_window_sizes);
    const std::string hp_log = format_ring_array(eff_heap_sizes);
    const std::string dp_log = format_ring_array(eff_dep_pool_capacities);
    LOG_INFO_V0("Ring config: task_window=%s heap=%s dep_pool=%s", tw_log.c_str(), hp_log.c_str(), dp_log.c_str());

    uint64_t total_heap_size = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (eff_heap_sizes[r] > std::numeric_limits<uint64_t>::max() - total_heap_size) {
            LOG_ERROR("Total ring heap size overflows uint64_t");
            return -1;
        }
        total_heap_size += eff_heap_sizes[r];
    }
    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(eff_task_window_sizes);
    int32_t eff_dep_pool_capacity = eff_dep_pool_capacities[0];  // used only by SIM path

    DeviceArena host_arena;

    // ── Extract flat_schedule before arena reservation ──
    // Need schedule size to reserve arena space for embedding the data block
    // that the AICPU's sonata_orchestrate_with_schedule() reads.
    size_t flat_sched_size = 0;
    const FlatSchedule *flat_sched = _find_stashed_schedule(&flat_sched_size);
    fprintf(stderr, "C2_DEBUG: find_stashed_schedule: sched=%p size=%zu\n",
            (void*)flat_sched, flat_sched_size);

    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(
        host_arena, eff_task_window_sizes, eff_heap_sizes, eff_dep_pool_capacities);
    // NOTE: scheduler_timeout_ms stays 0 (default) because
    // resolve_scheduler_timeout_ms() is a static function in the upstream TMARB
    // runtime_maker.cpp that is not available in the sonata_tmarb build unit.
    fprintf(stderr, "C2_DEBUG: reserve_layout done arena_size=%zu off_runtime=%zu\n",
            layout.arena_size, layout.off_runtime);

    // ── Reserve space for sonata data block in the prebuilt arena ──
    // Layout: uint64_t sentinel, int32_t tensor_count, 52-byte padding,
    // Tensor[tensor_count], FlatSchedule.  The Tensor struct has alignas(64),
    // so the Tensor array must start at a 64-byte aligned offset within the
    // block.  Sentinel is for C2 PROOF: host writes 0, AICPU overwrites with
    // 0xCAFEBABE+0xFACEFEED on entry; validate_runtime_impl reads back.
    static constexpr size_t kTensorArrayOff = 64;  // alignas(64) for Tensor[]
    size_t sched_block_size = kTensorArrayOff;
    sched_block_size += static_cast<size_t>(tensor_count) * sizeof(Tensor);
    if (flat_sched != nullptr) {
        sched_block_size += flat_sched_size;
    }
    size_t sched_block_off = host_arena.reserve(sched_block_size, 64);
    size_t actual_arena_size = host_arena.total_size();
    fprintf(stderr, "C2_DEBUG: sched block off=%zu size=%zu actual_arena=%zu\n",
            sched_block_off, sched_block_size, actual_arena_size);

    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena");
        return -1;
    }

    if (runtime->host_api.setup_static_arena(total_heap_size, sm_size, actual_arena_size) != 0) {
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
    runtime->set_gm_heap(gm_heap);
    runtime->set_gm_sm_ptr(sm_ptr);
    runtime->set_orch_args(device_args);

    // PTO2_ORCH_TO_SCHED (same as upstream TMARB)
    {
        const char *env_val = std::getenv("PTO2_ORCH_TO_SCHED");
        if (env_val && (env_val[0] == '1' || env_val[0] == 't' || env_val[0] == 'T')) {
            runtime->orch_to_sched = true;
        }
    }

    // ── NPU path: prebuilt arena init + sonata schedule upload ──
    {
        const char *rt_mode = std::getenv("SONATA_RUNTIME_MODE");
        fprintf(stderr, "C2_DEBUG: rt_mode=%s\n", rt_mode ? rt_mode : "NULL");
        if (rt_mode != nullptr && strcmp(rt_mode, "npu") == 0) {
            fprintf(stderr, "C2_DEBUG: arena init\n");
            PTO2Runtime *rt = runtime_init_data_from_layout(
                host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size,
                gm_heap, eff_heap_sizes);
            if (rt == nullptr) {
                LOG_ERROR("NPU: runtime_init_data_from_layout failed");
                return -1;
            }
            runtime_wire_arena_pointers(host_arena, layout, rt);
            rt->prebuilt_layout = layout;
            fprintf(stderr, "C2_DEBUG: init + wire done\n");

            // ── Embed sonata data block in prebuilt arena ──
            // Tensor registry + FlatSchedule at sched_block_off, read on AICPU
            // via (uint8_t*)rt + rt->total_cycles after attach().
            {
                uint8_t *arena_base = static_cast<uint8_t*>(host_arena.base());

                // Sentinel at offset 0 (C2 PROOF marker, initially 0)
                uint64_t *sentinel = reinterpret_cast<uint64_t*>(arena_base + sched_block_off);
                *sentinel = 0;

                // Tensor count at offset 8
                int32_t *reg_count = reinterpret_cast<int32_t*>(arena_base + sched_block_off + 8);
                *reg_count = tensor_count;

                // Tensor array at 64-byte aligned offset (alignas(64) Tensor)
                Tensor *reg_tensors = reinterpret_cast<Tensor*>(
                    arena_base + sched_block_off + kTensorArrayOff);
                for (int i = 0; i < tensor_count; i++) {
                    reg_tensors[i] = device_args.tensor(i);
                }

                // FlatSchedule after tensors (if available)
                if (flat_sched != nullptr && flat_sched_size >= sizeof(FlatSchedule)) {
                    uint8_t *sched_dst = arena_base + sched_block_off + kTensorArrayOff
                                       + static_cast<size_t>(tensor_count) * sizeof(Tensor);
                    std::memcpy(sched_dst, flat_sched, flat_sched_size);
                }
                fprintf(stderr, "C2_DEBUG: data block written (%d tensors, %zu sched)\n",
                        tensor_count, flat_sched_size);
            }

            // Set total_cycles to the AICPU-accessible offset from rt to the
            // schedule data block within the prebuilt arena.  Non-zero triggers
            // sonata_orchestrate_with_schedule() on the AICPU instead of TMARB.
            int64_t sonata_off = static_cast<int64_t>(sched_block_off)
                               - static_cast<int64_t>(layout.off_runtime);
            rt->total_cycles = sonata_off;
            fprintf(stderr, "C2_DEBUG: total_cycles=%ld (sched_off=%zu rt_off=%zu)\n",
                    (long)sonata_off, sched_block_off, layout.off_runtime);

            int rc = runtime->host_api.copy_to_device(
                runtime_arena_dev, host_arena.base(), actual_arena_size);
            if (rc != 0) {
                LOG_ERROR("NPU: copy_to_device(arena) failed rc=%d", rc);
                return -1;
            }
            runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);

            // Store sentinel probe address for C2 PROOF verification.
            // validate_runtime_impl reads this back via copy_from_device to
            // confirm sonata_orchestrate_with_schedule() was entered on AICPU.
            {
                uint64_t sentinel_dev = reinterpret_cast<uint64_t>(runtime_arena_dev)
                                      + sched_block_off;
                runtime->set_sonata_schedule(sentinel_dev, sizeof(uint64_t));
                fprintf(stderr, "C2_DEBUG: sentinel probe addr=0x%llx\n",
                        (unsigned long long)sentinel_dev);
            }

            fprintf(stderr, "C2_DEBUG: arena upload OK (%zu bytes) total_cycles=%ld\n",
                    actual_arena_size, (long)sonata_off);
            bind_succeeded = true;
            return 0;
        }
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
        gm_heap, eff_heap_sizes[0],
        0, 0,  // aic_count, aiv_count (unused by interpreter)
        static_cast<int32_t>(eff_task_window_sizes[0]),
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
    fprintf(stderr, "C2_PROOF: validate entered, sched_addr=0x%lx sched_size=%lu\n",
            (unsigned long)runtime->get_sonata_sched_addr(),
            (unsigned long)runtime->get_sonata_sched_size());

    // ── SONATA PROOF: read schedule buffer sentinel from device memory ──
    // If sonata_orchestrate_with_schedule() was entered on the AICPU, it
    // overwrote the first 8 bytes of the schedule buffer with 0xCAFEBABE
    // and 0xFACEFEED.  Reading them back proves the sonata path was taken.
    // Works on both sim (same address space) and NPU (HBM shared memory).
    {
        uint64_t probe_addr = runtime->get_sonata_sched_addr();
        if (probe_addr != 0 && runtime->get_sonata_sched_size() >= 8) {
            uint32_t marker[2] = {0, 0};
            int rc = runtime->host_api.copy_from_device(
                marker, reinterpret_cast<void*>(static_cast<uintptr_t>(probe_addr)), 8);
            if (rc == 0 && marker[0] == 0xCAFEBABE && marker[1] == 0xFACEFEED) {
                LOG_INFO_V0("[SONATA PROOF] orchestrator branch executed on AICPU");
                fprintf(stderr, "C2_PROOF: SONATA ORCHESTRATOR EXECUTED ON AICPU: 0x%08x 0x%08x\n",
                        marker[0], marker[1]);
            } else if (rc == 0) {
                LOG_INFO_V0("[SONATA PROOF] NOT executed (magic=0x%08x 0x%08x)",
                            marker[0], marker[1]);
                fprintf(stderr, "C2_PROOF: NOT executed (magic=0x%08x 0x%08x)\n",
                        marker[0], marker[1]);
            } else {
                fprintf(stderr, "C2_PROOF: copy_from_device failed rc=%d\n", rc);
            }
        } else {
            fprintf(stderr, "C2_PROOF: probe_addr=0x%lx size=%lu\n",
                    (unsigned long)probe_addr, (unsigned long)runtime->get_sonata_sched_size());
        }
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
