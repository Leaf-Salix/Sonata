// test_hook_harness.cpp — gtest harness for sonata_hook.h fail-open modes.
//
// Compiled against sonata_hook.cpp with a stub aicpu_entry.
// Validates that all 6 fail-open modes return correct status codes and
// do NOT crash (graceful degradation to original path).

#include <gtest/gtest.h>

#include "sonata_hook.h"
#include "flat_schedule.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_aicpu_call_count = 0;
static int g_aicpu_return_code = 0;

// Stub: replaces real aicpu_entry (which requires TMARB runtime).
// Returns g_aicpu_return_code; increments counter for diagnostics.
extern "C" int aicpu_entry(void*, uint64_t,
                           void*, uint64_t,
                           void*, uint64_t,
                           int32_t, int32_t,
                           int32_t,
                           const FlatSchedule*,
                           const void*, int32_t) {
    g_aicpu_call_count++;
    return g_aicpu_return_code;
}

// ── Helpers ──

static FlatSchedule make_valid_header() {
    FlatSchedule h;
    std::memset(&h, 0, sizeof(h));
    h.magic = 0x534F4E41;  // "SONA"
    h.version = 1;
    h.num_regions = 0;
    h.total_tasks = 0;
    h.total_args = 0;
    h.total_deps = 0;
    return h;
}

class SonataHook : public ::testing::Test {
protected:
    void SetUp() override {
        g_aicpu_call_count = 0;
        g_aicpu_return_code = 0;
    }
    void TearDown() override {
        sonata_hook_fini();
        unsetenv("SONATA_ENABLED");
    }
};

// ── Test: struct sizes match flat_schedule.h packing ──

TEST_F(SonataHook, StructSizesArePacked) {
    EXPECT_EQ(sizeof(FlatSchedule), 88u);
    EXPECT_EQ(sizeof(FlatRegion), 24u);
    EXPECT_EQ(sizeof(FlatTask), 16u);
    EXPECT_EQ(sizeof(FlatArg), 6u);
    EXPECT_EQ(sizeof(FlatDep), 8u);
}

// ── Test: init/fini without SONATA_ENABLED ──

TEST_F(SonataHook, DisabledModeSkipsAicpu) {
    unsetenv("SONATA_ENABLED");

    EXPECT_EQ(sonata_hook_init(), SONATA_HOOK_OK);

    FlatSchedule h = make_valid_header();
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_DISABLED);
    EXPECT_EQ(g_aicpu_call_count, 0);

    EXPECT_EQ(sonata_hook_fini(), SONATA_HOOK_OK);
}

// ── Test: SONATA_ENABLED set, aicpu_entry returns 0 (success) ──

TEST_F(SonataHook, EnabledSuccessCallsAicpu) {
    setenv("SONATA_ENABLED", "1", 1);

    EXPECT_EQ(sonata_hook_init(), SONATA_HOOK_OK);

    FlatSchedule h = make_valid_header();
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_OK);
    EXPECT_EQ(g_aicpu_call_count, 1);
}

// ── Test: SONATA_ENABLED set, aicpu_entry returns -2 (device init failure) ──

TEST_F(SonataHook, EnabledAicpuFailureReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    g_aicpu_return_code = -2;

    EXPECT_EQ(sonata_hook_init(), SONATA_HOOK_OK);

    FlatSchedule h = make_valid_header();
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 1);
}

// ── Test: B4 mode 1 — schedule blob NULL ──

TEST_F(SonataHook, NullBlobReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    EXPECT_EQ(sonata_hook_process_schedule(nullptr, 0), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 0);
}

// ── Test: B4 mode 1b — blob too small ──

TEST_F(SonataHook, TooSmallBlobReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    uint32_t tiny = 0x534F4E41;
    EXPECT_EQ(sonata_hook_process_schedule(&tiny, sizeof(tiny)), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 0);
}

// ── Test: B4 mode 2 — wrong magic ──

TEST_F(SonataHook, WrongMagicReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    FlatSchedule h = make_valid_header();
    h.magic = 0xDEADBEEF;
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 0);
}

// ── Test: B4 mode 3 — wrong version ──

TEST_F(SonataHook, WrongVersionReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    FlatSchedule h = make_valid_header();
    h.version = 99;
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 0);
}

// ── Test: B4 mode 4 — negative region count (bounds violation) ──

TEST_F(SonataHook, NegativeRegionsReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    FlatSchedule h = make_valid_header();
    h.num_regions = -1;
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 0);
}

// ── Test: B4 — truncated blob (header OK but arrays too short) ──

TEST_F(SonataHook, TruncatedBlobReturnsError) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    FlatSchedule h = make_valid_header();
    h.num_regions = 1;  // claims 1 region (24 bytes) but blob is only sizeof(header)
    EXPECT_EQ(sonata_hook_process_schedule(&h, sizeof(h)), SONATA_HOOK_ERROR);
    EXPECT_EQ(g_aicpu_call_count, 0);
}

// ── Test: valid minimal schedule with one empty static region ──

TEST_F(SonataHook, ValidMinimalScheduleReturnsOk) {
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    // Build: header + 1 empty static region (0 tasks, 0 deps)
    size_t blob_size = sizeof(FlatSchedule) + sizeof(FlatRegion);
    std::vector<uint8_t> blob(blob_size, 0);

    FlatSchedule* h = reinterpret_cast<FlatSchedule*>(blob.data());
    h->magic = 0x534F4E41;
    h->version = 1;
    h->num_regions = 1;

    FlatRegion* r = reinterpret_cast<FlatRegion*>(blob.data() + sizeof(FlatSchedule));
    r->kind = 0;  // static
    r->scope_mode = 0;  // auto
    r->task_start = 0;
    r->num_tasks = 0;
    r->dep_start = 0;
    r->num_deps = 0;

    EXPECT_EQ(sonata_hook_process_schedule(blob.data(), blob_size), SONATA_HOOK_OK);
    EXPECT_EQ(g_aicpu_call_count, 1);
}

// ── Test: sonata_hook_info on valid blob (with region array) ──

TEST_F(SonataHook, HookInfoOnValidBlob) {
    // Build a blob with header + 1 region (zero tasks/deps) so validate_schedule
    // sees a complete, consistent binary.
    size_t blob_size = sizeof(FlatSchedule) + 1 * sizeof(FlatRegion);
    std::vector<uint8_t> blob(blob_size, 0);
    FlatSchedule* h = reinterpret_cast<FlatSchedule*>(blob.data());
    h->magic = 0x534F4E41;
    h->version = 1;
    h->num_regions = 1;
    h->total_tasks = 0;
    h->total_args = 0;
    h->total_deps = 0;
    memcpy(h->fingerprint, "test_fp_123", 12);

    SonataScheduleInfo info;
    std::memset(&info, 0xFF, sizeof(info));
    EXPECT_EQ(sonata_hook_info(blob.data(), blob_size, &info), SONATA_HOOK_OK);
    EXPECT_EQ(info.num_regions, 1);
    EXPECT_EQ(info.total_tasks, 0);
    EXPECT_EQ(info.total_args, 0);
    EXPECT_EQ(info.total_deps, 0);
    EXPECT_EQ(memcmp(info.fingerprint, "test_fp_123", 12), 0);
}

// ── Test: sonata_hook_info on invalid blob ──

TEST_F(SonataHook, HookInfoOnInvalidBlobReturnsError) {
    FlatSchedule h = make_valid_header();
    h.magic = 0;  // invalid
    SonataScheduleInfo info;
    EXPECT_EQ(sonata_hook_info(&h, sizeof(h), &info), SONATA_HOOK_ERROR);
}

// ── Test: null pointer to hook_info ──

TEST_F(SonataHook, HookInfoWithNullOutputReturnsError) {
    FlatSchedule h = make_valid_header();
    EXPECT_EQ(sonata_hook_info(&h, sizeof(h), nullptr), SONATA_HOOK_ERROR);
}

// ── Cross-language file validation mode ──
//
// Reads a .bin file (produced by Python's to_binary()), sets SONATA_ENABLED,
// and validates it through the full hook pipeline. This exercises the
// cross-language round-trip: Python serialization → C validation + dispatch.
// Not a gtest case — invoked directly from main() when argv[1] is a path,
// so test_fail_open.py's test_cross_language_binary_validation can keep
// calling the harness binary with a file argument.

static bool try_load_and_validate(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "ERROR: empty file %s\n", path);
        fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    size_t nread = fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    if (static_cast<long>(nread) != sz) {
        fprintf(stderr, "ERROR: short read %s\n", path);
        return false;
    }

    setenv("SONATA_ENABLED", "1", 1);
    if (sonata_hook_init() != SONATA_HOOK_OK) {
        fprintf(stderr, "ERROR: sonata_hook_init failed\n");
        unsetenv("SONATA_ENABLED");
        return false;
    }

    g_aicpu_return_code = 0;
    int rc = sonata_hook_process_schedule(buf.data(), buf.size());
    if (rc != SONATA_HOOK_OK) {
        printf("FAIL: process_schedule returned %d for %s\n", rc, path);
        sonata_hook_fini();
        unsetenv("SONATA_ENABLED");
        return false;
    }
    printf("OK: process_schedule OK for %s (aicpu_entry called %d time(s))\n",
           path, g_aicpu_call_count);

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
    return true;
}

int main(int argc, char** argv) {
    // InitGoogleTest strips --gtest_* flags from argv first, so a leftover
    // positional arg here is a real file path (cross-language mode), not a
    // gtest flag this harness doesn't otherwise consume.
    ::testing::InitGoogleTest(&argc, argv);
    if (argc >= 2) {
        bool ok = try_load_and_validate(argv[1]);
        return ok ? 0 : 1;
    }

    return RUN_ALL_TESTS();
}
