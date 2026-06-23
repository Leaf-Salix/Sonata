// test_hook_harness.cpp — C++ test harness for sonata_hook.h fail-open modes.
//
// Compiled against sonata_hook.cpp with a stub aicpu_entry.
// Validates that all 6 fail-open modes return correct status codes and
// do NOT crash (graceful degradation to original path).

#include "sonata_hook.h"
#include "flat_schedule.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
    g_tests_run++; \
    if (cond) { g_tests_passed++; printf("  PASS: %s\n", msg); } \
    else { g_tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

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

// ── Test: init/fini without SONATA_ENABLED ──

static void test_init_fini_default() {
    printf("\n[TEST] init/fini without SONATA_ENABLED\n");
    unsetenv("SONATA_ENABLED");

    int rc_init = sonata_hook_init();
    CHECK(rc_init == SONATA_HOOK_OK, "sonata_hook_init returns OK");

    g_aicpu_call_count = 0;
    FlatSchedule h = make_valid_header();
    int rc_proc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc_proc == SONATA_HOOK_DISABLED, "process_schedule returns DISABLED (no SONATA_ENABLED)");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called when disabled");

    int rc_fini = sonata_hook_fini();
    CHECK(rc_fini == SONATA_HOOK_OK, "sonata_hook_fini returns OK");
}

// ── Test: SONATA_ENABLED set, aicpu_entry returns 0 (success) ──

static void test_enabled_success() {
    printf("\n[TEST] SONATA_ENABLED set, aicpu_entry returns 0\n");
    setenv("SONATA_ENABLED", "1", 1);

    int rc_init = sonata_hook_init();
    CHECK(rc_init == SONATA_HOOK_OK, "sonata_hook_init returns OK");

    g_aicpu_call_count = 0;
    g_aicpu_return_code = 0;
    FlatSchedule h = make_valid_header();
    int rc_proc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc_proc == SONATA_HOOK_OK, "process_schedule returns OK");
    CHECK(g_aicpu_call_count == 1, "aicpu_entry called once");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: SONATA_ENABLED set, aicpu_entry returns -2 (device init failure) ──

static void test_enabled_aicpu_fails() {
    printf("\n[TEST] SONATA_ENABLED set, aicpu_entry returns -2\n");
    setenv("SONATA_ENABLED", "1", 1);

    int rc_init = sonata_hook_init();
    CHECK(rc_init == SONATA_HOOK_OK, "sonata_hook_init returns OK");

    g_aicpu_call_count = 0;
    g_aicpu_return_code = -2;
    FlatSchedule h = make_valid_header();
    int rc_proc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc_proc == SONATA_HOOK_ERROR, "process_schedule returns ERROR (aicpu failed)");
    CHECK(g_aicpu_call_count == 1, "aicpu_entry called once despite failure");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: B4 mode 1 — schedule blob NULL ──

static void test_null_blob() {
    printf("\n[TEST] B4 mode 1: null blob\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    g_aicpu_call_count = 0;
    int rc = sonata_hook_process_schedule(nullptr, 0);
    CHECK(rc == SONATA_HOOK_ERROR, "null blob returns ERROR");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called for null blob");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: B4 mode 1b — blob too small ──

static void test_too_small_blob() {
    printf("\n[TEST] B4 mode 1b: blob too small (4 bytes)\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    g_aicpu_call_count = 0;
    uint32_t tiny = 0x534F4E41;
    int rc = sonata_hook_process_schedule(&tiny, sizeof(tiny));
    CHECK(rc == SONATA_HOOK_ERROR, "too-small blob returns ERROR");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called for too-small blob");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: B4 mode 2 — wrong magic ──

static void test_wrong_magic() {
    printf("\n[TEST] B4 mode 2: wrong magic (0xDEADBEEF)\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    g_aicpu_call_count = 0;
    FlatSchedule h = make_valid_header();
    h.magic = 0xDEADBEEF;
    int rc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc == SONATA_HOOK_ERROR, "wrong magic returns ERROR");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called for wrong magic");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: B4 mode 3 — wrong version ──

static void test_wrong_version() {
    printf("\n[TEST] B4 mode 3: wrong version (99)\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    g_aicpu_call_count = 0;
    FlatSchedule h = make_valid_header();
    h.version = 99;
    int rc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc == SONATA_HOOK_ERROR, "wrong version returns ERROR");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called for wrong version");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: B4 mode 4 — negative region count (bounds violation) ──

static void test_negative_regions() {
    printf("\n[TEST] B4 mode 4: negative num_regions (-1)\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    g_aicpu_call_count = 0;
    FlatSchedule h = make_valid_header();
    h.num_regions = -1;
    int rc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc == SONATA_HOOK_ERROR, "negative regions returns ERROR");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called for invalid header");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: B4 — truncated blob (header OK but arrays too short) ──

static void test_truncated_blob() {
    printf("\n[TEST] B4: truncated blob (header claims 1 region, blob too short)\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    g_aicpu_call_count = 0;
    FlatSchedule h = make_valid_header();
    h.num_regions = 1;  // claims 1 region (24 bytes) but blob is only sizeof(header)
    int rc = sonata_hook_process_schedule(&h, sizeof(h));
    CHECK(rc == SONATA_HOOK_ERROR, "truncated blob returns ERROR");
    CHECK(g_aicpu_call_count == 0, "aicpu_entry NOT called for truncated blob");

    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: valid minimal schedule with one empty static region ──

static void test_valid_minimal_schedule() {
    printf("\n[TEST] valid minimal schedule (1 empty static region)\n");
    setenv("SONATA_ENABLED", "1", 1);
    sonata_hook_init();

    // Build: header + 1 empty static region (0 tasks, 0 deps)
    size_t blob_size = sizeof(FlatSchedule) + sizeof(FlatRegion);
    uint8_t* blob = new uint8_t[blob_size];
    std::memset(blob, 0, blob_size);

    FlatSchedule* h = reinterpret_cast<FlatSchedule*>(blob);
    h->magic = 0x534F4E41;
    h->version = 1;
    h->num_regions = 1;

    FlatRegion* r = reinterpret_cast<FlatRegion*>(blob + sizeof(FlatSchedule));
    r->kind = 0;  // static
    r->scope_mode = 0;  // auto
    r->task_start = 0;
    r->num_tasks = 0;
    r->dep_start = 0;
    r->num_deps = 0;

    g_aicpu_call_count = 0;
    g_aicpu_return_code = 0;
    int rc = sonata_hook_process_schedule(blob, blob_size);
    CHECK(rc == SONATA_HOOK_OK, "valid minimal schedule returns OK");
    CHECK(g_aicpu_call_count == 1, "aicpu_entry called for valid schedule");

    delete[] blob;
    sonata_hook_fini();
    unsetenv("SONATA_ENABLED");
}

// ── Test: sonata_hook_info on valid blob (with region array) ──

static void test_hook_info() {
    printf("\n[TEST] sonata_hook_info on valid blob (1 region)\n");

    // Build a blob with header + 1 region (zero tasks/deps) so validate_schedule
    // sees a complete, consistent binary.
    size_t blob_size = sizeof(FlatSchedule) + 1 * sizeof(FlatRegion);
    uint8_t* blob = new uint8_t[blob_size];
    std::memset(blob, 0, blob_size);
    FlatSchedule* h = reinterpret_cast<FlatSchedule*>(blob);
    h->magic = 0x534F4E41;
    h->version = 1;
    h->num_regions = 1;
    h->total_tasks = 0;
    h->total_args = 0;
    h->total_deps = 0;
    memcpy(h->fingerprint, "test_fp_123", 12);

    SonataScheduleInfo info;
    std::memset(&info, 0xFF, sizeof(info));
    int rc = sonata_hook_info(blob, blob_size, &info);
    CHECK(rc == SONATA_HOOK_OK, "hook_info returns OK");
    CHECK(info.num_regions == 1, "info.num_regions == 1");
    CHECK(info.total_tasks == 0, "info.total_tasks == 0");
    CHECK(info.total_args == 0, "info.total_args == 0");
    CHECK(info.total_deps == 0, "info.total_deps == 0");
    CHECK(memcmp(info.fingerprint, "test_fp_123", 12) == 0, "info.fingerprint matches");
    delete[] blob;
}

// ── Test: sonata_hook_info on invalid blob ──

static void test_hook_info_invalid() {
    printf("\n[TEST] sonata_hook_info on invalid blob\n");

    FlatSchedule h = make_valid_header();
    h.magic = 0;  // invalid
    SonataScheduleInfo info;
    int rc = sonata_hook_info(&h, sizeof(h), &info);
    CHECK(rc == SONATA_HOOK_ERROR, "hook_info returns ERROR for invalid blob");
}

// ── Test: null pointer to hook_info ──

static void test_hook_info_null() {
    printf("\n[TEST] sonata_hook_info with null output pointer\n");

    FlatSchedule h = make_valid_header();
    int rc = sonata_hook_info(&h, sizeof(h), nullptr);
    CHECK(rc == SONATA_HOOK_ERROR, "hook_info returns ERROR for null output");
}

// ── Main ──

int main() {
    printf("=== Sonata Hook Fail-Open Test Harness ===\n");
    printf("sizeof(FlatSchedule) = %zu\n", sizeof(FlatSchedule));
    printf("sizeof(FlatRegion)   = %zu\n", sizeof(FlatRegion));
    printf("sizeof(FlatTask)     = %zu\n", sizeof(FlatTask));
    printf("sizeof(FlatArg)      = %zu\n", sizeof(FlatArg));
    printf("sizeof(FlatDep)      = %zu\n", sizeof(FlatDep));

    CHECK(sizeof(FlatSchedule) == 88, "FlatSchedule is 88 bytes (packed)");
    CHECK(sizeof(FlatRegion) == 24, "FlatRegion is 24 bytes (packed)");
    CHECK(sizeof(FlatTask) == 16, "FlatTask is 16 bytes (packed)");
    CHECK(sizeof(FlatArg) == 6, "FlatArg is 6 bytes (packed)");
    CHECK(sizeof(FlatDep) == 8, "FlatDep is 8 bytes (packed)");

    test_init_fini_default();       // B3: no SONATA_ENABLED → disabled
    test_enabled_success();         // enabled path, aicpu succeeds
    test_enabled_aicpu_fails();     // enabled path, aicpu fails
    test_null_blob();               // B4 mode 1
    test_too_small_blob();          // B4 mode 1b
    test_wrong_magic();             // B4 mode 2
    test_wrong_version();           // B4 mode 3
    test_negative_regions();        // B4 mode 4 (bounds)
    test_truncated_blob();          // B4 mode 4b (bounds)
    test_valid_minimal_schedule();  // valid schedule (baseline)
    test_hook_info();               // introspection OK
    test_hook_info_invalid();       // introspection on bad blob
    test_hook_info_null();          // introspection with null out

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           g_tests_passed, g_tests_run, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
