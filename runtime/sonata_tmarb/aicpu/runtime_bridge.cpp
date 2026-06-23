// Minimal implementation of orchestration TLS bridge for sonata aicpu.
// Replaces orchestration/common.cpp to avoid pulling in pto_orchestration_api.h
// which redefines PTO2RuntimeOps and LOG macros.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

struct PTO2Runtime;

static PTO2Runtime *g_current_runtime = nullptr;

extern "C" void framework_bind_runtime(PTO2Runtime *rt) {
    g_current_runtime = rt;
}

extern "C" PTO2Runtime *framework_current_runtime() {
    return g_current_runtime;
}

// Minimal assert_impl — aborts with message.  Matches C++ linkage
// from runtime/common.h (not extern "C").
[[noreturn]] void assert_impl(const char *condition, const char *file, int line) {
    fprintf(stderr, "Assertion failed: %s\n  at %s:%d\n", condition, file, line);
    abort();
}
