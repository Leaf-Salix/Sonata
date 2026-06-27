// Minimal implementation of orchestration TLS bridge for sonata aicpu.
// Replaces orchestration/common.cpp to avoid pulling in pto_orchestration_api.h
// which redefines PTO2RuntimeOps and LOG macros.
//
// Weak linkage: when both this file and upstream orchestration/common.cpp are
// compiled into the same aicpu_kernel.so (NPU dual-path build), the upstream
// strong symbols override these. When only this file is compiled (sim standalone
// path), these weak symbols serve as the sole implementation.
// See ADR-002 §B3 for the rationale.

#include <cstdio>
#include <cstdlib>

struct PTO2Runtime;

static PTO2Runtime *g_current_runtime = nullptr;

extern "C" __attribute__((weak)) void framework_bind_runtime(PTO2Runtime *rt) {
    g_current_runtime = rt;
}

extern "C" __attribute__((weak)) PTO2Runtime *framework_current_runtime() {
    return g_current_runtime;
}

// Minimal assert_impl — aborts with message.  Matches C++ linkage
// from runtime/common.h (not extern "C").
__attribute__((weak)) [[noreturn]] void assert_impl(const char *condition, const char *file, int line) {
    fprintf(stderr, "Assertion failed: %s\n  at %s:%d\n", condition, file, line);
    abort();
}
