// Stub for assert_impl — required by runtime/common.h (included from shared modules).
// Matches C++ linkage from runtime/common.h (not extern "C").
// Marked __attribute__((weak)) so host_runtime.so's definition does not conflict
// with aicpu_kernel.so's definition when both are loaded in the same process
// (aicpu_kernel.so is loaded first by ChipWorker with RTLD_GLOBAL on some platforms).
#include <cstdio>
#include <cstdlib>

[[noreturn]] __attribute__((weak)) void assert_impl(const char *condition, const char *file, int line) {
    fprintf(stderr, "Assertion failed: %s\n  at %s:%d\n", condition, file, line);
    abort();
}
