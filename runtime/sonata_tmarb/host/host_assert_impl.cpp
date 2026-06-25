// Stub for assert_impl — required by runtime/common.h (included from shared modules).
// Matches C++ linkage from runtime/common.h (not extern "C").
#include <cstdio>
#include <cstdlib>

[[noreturn]] void assert_impl(const char *condition, const char *file, int line) {
    fprintf(stderr, "Assertion failed: %s\n  at %s:%d\n", condition, file, line);
    abort();
}
