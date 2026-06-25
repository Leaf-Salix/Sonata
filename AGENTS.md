# Sonata Codex Instructions

This repository is the Sonata project root. PyPTO should be treated as an
external upstream dependency under `upstream/pypto/`, ideally as a git
submodule.

## Working Rules

- Keep Sonata-owned code in `src/sonata/`.
- Keep PyPTO-independent tests in `tests/sonata/`.
- Keep PyPTO compatibility/extraction tests in `tests/pypto_cases/`.
- Keep PyPTO modifications as patches under `patches/pypto/` unless the user
  explicitly asks to edit the submodule worktree.
- Do not mix Codex automation reports into this repository; those live in the
  outer `../reports/` workspace.
- Prefer extracting structural facts from PyPTO over depending on volatile
  runtime/orchestrator behavior.

## Verification

Use Python 3.10+.

```bash
PYTHONPATH=src python -m pytest tests/sonata
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/pypto_cases
```

## C++ Test Frameworks

- Sonata's own C++ test (`tests/sonata/tools/test_hook_harness.cpp`) uses
  GoogleTest (`TEST_F`/`EXPECT_*`), matching upstream's style. Built by
  `tests/sonata/tools/Makefile`, which searches `/opt/homebrew` (macOS) or
  `/usr/local` (Linux) for `libgtest.a`/`libgtest_main.a` — install via
  `brew install googletest` or `apt install libgtest-dev` first (Ubuntu's
  package ships source only; build it once and `cp lib/*.a /usr/local/lib/`,
  see the CI step in `.github/workflows/sonata-tests.yml`). The Makefile has
  no `FetchContent`-style fallback, unlike upstream's CMake build.
  `tests/sonata/test_fail_open.py` runs the compiled binary with
  `--gtest_output=json:...` and asserts against the structured results
  rather than parsing stdout text. The binary also supports a non-gtest
  cross-language mode: `test_hook_harness <path/to/schedule.bin>` validates
  a Python-produced binary schedule directly (no gtest flags consumed).
- Upstream PyPTO's C++ unit tests
  (`upstream/pypto/runtime/tests/ut/cpp/`) also use GoogleTest, wired through
  `CMakeLists.txt` (system `find_library(gtest)` with a `FetchContent`
  fallback). See `upstream/pypto/runtime/docs/troubleshooting/ut-cpp-gtest-abi.md`
  for ABI pitfalls when linking a hand-built gtest.
