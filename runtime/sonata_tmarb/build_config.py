# Copyright (c) PyPTO Contributors.
# ---------------------------------------------------------------------------
# Sonata TensorMap Hybrid — interpreter replay runtime build configuration.
#
# Reuses TMARB's shared runtime infrastructure (ring buffer, TensorMap,
# scope, scheduler) but replaces the on-device orchestration .so with a
# schedule-driven interpreter loop.
#
# All paths are relative to this file's directory (sonata_tmarb/).

UPSTREAM_TMARB = "../../upstream/pypto/runtime/src/a2a3/runtime/tensormap_and_ringbuffer"

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": [
            f"{UPSTREAM_TMARB}/runtime",
            f"{UPSTREAM_TMARB}/common",
            f"{UPSTREAM_TMARB}/orchestration",
            f"{UPSTREAM_TMARB}/aicore",
            f"{UPSTREAM_TMARB}/..",
            "../../upstream/pypto/runtime/src/common",
        ],
        "source_dirs": [
            f"{UPSTREAM_TMARB}/aicore",
        ],
    },
    "aicpu": {
        "include_dirs": [
            f"{UPSTREAM_TMARB}/runtime",
            f"{UPSTREAM_TMARB}/common",
            f"{UPSTREAM_TMARB}/orchestration",
            f"{UPSTREAM_TMARB}/aicpu",
            f"{UPSTREAM_TMARB}/..",
            f"{UPSTREAM_TMARB}/runtime/scheduler",
            f"{UPSTREAM_TMARB}/runtime/shared",
            f"{UPSTREAM_TMARB}/runtime/backend",
            "../../upstream/pypto/runtime/src/common",
            "runtime",
            "include",
        ],
        # aicpu source: local interpreter executor + upstream runtime shared
        # infrastructure (runtime_init_data_from_layout, rt_submit_*_task,
        # DeviceArena, etc.). The interpreter does NOT use the host-side
        # orchestration flow or the AICPU scheduler; their symbols are
        # discarded by the linker because nothing references them.
        #
        # CMake's CUSTOM_SOURCE_DIRS only supports directory-level recursive
        # globs, so including the runtime root necessarily pulls in
        # pto_orchestrator.cpp, pto_ring_buffer.cpp, and scheduler/*.cpp
        # even though the interpreter doesn't need them. The only file from
        # the root that IS needed is pto_runtime2.cpp (runtime2 core APIs).
        "source_dirs": [
            "aicpu",
            f"{UPSTREAM_TMARB}/runtime",
        ],
    },
    "host": {
        "include_dirs": [
            f"{UPSTREAM_TMARB}/runtime",
            f"{UPSTREAM_TMARB}/common",
            f"{UPSTREAM_TMARB}/orchestration",
            f"{UPSTREAM_TMARB}/host",
            f"{UPSTREAM_TMARB}/runtime/shared",
            f"{UPSTREAM_TMARB}/..",
            "../../upstream/pypto/runtime/src/common",
            "../../upstream/pypto/runtime/src/common/log/include",
            "runtime",
            "include",
        ],
        "source_dirs": ["host"],
    },
}
