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
            f"{UPSTREAM_TMARB}/runtime/scheduler",
            f"{UPSTREAM_TMARB}/..",
            "../../upstream/pypto/runtime/src/common",
            "../../upstream/pypto/runtime/src/common/log/include",
            "../../upstream/pypto/runtime/src/common/platform/sim/aicpu",
            "../../upstream/pypto/runtime/src/common/platform/sim/host",
            "../../upstream/pypto/runtime/src/common/platform/shared",
            "../../upstream/pypto/runtime/src/common/platform/include",
            f"{UPSTREAM_TMARB}/../../platform/include/aicpu",
            f"{UPSTREAM_TMARB}/../../platform/shared",
            f"{UPSTREAM_TMARB}/../../platform/shared/aicpu",
            "runtime",
            "include",
        ],
        # host source_dirs 仅包含 sonata 特有文件 + runtime/shared 基础设施。
        # 平台特定的源文件（sim/onboard）由各平台 CMakeLists.txt 管理，不在
        # build_config.py 中重复添加，以免 onboard 编译时引入 sim 特有代码。
        "source_dirs": [
            "host",
            f"{UPSTREAM_TMARB}/runtime/shared",
        ],
    },
}

# NPU dual-path build (ADR-002):
# Before building for a2a3 onboard, apply the upstream patch:
#   ./patches/apply.sh ../../upstream/pypto/runtime/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/
