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
            f"{UPSTREAM_TMARB}/orchestration",
            f"{UPSTREAM_TMARB}/aicore",
            "../../upstream/pypto/runtime/src/common",
        ],
        "source_dirs": [
            f"{UPSTREAM_TMARB}/aicore",
        ],
    },
    "aicpu": {
        "include_dirs": [
            f"{UPSTREAM_TMARB}/runtime",
            f"{UPSTREAM_TMARB}/orchestration",
            f"{UPSTREAM_TMARB}/aicpu",
            f"{UPSTREAM_TMARB}/runtime/scheduler",
            f"{UPSTREAM_TMARB}/runtime/shared",
            f"{UPSTREAM_TMARB}/runtime/backend",
            "../../upstream/pypto/runtime/src/common",
            "runtime",
        ],
        "source_dirs": [
            "aicpu",
            f"{UPSTREAM_TMARB}/runtime",
        ],
    },
    "host": {
        "include_dirs": [
            f"{UPSTREAM_TMARB}/runtime",
            f"{UPSTREAM_TMARB}/orchestration",
            f"{UPSTREAM_TMARB}/host",
            f"{UPSTREAM_TMARB}/runtime/shared",
            "../../upstream/pypto/runtime/src/common",
            "runtime",
        ],
        "source_dirs": ["host"],
    },
}
