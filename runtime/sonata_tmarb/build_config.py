# Copyright (c) PyPTO Contributors.
# ---------------------------------------------------------------------------
# Sonata TensorMap Hybrid — interpreter replay runtime build configuration.
#
# Reuses TMARB's shared runtime infrastructure (ring buffer, TensorMap,
# scope, scheduler) but replaces the on-device orchestration .so with a
# schedule-driven interpreter loop.
#
# All paths are relative to this file's directory.

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": [
            "../tensormap_and_ringbuffer/runtime",
            "../tensormap_and_ringbuffer/orchestration",
            "../tensormap_and_ringbuffer/aicore",
            "../../common",
        ],
        "source_dirs": ["aicore"],
    },
    "aicpu": {
        "include_dirs": [
            "../tensormap_and_ringbuffer/runtime",
            "../tensormap_and_ringbuffer/orchestration",
            "../tensormap_and_ringbuffer/aicpu",
            "../tensormap_and_ringbuffer/runtime/scheduler",
            "../tensormap_and_ringbuffer/runtime/shared",
            "../tensormap_and_ringbuffer/runtime/backend",
            "../../common",
        ],
        "source_dirs": ["aicpu", "../tensormap_and_ringbuffer/runtime"],
    },
    "host": {
        "include_dirs": [
            "../tensormap_and_ringbuffer/runtime",
            "../tensormap_and_ringbuffer/orchestration",
            "../tensormap_and_ringbuffer/host",
            "../tensormap_and_ringbuffer/runtime/shared",
            "../../common",
        ],
        "source_dirs": ["host"],
    },
}
