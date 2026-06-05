# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata RUNTIME_CONFIG schema.

Defines ``SonataRuntimeConfig`` — Sonata analysis results embedded in
``RUNTIME_CONFIG["sonata"]`` inside ``kernel_config.py``. This is the
first-class config field that flows through PyPTO codegen → runtime
config → simpler execution pipeline.

The ``sonata`` key is ignored by older simpler versions (they only read
``runtime``, ``block_dim``, ``aicpu_thread_num``). Newer versions can
opt in to reading it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SONATA_RUNTIME_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SonataRuntimeConfig:
    """Sonata analysis results embedded in RUNTIME_CONFIG.

    Attributes:
        eligible: Whether the program is eligible for Sonata optimization.
        task_count: Number of tasks in the Score.
        suggested_block_dim: Suggested block_dim from scheduling instructions.
        suggested_aicpu_thread_num: Suggested AICPU thread count.
        memory_peak_bytes: Peak memory from memory plan.
        region_statuses: Per-region status map (region_id → "static"/"dynamic"/"mixed").
        guard_count: Number of shape assumptions (guards).
        guard_symbols: Unique guard symbol names.
        schema_version: Schema version for forward compatibility.
    """

    eligible: bool
    task_count: int
    suggested_block_dim: int | None = None
    suggested_aicpu_thread_num: int | None = None
    memory_peak_bytes: int | None = None
    region_statuses: dict[str, str] = field(default_factory=dict)
    guard_count: int = 0
    guard_symbols: tuple[str, ...] = ()
    schema_version: int = SONATA_RUNTIME_CONFIG_SCHEMA_VERSION

    def to_run_config_dict(self) -> dict[str, Any]:
        """Emit as dict for embedding in RUNTIME_CONFIG.

        Only includes non-default values to keep kernel_config.py compact.
        """
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "eligible": self.eligible,
            "task_count": self.task_count,
        }
        if self.suggested_block_dim is not None:
            d["suggested_block_dim"] = self.suggested_block_dim
        if self.suggested_aicpu_thread_num is not None:
            d["suggested_aicpu_thread_num"] = self.suggested_aicpu_thread_num
        if self.memory_peak_bytes is not None:
            d["memory_peak_bytes"] = self.memory_peak_bytes
        if self.region_statuses:
            d["region_statuses"] = dict(self.region_statuses)
        if self.guard_count > 0:
            d["guard_count"] = self.guard_count
            d["guard_symbols"] = list(self.guard_symbols)
        return d


__all__ = [
    "SONATA_RUNTIME_CONFIG_SCHEMA_VERSION",
    "SonataRuntimeConfig",
]
