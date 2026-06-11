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

    This class carries scheduling hints (block_dim, aicpu_thread_num)
    for the advisory ``apply_sonata_runtime_hints()`` hook, and a
    reference to ``sonata_schedule.json`` for optional backend consumption.

    Note: Guard metadata (guard_count, guard_symbols) is informational only.
    Runtime guard evaluation (HARD → skip, STALE → rebuild) is handled by
    ``execute_with_sonata()``, not by this config class.

    Attributes:
        eligible: Whether the program is eligible for Sonata optimization.
        task_count: Number of tasks in the Score.
        suggested_block_dim: Suggested block_dim from scheduling instructions.
        suggested_aicpu_thread_num: Suggested AICPU thread count.
        memory_peak_bytes: Peak memory from memory plan.
        region_statuses: Per-region status map (region_id → "static"/"dynamic"/"mixed").
        guard_count: Number of shape assumptions (guards) — informational only.
        guard_symbols: Unique guard symbol names — informational only.
        schedule_path: Path to sonata_schedule.json relative to work_dir.
            None when no schedule artifact exists (v0.23+).
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
    schedule_path: str | None = None
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
        if self.schedule_path is not None:
            d["schedule_path"] = self.schedule_path
        return d

    @classmethod
    def from_dict(cls, d: Any) -> SonataRuntimeConfig | None:
        """Load from RUNTIME_CONFIG["sonata"] dict. Returns None on error."""
        if not isinstance(d, dict):
            return None
        version = d.get("schema_version", 0)
        if version < 1 or version > SONATA_RUNTIME_CONFIG_SCHEMA_VERSION:
            return None
        try:
            return cls(
                eligible=d.get("eligible", False),
                task_count=d.get("task_count", 0),
                suggested_block_dim=d.get("suggested_block_dim"),
                suggested_aicpu_thread_num=d.get("suggested_aicpu_thread_num"),
                memory_peak_bytes=d.get("memory_peak_bytes"),
                region_statuses=d.get("region_statuses", {}) or {},
                guard_count=d.get("guard_count", 0),
                guard_symbols=tuple(d.get("guard_symbols", ())),
                schedule_path=d.get("schedule_path"),
                schema_version=version,
            )
        except (TypeError, ValueError):
            return None


__all__ = [
    "SONATA_RUNTIME_CONFIG_SCHEMA_VERSION",
    "SonataRuntimeConfig",
]
