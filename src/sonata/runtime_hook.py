# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata runtime hook — thin advisory layer consumed by PyPTO runner.

This module is the ONLY Sonata entry point that PyPTO's ``execute_compiled()``
imports. It loads ``sonata_plan.json`` from the work directory and computes
runtime hints (block_dim, aicpu_thread_num) without touching simpler C++ or
PyPTO internals.

Design constraints:
- No import of simpler or PyPTO C++ bindings
- No modification of CallConfig wire layout
- Fail-open: any error returns original parameters unchanged
- User-supplied block_dim always takes precedence
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("sonata.runtime_hook")


@dataclass(frozen=True)
class SonataRuntimeHints:
    """Runtime parameters suggested by Sonata analysis."""

    block_dim: int | None
    aicpu_thread_num: int | None
    sonata_applied: bool
    reason: str


def apply_sonata_runtime_hints(
    *,
    work_dir: str | Path,
    block_dim: int | None,
    aicpu_thread_num: int | None,
) -> SonataRuntimeHints:
    """Compute runtime hints from sonata_plan.json if available.

    This function is called by PyPTO's ``execute_compiled()`` as an optional
    pre-dispatch hook. It follows these rules:

    1. No ``sonata_plan.json`` → return original parameters unchanged.
    2. Plan not eligible → return original parameters unchanged.
    3. User already supplied ``block_dim`` (not None) → don't override.
    4. User didn't supply ``block_dim`` → use Sonata's scheduling instruction.
    5. Any failure → fail open: log warning, return original parameters.

    Args:
        work_dir: Compiled artifacts directory (contains sonata_plan.json).
        block_dim: Caller-supplied block_dim (None = not specified).
        aicpu_thread_num: Caller-supplied aicpu_thread_num (None = not specified).

    Returns:
        SonataRuntimeHints with (possibly updated) parameters.
    """
    try:
        return _do_apply(work_dir, block_dim, aicpu_thread_num)
    except Exception as exc:
        log.warning("[Sonata] hook failed, using original params: %s", exc)
        return SonataRuntimeHints(
            block_dim=block_dim,
            aicpu_thread_num=aicpu_thread_num,
            sonata_applied=False,
            reason=f"hook_error: {exc}",
        )


def _do_apply(
    work_dir: str | Path,
    block_dim: int | None,
    aicpu_thread_num: int | None,
) -> SonataRuntimeHints:
    plan_path = Path(work_dir) / "sonata_plan.json"
    if not plan_path.exists():
        return SonataRuntimeHints(
            block_dim=block_dim,
            aicpu_thread_num=aicpu_thread_num,
            sonata_applied=False,
            reason="no_sonata_plan",
        )

    plan_data = json.loads(plan_path.read_text())
    if not plan_data.get("eligible", False):
        return SonataRuntimeHints(
            block_dim=block_dim,
            aicpu_thread_num=aicpu_thread_num,
            sonata_applied=False,
            reason="plan_not_eligible",
        )

    # User explicitly supplied block_dim — don't override
    if block_dim is not None:
        return SonataRuntimeHints(
            block_dim=block_dim,
            aicpu_thread_num=aicpu_thread_num,
            sonata_applied=False,
            reason="user_supplied_block_dim",
        )

    # Compute scheduling instruction from region dispatch
    region_statuses = plan_data.get("region_statuses", {})
    if not region_statuses:
        return SonataRuntimeHints(
            block_dim=block_dim,
            aicpu_thread_num=aicpu_thread_num,
            sonata_applied=False,
            reason="no_regions",
        )

    # Determine block_dim from first region's status
    first_status = next(iter(region_statuses.values()))
    if first_status == "static":
        suggested_block_dim = 32
        reason = "static_region_optimized"
    elif first_status == "dynamic":
        suggested_block_dim = 1
        reason = "dynamic_region_fallback"
    else:  # mixed
        suggested_block_dim = 16
        reason = "mixed_region_conservative"

    log.info(
        "[Sonata] hook applied: block_dim=%d (%s)", suggested_block_dim, reason,
    )

    return SonataRuntimeHints(
        block_dim=suggested_block_dim,
        aicpu_thread_num=aicpu_thread_num,
        sonata_applied=True,
        reason=reason,
    )


__all__ = [
    "SonataRuntimeHints",
    "apply_sonata_runtime_hints",
]
