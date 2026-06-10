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
imports. It reads Sonata hints from two possible sources (in priority order):

1. ``sonata_plan.json`` — full Sonata analysis result (preferred)
2. ``RUNTIME_CONFIG["sonata"]`` in ``kernel_config.py`` — compact config field

In both cases it computes runtime hints (block_dim, aicpu_thread_num) without
touching simpler C++ or PyPTO internals.

Design constraints:
- No import of simpler or PyPTO C++ bindings
- No modification of CallConfig wire layout
- Fail-open: any error returns original parameters unchanged
- User-supplied block_dim always takes precedence
- Guard metadata is informational only; guard evaluation is handled by
  ``execute_with_sonata()``, not by this advisory hook
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
    user_block_dim: int | None = None,
) -> SonataRuntimeHints:
    """Compute runtime hints from sonata_plan.json if available.

    This function is called by PyPTO's ``execute_compiled()`` as an optional
    pre-dispatch hook. It follows these rules:

    1. No ``sonata_plan.json`` → return original parameters unchanged.
    2. Plan not eligible → return original parameters unchanged.
    3. User explicitly supplied ``block_dim`` → don't override.
    4. ``block_dim`` came from RUNTIME_CONFIG (not user) → use Sonata hint.
    5. Any failure → fail open: log warning, return original parameters.

    Args:
        work_dir: Compiled artifacts directory (contains sonata_plan.json).
        block_dim: Current effective block_dim (may come from RUNTIME_CONFIG).
        aicpu_thread_num: Caller-supplied aicpu_thread_num (None = not specified).
        user_block_dim: The original user-supplied block_dim BEFORE RUNTIME_CONFIG
            fallback. None means user didn't specify. When not None, Sonata
            will not override block_dim.

    Returns:
        SonataRuntimeHints with (possibly updated) parameters.
    """
    try:
        return _do_apply(work_dir, block_dim, aicpu_thread_num, user_block_dim)
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
    user_block_dim: int | None,
) -> SonataRuntimeHints:
    work_path = Path(work_dir)
    plan_path = work_path / "sonata_plan.json"

    # Strategy 1: Read sonata_plan.json (preferred — full data)
    if plan_path.exists():
        try:
            hints = _apply_from_plan_json(
                plan_path, block_dim, aicpu_thread_num, user_block_dim,
            )
            if hints is not None:
                return hints
        except Exception:
            log.warning(
                "[Sonata] plan.json read failed, trying kernel_config fallback",
                exc_info=True,
            )
            # Fall through to strategy 2

    # Strategy 2: Read RUNTIME_CONFIG["sonata"] from kernel_config.py
    config_path = work_path / "kernel_config.py"
    if config_path.exists():
        return _apply_from_kernel_config(
            config_path, block_dim, aicpu_thread_num, user_block_dim,
        )

    return _unchanged(block_dim, aicpu_thread_num, "no_sonata_data")


def _apply_from_plan_json(
    plan_path: Path,
    block_dim: int | None,
    aicpu_thread_num: int | None,
    user_block_dim: int | None,
) -> SonataRuntimeHints | None:
    """Read Sonata hints from sonata_plan.json. Returns None if absent."""
    if not plan_path.exists():
        return None

    plan_data = json.loads(plan_path.read_text())
    if not plan_data.get("eligible", False):
        return _unchanged(block_dim, aicpu_thread_num, "plan_not_eligible")

    if user_block_dim is not None:
        return _unchanged(block_dim, aicpu_thread_num, "user_supplied_block_dim")

    # Lazy import to avoid circular dependency: pipeline.py imports runtime_hook.py
    from .pipeline import SonataAnalysisResult, compute_scheduling_instructions, dispatch_regions

    region_statuses = plan_data.get("region_statuses", {})
    if not region_statuses:
        return _unchanged(block_dim, aicpu_thread_num, "no_regions")

    temp_result = SonataAnalysisResult(
        eligible=True,
        region_statuses=region_statuses,
    )

    dispatch = dispatch_regions(temp_result)
    instructions = compute_scheduling_instructions(dispatch)

    if not instructions:
        return _unchanged(block_dim, aicpu_thread_num, "no_scheduling_instructions")

    suggested_block_dim = instructions[0].block_dim
    reason = instructions[0].reason

    log.info(
        "[Sonata] hook from plan_json: block_dim=%d (%s)", suggested_block_dim, reason,
    )
    return SonataRuntimeHints(
        block_dim=suggested_block_dim,
        aicpu_thread_num=aicpu_thread_num,
        sonata_applied=True,
        reason=reason,
    )


def _apply_from_kernel_config(
    config_path: Path,
    block_dim: int | None,
    aicpu_thread_num: int | None,
    user_block_dim: int | None,
) -> SonataRuntimeHints:
    """Read RUNTIME_CONFIG["sonata"] from kernel_config.py.

    Uses exec_module() to load kernel_config.py — only call on trusted paths
    (the compiled artifacts directory, written by the project's own codegen).
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kc", str(config_path))
        if spec is None or spec.loader is None:
            return _unchanged(block_dim, aicpu_thread_num, "config_load_failed")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sonata_raw = getattr(mod, "RUNTIME_CONFIG", {}).get("sonata")
    except Exception as exc:
        log.warning("[Sonata] kernel_config load failed: %s", exc)
        return _unchanged(block_dim, aicpu_thread_num, f"config_load_error: {exc}")

    from .runtime_config import SonataRuntimeConfig
    sonata_cfg = SonataRuntimeConfig.from_dict(sonata_raw)
    if sonata_cfg is None:
        return _unchanged(block_dim, aicpu_thread_num, "invalid_sonata_config")

    if not sonata_cfg.eligible:
        return _unchanged(block_dim, aicpu_thread_num, "plan_not_eligible")

    if user_block_dim is not None:
        return _unchanged(block_dim, aicpu_thread_num, "user_supplied_block_dim")

    if sonata_cfg.suggested_block_dim is None and sonata_cfg.suggested_aicpu_thread_num is None:
        return _unchanged(block_dim, aicpu_thread_num, "no_suggestions")

    new_block_dim = sonata_cfg.suggested_block_dim if sonata_cfg.suggested_block_dim is not None else block_dim
    new_aicpu = sonata_cfg.suggested_aicpu_thread_num if sonata_cfg.suggested_aicpu_thread_num is not None else aicpu_thread_num

    log.info(
        "[Sonata] hook from kernel_config: block_dim=%s, aicpu_thread_num=%s",
        new_block_dim, new_aicpu,
    )
    return SonataRuntimeHints(
        block_dim=new_block_dim,
        aicpu_thread_num=new_aicpu,
        sonata_applied=True,
        reason="runtime_config_sonata",
    )


def _unchanged(
    block_dim: int | None,
    aicpu_thread_num: int | None,
    reason: str,
) -> SonataRuntimeHints:
    """Return original params unchanged with a reason."""
    return SonataRuntimeHints(
        block_dim=block_dim,
        aicpu_thread_num=aicpu_thread_num,
        sonata_applied=False,
        reason=reason,
    )


__all__ = [
    "SonataRuntimeHints",
    "apply_sonata_runtime_hints",
]
