# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata-integrated st test runner.

Runs any upstream ``pypto/tests/st`` test through Sonata analysis before
execution, without modifying the upstream test files.

Usage::

    # Single test with Sonata
    python -m pytest tests/st/runtime/ops/test_abs.py --with-sonata --platform=a2a3sim -v

    # All ops tests with Sonata
    python -m pytest tests/st/runtime/ops/ --with-sonata --platform=a2a3sim -v

The ``--with-sonata`` flag is a no-op when Sonata is not installed.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

log = logging.getLogger("sonata.st_runner")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --with-sonata flag to pytest."""
    parser.addoption(
        "--with-sonata",
        action="store_true",
        default=False,
        help="Run Sonata analysis on each st test's certified IR dump before execution.",
    )


def _extract_program_from_module(module: Any) -> Any | None:
    """Find the first @pl.program object in a module."""
    try:
        from pypto.pypto_core import ir as _ir
        ProgramType = _ir.Program
    except ImportError:
        return None

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, ProgramType):
            return obj
    return None


def _run_sonata_analysis(program: Any, entry_name: str) -> dict[str, Any] | None:
    """Run Sonata analysis on a program, return result dict or None on failure."""
    try:
        from pypto.backend import BackendType, is_backend_configured, set_backend_type
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager
        from pypto.pypto_core import passes as _core_passes
        from sonata.pipeline import sonata_analyze

        if not is_backend_configured():
            set_backend_type(BackendType.Ascend910B)

        with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
            manager = PassManager.get_strategy(OptimizationStrategy.Default)
            current = program
            after_ccg = False
            certified_ir = None
            for pname, pobj in zip(manager.pass_names, manager.passes):
                current = pobj(current)
                if pname == "CollectCommGroups":
                    after_ccg = True
                elif after_ccg and pname == "Simplify":
                    certified_ir = current
                    break

        if certified_ir is None:
            return {"error": "no certified dump found"}

        result = sonata_analyze(certified_ir, entry_name=entry_name)
        return {
            "eligible": result.eligible,
            "task_count": result.task_count,
            "region_statuses": result.region_statuses,
            "has_plan": result.has_plan,
        }
    except ImportError:
        return None
    except Exception as e:
        return {"error": str(e)}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Before each test, run Sonata analysis if --with-sonata is set."""
    if not item.config.getoption("--with-sonata", default=False):
        return

    module = getattr(item, "module", None)
    if module is None:
        return

    program = _extract_program_from_module(module)
    if program is None:
        return

    test_name = item.name
    verbose = not item.config.getoption("quiet", default=False)

    analysis = _run_sonata_analysis(program, entry_name=test_name)

    if analysis is None:
        return

    if "error" in analysis:
        if verbose:
            warnings.warn(f"[SONATA] {test_name}: analysis error: {analysis['error']}", stacklevel=1)
        return

    if verbose:
        regions = analysis.get("region_statuses", {})
        static_count = sum(1 for v in regions.values() if v == "static")
        dynamic_count = sum(1 for v in regions.values() if v == "dynamic")
        msg = (
            f"[SONATA] {test_name}: eligibility={analysis['eligible']}, "
            f"tasks={analysis['task_count']}, "
            f"regions={len(regions)} ({static_count} static, {dynamic_count} dynamic)"
        )
        if analysis.get("has_plan"):
            msg += ", HostBuildGraphPlan=generated"
        warnings.warn(msg, stacklevel=1)
