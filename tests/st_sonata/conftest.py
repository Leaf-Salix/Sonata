# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata-integrated st test runner.

Hooks into the PyPTO st test infrastructure to run Sonata analysis on
the certified IR dump (after Simplify pass) and write ``sonata_plan.json``
alongside compiled artifacts — without modifying any upstream test files.

The hook monkeypatches ``compile_program`` so that after the real
compilation completes, Sonata analysis runs on the same program and
writes ``sonata_plan.json`` into the compilation work directory.

The PyPTO runner's optional Sonata hook (in ``runtime_hook.py``) then
automatically consumes ``sonata_plan.json`` at execution time — no
block_dim monkeypatch needed.

Usage::

    python tests/st_sonata/sonata_st_runner.py tests/st/runtime/ops/test_abs.py -- -v
"""

from __future__ import annotations

import functools
import logging
import warnings
from pathlib import Path
from typing import Any

import pytest

log = logging.getLogger("sonata.st_runner")

# Session-level results collector for B2 summary report
_session_results: list[dict[str, Any]] = []

# Cache: program id → sonata result dict (avoids double pipeline replay)
_analysis_cache: dict[int, dict[str, Any]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--with-sonata",
        action="store_true",
        default=False,
        help="Run Sonata analysis on each st test's certified IR dump and write sonata_plan.json.",
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
            return None

        result = sonata_analyze(certified_ir, entry_name=entry_name)
        return {
            "eligible": result.eligible,
            "task_count": result.task_count,
            "region_statuses": result.region_statuses,
            "has_plan": result.has_plan,
            "result_obj": result,
        }
    except ImportError:
        return None
    except Exception as e:
        return {"error": str(e)}


def _make_patched_compile(original_compile):
    """Wrap compile_program to run Sonata analysis after compilation.

    Uses cached analysis from pytest_runtest_setup when available,
    avoiding a second pipeline replay.
    """
    @functools.wraps(original_compile)
    def patched_compile(program, work_dir, **kwargs):
        result = original_compile(program, work_dir, **kwargs)
        log.info("[SONATA] compile_program completed, work_dir=%s", work_dir)
        try:
            prog_id = id(program)
            cached = _analysis_cache.get(prog_id)

            if cached is None:
                # Fallback: run analysis if not cached from setup
                from sonata.pipeline import sonata_analyze
                from pypto.backend import BackendType, is_backend_configured, set_backend_type
                from pypto.ir.pass_manager import OptimizationStrategy, PassManager
                from pypto.pypto_core import passes as _core_passes

                if not is_backend_configured():
                    set_backend_type(BackendType.Ascend910B)

                with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
                    mgr = PassManager.get_strategy(OptimizationStrategy.Default)
                    cur = program
                    after_ccg = False
                    certified_ir = None
                    for pn, po in zip(mgr.pass_names, mgr.passes):
                        cur = po(cur)
                        if pn == "CollectCommGroups":
                            after_ccg = True
                        elif after_ccg and pn == "Simplify":
                            certified_ir = cur
                            break

                if certified_ir is not None:
                    sonata_result = sonata_analyze(certified_ir, entry_name=Path(str(work_dir)).name)
                else:
                    sonata_result = None
            else:
                sonata_result = cached.get("result_obj")

            if sonata_result is not None and sonata_result.eligible:
                plan_path = sonata_result.save(Path(str(work_dir)) / "sonata_plan.json")
                log.info("[SONATA] sonata_plan.json written: %s", plan_path)
                persistent_dir = Path(__file__).resolve().parents[2] / "build" / "sonata_plans"
                persistent_dir.mkdir(parents=True, exist_ok=True)
                persistent_path = sonata_result.save(persistent_dir / f"{Path(str(work_dir)).name}_sonata_plan.json")
                log.info("[SONATA] persistent copy: %s", persistent_path)

                # Run region dispatch (informational)
                from sonata.pipeline import dispatch_regions
                dispatch = dispatch_regions(sonata_result)
                log.info(
                    "[SONATA] dispatch: %d optimized, %d fallback, %d mixed",
                    dispatch.optimized_count, dispatch.fallback_count, dispatch.mixed_count,
                )
            else:
                log.info("[SONATA] analysis: not eligible or not available")
        except Exception as e:
            log.warning("[SONATA] compile-time analysis failed: %s", e, exc_info=True)

        return result
    return patched_compile


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Before each test, monkeypatch compile_program and log analysis."""
    if not item.config.getoption("--with-sonata", default=False):
        return

    # Monkeypatch compile_program in the harness test_runner module
    # pytest adds tests/st to sys.path during collection, so harness is importable
    patched_ok = False
    for mod_name in ("harness.core.test_runner", "tests.st.harness.core.test_runner"):
        try:
            tr = __import__(mod_name, fromlist=["compile_program"])
            from pypto.runtime import compile_program as orig_compile
            if not getattr(tr.compile_program, "_sonata_patched", False):
                tr.compile_program = _make_patched_compile(orig_compile)
                tr.compile_program._sonata_patched = True
            patched_ok = True
            break
        except (ImportError, AttributeError, ValueError):
            continue

    # Standalone analysis for logging
    module = getattr(item, "module", None)
    if module is None:
        return

    program = _extract_program_from_module(module)
    if program is None:
        return

    test_name = item.name
    verbose = not item.config.getoption("quiet", default=False)

    analysis = _run_sonata_analysis(program, entry_name=test_name)

    # Cache for compile hook to reuse (avoids second pipeline replay)
    if analysis is not None and "error" not in analysis:
        _analysis_cache[id(program)] = analysis

    if analysis is None or "error" in analysis:
        _session_results.append({"test": test_name, "status": "error", "error": analysis.get("error", "unknown") if analysis else "no analysis"})
        return

    regions = analysis.get("region_statuses", {})
    sc = sum(1 for v in regions.values() if v == "static")
    dc = sum(1 for v in regions.values() if v == "dynamic")

    _session_results.append({
        "test": test_name,
        "status": "analyzed",
        "eligible": analysis["eligible"],
        "task_count": analysis["task_count"],
        "region_count": len(regions),
        "static_regions": sc,
        "dynamic_regions": dc,
        "has_plan": analysis.get("has_plan", False),
    })

    verbose = not item.config.getoption("quiet", default=False)
    if verbose:
        msg = (
            f"[SONATA] {test_name}: eligible={analysis['eligible']}, "
            f"tasks={analysis['task_count']}, "
            f"regions={len(regions)} ({sc} static, {dc} dynamic)"
        )
        if analysis.get("has_plan"):
            msg += ", plan=generated"
        warnings.warn(msg, stacklevel=1)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print Sonata analysis summary report at end of session."""
    if not session.config.getoption("--with-sonata", default=False):
        return
    if not _session_results:
        return

    analyzed = [r for r in _session_results if r["status"] == "analyzed"]
    errors = [r for r in _session_results if r["status"] == "error"]
    eligible = [r for r in analyzed if r.get("eligible")]
    with_plan = [r for r in analyzed if r.get("has_plan")]

    total_tasks = sum(r.get("task_count", 0) for r in analyzed)
    total_static = sum(r.get("static_regions", 0) for r in analyzed)
    total_dynamic = sum(r.get("dynamic_regions", 0) for r in analyzed)

    print("\n" + "=" * 60)
    print("SONATA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Tests analyzed:    {len(analyzed)}")
    print(f"  Eligible:          {len(eligible)}/{len(analyzed)}")
    print(f"  Plans generated:   {len(with_plan)}/{len(analyzed)}")
    print(f"  Errors:            {len(errors)}")
    print(f"  Total tasks:       {total_tasks}")
    print(f"  Total regions:     {total_static + total_dynamic} ({total_static} static, {total_dynamic} dynamic)")
    if errors:
        print(f"\n  Failed tests:")
        for r in errors:
            print(f"    - {r['test']}: {r.get('error', 'unknown')}")
    print("=" * 60 + "\n")
