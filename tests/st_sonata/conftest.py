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

v0.22: Also injects ``RUNTIME_CONFIG["sonata"]`` into ``kernel_config.py``
via the compile hook, so downstream runtime consumers (runtime_hook,
execute_with_sonata) can read Sonata hints from the standard config.

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
import os as _os
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


def _extract_certified_ir(program: Any) -> Any | None:
    """Run pass pipeline and return the IR after the last Simplify pass.

    This is the "certified IR" that Sonata analyzes. We take the last
    Simplify in the Default strategy (currently index 39 of 40 passes).
    """
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager
    from pypto.pypto_core import passes as _core_passes

    with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        current = program
        certified_ir = None
        for pname, pobj in zip(manager.pass_names, manager.passes):
            current = pobj(current)
            if pname == "Simplify":
                certified_ir = current
    return certified_ir


def _run_sonata_analysis(program: Any, entry_name: str) -> dict[str, Any] | None:
    """Run Sonata analysis on a program, return result dict or None on failure."""
    try:
        from pypto.backend import BackendType, is_backend_configured, set_backend_type
        from sonata.pipeline import sonata_analyze

        if not is_backend_configured():
            set_backend_type(BackendType.Ascend910B)

        certified_ir = _extract_certified_ir(program)

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

    v0.22: Also injects RUNTIME_CONFIG["sonata"] into kernel_config.py.
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

                if not is_backend_configured():
                    set_backend_type(BackendType.Ascend910B)

                certified_ir = _extract_certified_ir(program)

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

                # v0.27: Write sonata_schedule.json + sonata_schedule.bin
                try:
                    from sonata.pipeline import _write_bound_schedule
                    _write_bound_schedule(sonata_result, result, Path(str(work_dir)), program=program)
                    log.info(
                        "[SONATA] bound schedule + binary written to %s",
                        work_dir,
                    )
                except Exception as exc:
                    log.debug("[SONATA] _write_bound_schedule skipped: %s", exc)

                # v0.29 C2: Set SONATA_SCHEDULE_PATH so bind finds the binary.
                # Done in the compile hook (not execute hook) because the test
                # harness calls _execute_on_device directly, not execute_compiled.
                sched_bin = Path(str(work_dir)) / "sonata_schedule.bin"
                if sched_bin.exists():
                    _os.environ["SONATA_SCHEDULE_PATH"] = str(sched_bin)
                    log.info("[SONATA] SONATA_SCHEDULE_PATH=%s", sched_bin)

                # SIM path: SONATA_RUNTIME_MODE not set → default sim.

                # Set SONATA_AICPU_PATH so the host-side interpreter (SIM path)
                # can dlsym aicpu_execute / sonata_standalone_interpreter from the
                # aicpu_kernel.so binary.  Needed because the chip worker loads
                # the kernel .so with RTLD_LOCAL (not RTLD_GLOBAL), so
                # dlsym(RTLD_DEFAULT, ...) fails.
                _aicpu_so = (
                    Path(__file__).resolve().parents[2]
                    / "upstream" / "pypto" / "runtime" / "build" / "lib"
                    / "a2a3" / "sim" / "sonata_tmarb" / "libaicpu_kernel.so"
                )
                if _aicpu_so.exists():
                    _os.environ["SONATA_AICPU_PATH"] = str(_aicpu_so)
                    log.info("[SONATA] SONATA_AICPU_PATH=%s", _aicpu_so)

                # v0.22: Inject RUNTIME_CONFIG["sonata"] into kernel_config.py
                sonata_cfg = sonata_result.to_runtime_config()
                _patch_kernel_config_sonata(Path(str(work_dir)), sonata_cfg.to_run_config_dict())

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


def _patch_kernel_config_sonata(work_dir: Path, sonata_dict: dict) -> None:
    """Inject sonata dict into RUNTIME_CONFIG in kernel_config.py.

    v0.22 Phase 1 A3: Extends kernel_config.py with a "sonata" key
    in the RUNTIME_CONFIG dict. Uses repr() (not json.dumps) because
    kernel_config.py is Python source loaded via importlib.util, not JSON.

    Atomic write: writes to temp file, then renames.
    """
    import contextlib
    import os
    import tempfile

    config_path = work_dir / "kernel_config.py"
    if not config_path.exists():
        log.warning("[SONATA] kernel_config.py not found, skipping sonata injection")
        return

    content = config_path.read_text(encoding="utf-8")

    # Find RUNTIME_CONFIG = { ... }
    marker = "RUNTIME_CONFIG"
    idx = content.find(marker)
    if idx < 0:
        log.warning("[SONATA] RUNTIME_CONFIG not found in kernel_config.py")
        return

    # Find the closing "}" of the RUNTIME_CONFIG block
    brace_count = 0
    close_idx = -1
    for i in range(idx, len(content)):
        if content[i] == "{":
            brace_count += 1
        elif content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                close_idx = i
                break
    if close_idx < 0:
        log.warning("[SONATA] Could not find RUNTIME_CONFIG closing brace")
        return

    # Use repr() for Python-native types, NOT json.dumps
    # repr() produces True/False/None, json.dumps produces true/false/null
    sonata_repr = repr(sonata_dict)
    sonata_line = f'\t"sonata": {sonata_repr},\n'

    # Also promote block_dim / aicpu_thread_num to top-level RUNTIME_CONFIG
    # keys that the runtime actually reads (device_runner.py → Worker.run)
    extra_lines = ""
    bd = sonata_dict.get("suggested_block_dim")
    if bd is not None:
        extra_lines += f'\t"block_dim": {bd!r},\n'
    at = sonata_dict.get("suggested_aicpu_thread_num")
    if at is not None:
        extra_lines += f'\t"aicpu_thread_num": {at!r},\n'

    # Also set runtime = "sonata_tmarb" so the Worker loads the correct
    # host_runtime.so variant. Without this, the default TMARB runtime is
    # loaded and the NPU path (SONATA_RUNTIME_MODE=npu) never fires.
    extra_lines += '\t"runtime": "sonata_tmarb",\n'

    # Insert sonata dict + promoted keys before closing brace
    insert_block = sonata_line + extra_lines
    new_content = content[:close_idx] + insert_block + content[close_idx:]

    # Atomic write: write to temp, then replace
    fd, tmp_path = tempfile.mkstemp(
        dir=str(work_dir), prefix="kernel_config_", suffix=".py.tmp"
    )
    try:
        os.write(fd, new_content.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.replace(tmp_path, str(config_path))
        log.info("[SONATA] RUNTIME_CONFIG['sonata'] injected into kernel_config.py")
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _make_patched_execute(original_execute):
    """Wrap execute_compiled to apply Sonata runtime hook.

    v0.21: Replaces the previous block_dim monkeypatch with the
    formal runtime_hook.apply_sonata_runtime_hints() integration.

    v0.29 C2: Sets SONATA_SCHEDULE_PATH env var so the sonata_tmarb
    NPU runtime can find the schedule binary in the work directory.
    """
    @functools.wraps(original_execute)
    def patched_execute(work_dir, *args, **kwargs):
        # Set SONATA_SCHEDULE_PATH for the NPU runtime (if schedule exists)
        sched_path = Path(str(work_dir)) / "sonata_schedule.bin"
        if sched_path.exists():
            _os.environ["SONATA_SCHEDULE_PATH"] = str(sched_path)
            log.info("[SONATA] SONATA_SCHEDULE_PATH=%s", sched_path)

            # v0.29 C2: bind func_ids from kernel_config.py now that it exists
            _bind_schedule_from_work_dir(Path(str(work_dir)))
        else:
            _os.environ.pop("SONATA_SCHEDULE_PATH", None)

        # Apply Sonata runtime hints before execution
        try:
            from sonata.runtime_hook import apply_sonata_runtime_hints
            hints = apply_sonata_runtime_hints(
                work_dir=str(work_dir),
                block_dim=kwargs.get("block_dim"),
                aicpu_thread_num=kwargs.get("aicpu_thread_num"),
                user_block_dim=None,  # kwargs block_dim comes from RUNTIME_CONFIG, not user
            )
            if hints.sonata_applied:
                kwargs["block_dim"] = hints.block_dim
                log.info("[SONATA] hook applied: block_dim=%d (%s)", hints.block_dim, hints.reason)
            else:
                log.info("[SONATA] hook not applied: %s", hints.reason)
        except Exception as e:
            log.warning("[SONATA] hook failed: %s", e)
        return original_execute(work_dir, *args, **kwargs)
    return patched_execute


def _bind_schedule_from_work_dir(work_dir: Path) -> None:
    """Rebind func_ids in sonata_schedule.json/binary using kernel_config.py KERNELS.

    Called during the execute hook (patched_execute), when kernel_config.py
    already exists with the ``KERNELS`` list populated by compile_and_assemble.
    Fail-open: any error logs a warning and leaves the schedule unbound.
    """
    sched_json = work_dir / "sonata_schedule.json"
    if not sched_json.exists():
        return

    # Read KERNELS from kernel_config.py
    kc_path = work_dir / "kernel_config.py"
    if not kc_path.exists():
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kc_bind", str(kc_path))
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernels = getattr(mod, "KERNELS", None)
        if not kernels or not isinstance(kernels, (list, tuple)):
            return
        func_name_to_id: dict[str, int] = {}
        for k in kernels:
            if isinstance(k, dict) and "name" in k and "func_id" in k:
                func_name_to_id[str(k["name"])] = int(k["func_id"])
        if not func_name_to_id:
            return
    except Exception as exc:
        log.debug("[SONATA] bind_from_kc: read failed: %s", exc)
        return

    # Read the existing (unbound) schedule JSON and rebuild with bound func_ids
    try:
        from sonata.schedule import SonataScheduleContract
        from sonata.binding import bind_func_ids
        from pathlib import Path as _Path

        schedule = SonataScheduleContract.from_json(sched_json)
        bound, _reasons = bind_func_ids(schedule, func_name_to_id)
        bound.write_json(sched_json)
        bin_path = work_dir / "sonata_schedule.bin"
        if bin_path.exists():
            bin_path.write_bytes(bound.to_binary())
            log.info(
                "[SONATA] binding applied post-compile: %d func_ids mapped, binary rewritten (%d bytes)",
                len(func_name_to_id), len(bin_path.read_bytes()),
            )
    except Exception as exc:
        log.warning("[SONATA] bind_from_kc: rebind failed: %s", exc)
        return


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

    # Monkeypatch execute_compiled to apply Sonata runtime hook
    try:
        from pypto.runtime import runner as runner_mod
        if not getattr(runner_mod.execute_compiled, "_sonata_patched", False):
            runner_mod.execute_compiled = _make_patched_execute(runner_mod.execute_compiled)
            runner_mod.execute_compiled._sonata_patched = True
    except (ImportError, AttributeError):
        pass

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
