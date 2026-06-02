# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for Sonata analysis on pypto-lib examples.

Loads real PyPTO programs from upstream/pypto-lib/examples/ and verifies
Sonata can analyze them: eligibility, region extraction, plan generation.

Usage:
    PYTHONPATH=src:upstream/pypto/python python -m pytest tests/pypto_cases/test_pytplib_examples.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PYPTO_LIB = Path(__file__).resolve().parents[2] / "upstream" / "pypto-lib"


def _load_module(name: str, filepath: str):
    """Load a Python module from filepath."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_program_from_module(mod, builder_name: str = None):
    """Extract @pl.program from a module, trying builder function first."""
    from pypto.pypto_core import ir as _ir
    ProgramType = _ir.Program

    # Try builder function first
    if builder_name and hasattr(mod, builder_name):
        builder = getattr(mod, builder_name)
        result = builder()
        if isinstance(result, ProgramType):
            return result
        # Builder might return a class that needs to be instantiated
        if isinstance(result, type) and hasattr(result, '__call__'):
            try:
                prog = result()
                if isinstance(prog, ProgramType):
                    return prog
            except Exception:
                pass

    # Try direct @pl.program objects in module
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, ProgramType):
            return obj

    return None


def _compile_to_certified_dump(program):
    """Run PyPTO pipeline to Simplify stage."""
    from pypto.backend import BackendType, is_backend_configured, set_backend_type
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager
    from pypto.pypto_core import passes as _core_passes

    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)

    with _core_passes.PassContext([], _core_passes.VerificationLevel.NONE):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        current = program
        after_ccg = False
        for pass_name, pass_obj in zip(manager.pass_names, manager.passes):
            current = pass_obj(current)
            if pass_name == "CollectCommGroups":
                after_ccg = True
            elif after_ccg and pass_name == "Simplify":
                return current
    return None


# --- Example definitions ---

EXAMPLES = [
    ("hello_world", "examples/beginner/hello_world.py", "build_hello_world_program"),
    ("matmul", "examples/beginner/matmul.py", "build_matmul_program"),
    ("softmax", "examples/intermediate/softmax.py", "build_softmax_program"),
    ("rms_norm", "examples/intermediate/rms_norm.py", "build_rms_norm_program"),
    ("layer_norm", "examples/intermediate/layer_norm.py", "build_layer_norm_program"),
    ("rope", "examples/intermediate/rope.py", "build_rope_program"),
    ("gemm", "examples/intermediate/gemm.py", "build_gemm_program"),
    ("gemm_eltwise", "examples/advanced/gemm_eltwise.py", "build_gemm_eltwise_program"),
    ("multi_proj", "examples/advanced/multi_proj.py", "build_multi_proj_program"),
    ("topk", "examples/advanced/topk.py", "build_topk_program"),
]


class TestPyPTOLibExamples:
    """Sonata analysis on pypto-lib example programs."""

    @pytest.mark.parametrize("name,relpath,builder", EXAMPLES,
                             ids=[e[0] for e in EXAMPLES])
    def test_example_eligibility(self, name, relpath, builder):
        """Each pypto-lib example is analyzed by Sonata.

        Note: examples using pl.parallel generate ForStmt in the IR,
        which Sonata correctly rejects as control flow. These tests
        verify that the analysis runs without crashing, and eligible
        examples produce valid plans.
        """
        from sonata.pipeline import sonata_analyze

        filepath = str(_PYPTO_LIB / relpath)
        mod = _load_module(f"pytplib_{name}", filepath)
        program = _get_program_from_module(mod, builder)

        if program is None:
            pytest.skip(f"Could not extract program from {name}")

        certified = _compile_to_certified_dump(program)
        if certified is None:
            pytest.skip(f"Pipeline did not produce certified IR for {name}")

        result = sonata_analyze(certified, entry_name=name)
        # pypto-lib examples with pl.parallel generate ForStmt → correctly
        # rejected by Sonata as control flow. This is expected behavior.
        if not result.eligible:
            assert result.eligibility_result is not None
            assert any("ForStmt" in r or "control_flow" in r
                        for r in result.eligibility_result.reasons)
            return

        # If eligible, verify plan structure
        assert result.score is not None
        assert result.task_count >= 1
        assert result.has_plan

    @pytest.mark.parametrize("name,relpath,builder", EXAMPLES[:5],
                             ids=[e[0] for e in EXAMPLES[:5]])
    def test_example_region_analysis(self, name, relpath, builder):
        """Region analysis runs on pypto-lib examples without crashing."""
        from sonata.pipeline import sonata_analyze

        filepath = str(_PYPTO_LIB / relpath)
        mod = _load_module(f"pytplib_{name}_region", filepath)
        program = _get_program_from_module(mod, builder)

        if program is None:
            pytest.skip(f"Could not extract program from {name}")

        certified = _compile_to_certified_dump(program)
        if certified is None:
            pytest.skip(f"Pipeline did not produce certified IR for {name}")

        result = sonata_analyze(certified, entry_name=name)
        # pypto-lib examples may be ineligible due to ForStmt from pl.parallel
        if not result.eligible:
            # Verify the analysis ran without crashing
            assert result.eligibility_result is not None
            return

        assert result.region_tree is not None
        assert len(result.region_statuses) >= 1
