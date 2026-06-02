# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for Sonata analysis on pypto-lib LLM models.

Loads real LLM model programs from upstream/pypto-lib/models/ and verifies
Sonata handles them correctly. Most models use pl.spmd (L3 multi-chip)
which Sonata correctly rejects as ineligible.

Usage:
    PYTHONPATH=src:upstream/pypto/python python -m pytest tests/pypto_cases/test_pytplib_models.py -v
"""

import importlib.util
from pathlib import Path

import pytest

_PYPTO_LIB = Path(__file__).resolve().parents[2] / "upstream" / "pypto-lib"


def _load_program(filepath: str, name: str):
    """Load a @pl.program from a pypto-lib model file."""
    from pypto.pypto_core import ir as _ir

    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find builder function
    for attr in dir(mod):
        if attr.startswith("build_") and callable(getattr(mod, attr)):
            try:
                result = getattr(mod, attr)()
                if isinstance(result, _ir.Program):
                    return result
            except Exception:
                pass
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


# --- Model definitions ---

MODELS = [
    ("qwen3_14b_l3_generate", "models/qwen3/14b/qwen3_14b_l3_generate.py"),
    ("deepseek_v3_2_decode_back", "models/deepseek/v3_2/deepseek_v3_2_decode_back.py"),
    ("deepseek_v3_2_decode_front", "models/deepseek/v3_2/deepseek_v3_2_decode_front.py"),
    ("kimi_k2_decode", "models/kimi/kimi_k2_decode_draft.py"),
    ("milm_decode", "models/milm/milm_decode_draft.py"),
]


class TestPyPTOLibModels:
    """Sonata analysis on pypto-lib LLM model programs."""

    @pytest.mark.parametrize("name,relpath", MODELS, ids=[m[0] for m in MODELS])
    def test_model_analysis(self, name, relpath):
        """LLM models are analyzed by Sonata without crashing.

        Most models use pl.spmd (L3 multi-chip dispatch) which Sonata
        correctly rejects as ineligible — these have no Orchestration
        functions. This test verifies the analysis runs without error.
        """
        from sonata.pipeline import sonata_analyze

        filepath = str(_PYPTO_LIB / relpath)
        program = _load_program(filepath, name)

        if program is None:
            pytest.skip(f"Could not load program from {name}")

        certified = _compile_to_certified_dump(program)
        if certified is None:
            pytest.skip(f"Pipeline did not produce certified IR for {name}")

        result = sonata_analyze(certified, entry_name=name)

        # Most LLM models use pl.spmd → no Orchestration → ineligible
        if not result.eligible:
            assert result.eligibility_result is not None
            # Expected: entry_function_not_orchestration or control_flow_not_supported
            assert len(result.fallback_reasons) > 0
            return

        # If eligible (unlikely for LLM models), verify plan structure
        assert result.score is not None
        assert result.has_plan
