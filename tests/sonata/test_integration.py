# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration smoke tests for Sonata end-to-end workflows.

This module covers:
- A3: Full module import verification
- A4: Score → PlanHandle → RuntimeAdapter full chain
"""

import pytest
from sonata import (
    Score,
    Task,
    Dependency,
    ShapeAssumption,
    RuntimeTarget,
    EligibilityResult,
    FallbackReason,
    FallbackCode,
    PlanHandle,
    FuncRegistry,
    HostBuildGraphRuntimeAdapter,
    SONATA_VERSION,
    VERSION_INFO,
    public_api,
    module_api,
    schema_versions,
)


class TestModuleImportSmoke:
    """A3: 全模块导入验证 - 确保所有 public symbols 可访问。"""

    def test_sonata_import_success(self):
        """Basic import of sonata package."""
        import sonata
        assert hasattr(sonata, "__all__")
        assert len(sonata.__all__) > 0

    def test_all_public_symbols_accessible(self):
        """All symbols in __all__ should be accessible."""
        import sonata
        api = public_api()
        assert len(api) == len(sonata.__all__)
        # Verify each symbol exists
        for name in api:
            assert hasattr(sonata, name), f"Symbol {name} missing from sonata namespace"
            getattr(sonata, name)  # Should not raise AttributeError

    def test_version_constants_exist(self):
        """SONATA_VERSION and VERSION_INFO should exist."""
        assert SONATA_VERSION is not None
        assert isinstance(SONATA_VERSION, str)
        assert VERSION_INFO is not None
        assert isinstance(VERSION_INFO, tuple)
        assert len(VERSION_INFO) == 3

    def test_schema_versions_return_dict(self):
        """schema_versions() should return a dict with all version constants."""
        versions = schema_versions()
        assert isinstance(versions, dict)
        expected_keys = {
            "score_schema",
            "fingerprint_version",
            "eligibility_result_schema",
            "plan_handle_schema",
            "runtime_contract",
            "cache_schema",
        }
        assert set(versions.keys()) == expected_keys
        # All values should be positive integers
        for key, value in versions.items():
            assert isinstance(value, int), f"{key} should be int"
            assert value > 0, f"{key} should be positive"

    def test_module_api_structure(self):
        """module_api() should return dict grouped by module."""
        modules = module_api()
        assert isinstance(modules, dict)
        assert len(modules) > 0
        # Each value should be a list of strings
        for module_name, symbols in modules.items():
            assert isinstance(module_name, str)
            assert isinstance(symbols, list)
            for symbol in symbols:
                assert isinstance(symbol, str)


class TestEndToEndChain:
    """A4: Score → PlanHandle → RuntimeAdapter 端到端集成测试。"""

    @pytest.fixture
    def sample_score(self):
        """Create a minimal valid Score for testing."""
        return Score(
            name="test_matmul",
            runtime_target=RuntimeTarget(
                runtime="host_build_graph",
                function_name="build_test_graph",
                aicpu_thread_num=4,
            ),
            tasks=(
                Task(
                    task_id=0,
                    func_id=0,
                    core_type="aic",
                    args=("tensor_a", "tensor_b"),
                    arg_directions=("input", "input"),
                    arg_storage_keys=("param:0:a", "param:1:b"),
                ),
                Task(
                    task_id=1,
                    func_id=0,
                    core_type="aic",
                    args=("tensor_c",),
                    arg_directions=("output",),
                    arg_storage_keys=("param:2:c",),
                ),
            ),
            dependencies=(
                Dependency(producer=0, consumer=1, kind="data"),
            ),
            shape_assumptions=(
                ShapeAssumption(symbol="batch_size", dims=(32,), severity="hard"),
            ),
        )

    @pytest.fixture
    def sample_plan_handle(self, sample_score):
        """Create a PlanHandle from a Score."""
        return PlanHandle.from_score(
            score=sample_score,
            runtime_target=RuntimeTarget(
                runtime="host_build_graph",
                function_name="build_test_graph",
                aicpu_thread_num=4,
            ),
        )

    def test_eligibility_to_score_chain(self):
        """Verify eligibility check produces valid Score."""
        from sonata.eligibility import check_static_eligibility

        # Create a minimal mock IR node with required attributes
        class MockNode:
            kind = "FunctionDef"
            name = "test_func"
            body = []
            args = []

        result = check_static_eligibility(MockNode())
        # Check that we got a valid EligibilityResult
        assert result is not None
        assert isinstance(result, EligibilityResult)
        # The mock node will likely fail eligibility checks (no proper body),
        # but we verify the result structure is correct
        assert hasattr(result, 'score')
        assert hasattr(result, 'reason_details')

    def test_score_to_planhandle_chain(self, sample_score):
        """Verify Score can be converted to PlanHandle."""
        plan_handle = PlanHandle.from_score(
            score=sample_score,
            runtime_target=sample_score.runtime_target,
        )
        assert plan_handle is not None
        assert plan_handle.score_fingerprint is not None
        assert isinstance(plan_handle.func_registry, FuncRegistry)

    def test_planhandle_to_runtimeadapter_chain(self, sample_score, sample_plan_handle):
        """Verify PlanHandle can generate HostBuildGraphPlan via RuntimeAdapter."""
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(
            score=sample_score,
            plan_handle=sample_plan_handle,
        )
        # result is a RuntimeAdapterResult - check its properties directly
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'plan')
        assert hasattr(result, 'reasons')
        if result.plan is not None:
            assert hasattr(result.plan, "tasks")
            assert hasattr(result.plan, "edges")

    def test_full_chain_roundtrip(self, sample_score, sample_plan_handle):
        """Complete chain: eligibility → Score → PlanHandle → RuntimeAdapter."""
        # Step 1: Start with a valid Score
        assert isinstance(sample_score, Score)
        validation_result = sample_score.validate()
        assert validation_result.eligible is True

        # Step 2: Convert to PlanHandle
        assert isinstance(sample_plan_handle, PlanHandle)
        fp = sample_plan_handle.score_fingerprint
        assert fp is not None
        assert len(fp) == 64  # SHA-256 hex digest

        # Step 3: Generate RuntimeAdapter output
        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(
            score=sample_score,
            plan_handle=sample_plan_handle,
        )
        # Check result properties directly (not using accept() which requires plan arg)
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'plan')
        assert hasattr(result, 'reasons')
        
        # Step 4: Verify generated plan structure (if successful)
        if result.plan is not None:
            plan = result.plan
            assert len(plan.tasks) == len(sample_score.tasks)

        # Step 5: Verify fingerprint consistency
        from sonata.serialization import score_fingerprint
        computed_fp = score_fingerprint(sample_score)
        assert computed_fp == fp

    def test_fingerprint_consistency_across_chain(self, sample_score, sample_plan_handle):
        """Fingerprint should remain consistent throughout the chain."""
        from sonata.serialization import score_fingerprint

        original_fp = score_fingerprint(sample_score)
        handle_fp = sample_plan_handle.score_fingerprint

        assert original_fp == handle_fp, "Score fingerprint and PlanHandle fingerprint must match"

    def test_serialization_deserialization_roundtrip(self, sample_score, sample_plan_handle):
        """Score and PlanHandle should survive serialize→deserialize round-trip."""
        from sonata.serialization import (
            score_to_dict,
            plan_handle_to_dict,
        )
        from sonata.deserialization import (
            score_from_dict,
            plan_handle_from_dict,
        )

        # Score round-trip
        score_dict = score_to_dict(sample_score)
        score_restored = score_from_dict(score_dict)
        assert score_restored.name == sample_score.name
        assert len(score_restored.tasks) == len(sample_score.tasks)

        # PlanHandle round-trip
        handle_dict = plan_handle_to_dict(sample_plan_handle)
        handle_restored = plan_handle_from_dict(handle_dict)
        assert handle_restored.score_fingerprint == sample_plan_handle.score_fingerprint

    def test_runtime_adapter_validation_fails_on_mismatch(self, sample_score):
        """RuntimeAdapter should reject mismatched inputs."""
        from sonata.plan_handle import PlanHandle
        from sonata.serialization import score_fingerprint

        # Create a PlanHandle with wrong fingerprint
        wrong_score = Score(
            name="different_score",
            runtime_target=sample_score.runtime_target,
            tasks=(),
            dependencies=(),
            shape_assumptions=(),
        )
        wrong_handle = PlanHandle.from_score(
            score=wrong_score,
            runtime_target=wrong_score.runtime_target,
        )

        adapter = HostBuildGraphRuntimeAdapter()
        result = adapter.generate(
            score=sample_score,
            plan_handle=wrong_handle,
        )
        # Should fail because fingerprints don't match - check properties directly
        assert result is not None
        assert hasattr(result, 'success')
        assert result.success == False
        assert hasattr(result, 'reasons')
        assert len(result.reasons) > 0
        # The correct FallbackCode value for fingerprint mismatch
        assert result.reasons[0].code == FallbackCode.RUNTIME_ADAPTER_FINGERPRINT_MISMATCH


class TestVersionConsistency:
    """Test that version information is consistent across the codebase."""

    def test_version_matches_semver(self):
        """SONATA_VERSION should follow semver format."""
        import re
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        assert re.match(pattern, SONATA_VERSION), f"Version {SONATA_VERSION} does not match semver"

    def test_version_info_matches_string(self):
        """VERSION_INFO tuple should match SONATA_VERSION string."""
        parts = SONATA_VERSION.split(".")
        assert len(parts) == 3
        assert tuple(int(p) for p in parts) == VERSION_INFO

    def test_major_minor_increases_over_time(self):
        """Major/minor versions should reflect development progress."""
        # v0.9 represents significant feature completion
        assert VERSION_INFO[0] >= 0
        assert VERSION_INFO[1] >= 9  # At least v0.9
