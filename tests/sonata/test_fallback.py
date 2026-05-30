# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from sonata.fallback import FallbackCode, code_for_reason


class TestFallbackCodeEnum:
    def test_enum_values_are_stable_strings(self) -> None:
        assert FallbackCode.UNSUPPORTED_ROOT_KIND == "unsupported_root_kind"
        assert FallbackCode.CONTROL_FLOW_NOT_SUPPORTED == "control_flow_not_supported"
        assert FallbackCode.UNSUPPORTED_RUNTIME_SCOPE == "unsupported_runtime_scope"
        assert FallbackCode.TENSOR_READ_NOT_SUPPORTED == "tensor_read_not_supported"
        assert FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION == "entry_function_not_orchestration"
        assert FallbackCode.SCORE_VALIDATION_FAILED == "score_validation_failed"

    def test_enum_members_are_all_str(self) -> None:
        for member in FallbackCode:
            assert isinstance(member.value, str)


class TestCodeForReason:
    def test_control_flow_for_stmt(self) -> None:
        assert code_for_reason("ForStmt is not supported by initial Sonata eligibility") == FallbackCode.CONTROL_FLOW_NOT_SUPPORTED

    def test_control_flow_if_stmt(self) -> None:
        assert code_for_reason("IfStmt is not supported by initial Sonata eligibility") == FallbackCode.CONTROL_FLOW_NOT_SUPPORTED

    def test_control_flow_while_stmt(self) -> None:
        assert code_for_reason("WhileStmt is not supported by initial Sonata eligibility") == FallbackCode.CONTROL_FLOW_NOT_SUPPORTED

    def test_runtime_scope_stmt(self) -> None:
        assert code_for_reason("RuntimeScopeStmt is not supported by initial Sonata eligibility") == FallbackCode.UNSUPPORTED_RUNTIME_SCOPE

    def test_unsupported_root(self) -> None:
        assert code_for_reason("unsupported root for Sonata eligibility: Module") == FallbackCode.UNSUPPORTED_ROOT_KIND

    def test_tensor_read(self) -> None:
        assert code_for_reason("tensor.read calls are not supported by initial Sonata eligibility") == FallbackCode.TENSOR_READ_NOT_SUPPORTED

    def test_entry_not_orchestration(self) -> None:
        assert code_for_reason("entry function is not an orchestration function: main") == FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION

    def test_unknown_message_returns_none(self) -> None:
        assert code_for_reason("some completely unknown error") is None

    def test_empty_message_returns_none(self) -> None:
        assert code_for_reason("") is None


class TestRejectUsesEnumCode:
    def test_reject_maps_known_message_to_enum_code(self) -> None:
        from sonata.score import EligibilityResult

        result = EligibilityResult.reject("ForStmt is not supported by initial Sonata eligibility")

        assert result.reason_details[0].code == "control_flow_not_supported"

    def test_reject_falls_back_to_slug_for_unknown_message(self) -> None:
        from sonata.score import EligibilityResult

        result = EligibilityResult.reject("some novel error")

        assert result.reason_details[0].code == "some_novel_error"

    def test_validation_failure_uses_score_validation_failed(self) -> None:
        from sonata.score import Score, RuntimeTarget

        score = Score(name="", runtime_target=RuntimeTarget())
        result = score.validate()

        assert not result.eligible
        assert result.reason_details[0].code == "score_validation_failed"

    def test_accept_with_warnings_uses_enum_code_for_known_mapping(self) -> None:
        from sonata.score import EligibilityResult, Score, RuntimeTarget

        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = EligibilityResult.accept_with_warnings(
            score, "ForStmt is not supported by initial Sonata eligibility"
        )

        assert result.eligible
        assert result.reason_details[0].code == "control_flow_not_supported"
        assert result.reason_details[0].severity == "warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
