# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
from sonata import check_static_eligibility
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
    def test_legacy_message_lookup_returns_none(self) -> None:
        assert code_for_reason("ForStmt is not supported by initial Sonata eligibility") is None
        assert code_for_reason("RuntimeScopeStmt is not supported by initial Sonata eligibility") is None
        assert code_for_reason("") is None


class TestRejectUsesEnumCode:
    def test_reject_uses_slug_for_raw_string_message(self) -> None:
        from sonata.score import EligibilityResult

        result = EligibilityResult.reject("ForStmt is not supported by initial Sonata eligibility")

        assert result.reason_details[0].code == "forstmt_is_not_supported_by_initial_sonata_eligibility"

    def test_reject_preserves_explicit_reason_code_when_message_changes(self) -> None:
        from sonata.score import EligibilityResult, FallbackReason

        result = EligibilityResult.reject(
            FallbackReason(
                code=FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION.value,
                message="entry function main is not an orchestration function",
            )
        )

        assert result.reasons == ("entry function main is not an orchestration function",)
        assert result.reason_details[0].code == "entry_function_not_orchestration"

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

    def test_accept_with_warnings_uses_slug_for_raw_string_message(self) -> None:
        from sonata.score import EligibilityResult, Score, RuntimeTarget

        score = Score(name="valid", runtime_target=RuntimeTarget())
        result = EligibilityResult.accept_with_warnings(
            score, "ForStmt is not supported by initial Sonata eligibility"
        )

        assert result.eligible
        assert result.reason_details[0].code == "forstmt_is_not_supported_by_initial_sonata_eligibility"
        assert result.reason_details[0].severity == "warning"


class TestEligibilityEmitsExplicitCodes:
    def test_unsupported_root_uses_stable_code(self) -> None:
        result = check_static_eligibility(object())

        assert result.reason_details[0].code == FallbackCode.UNSUPPORTED_ROOT_KIND.value

    def test_runtime_scope_uses_runtime_scope_code(self) -> None:
        from tests.sonata.test_eligibility import Call, EvalStmt, Function, RuntimeScopeStmt

        result = check_static_eligibility(
            Function(name="main", body=RuntimeScopeStmt(body=(EvalStmt(Call("kernel.add")),)))
        )

        assert result.reason_details[0].code == FallbackCode.UNSUPPORTED_RUNTIME_SCOPE.value

    def test_control_flow_uses_control_flow_code(self) -> None:
        from tests.sonata.test_eligibility import Call, EvalStmt, ForStmt, Function

        result = check_static_eligibility(
            Function(name="main", body=(ForStmt(body=(EvalStmt(Call("kernel.add")),)),))
        )

        assert result.reason_details[0].code == FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value

    def test_tensor_read_uses_tensor_read_code(self) -> None:
        from tests.sonata.test_eligibility import Call, EvalStmt, Function

        result = check_static_eligibility(Function(name="main", body=(EvalStmt(Call("tensor.read")),)))

        assert result.reason_details[0].code == FallbackCode.TENSOR_READ_NOT_SUPPORTED.value

    def test_entry_mismatch_uses_entry_code(self) -> None:
        from tests.sonata.test_eligibility import FuncType, Function, Program

        helper = Function(name="helper", body=())
        helper.func_type = FuncType("AIC")

        result = check_static_eligibility(Program(functions={"helper": helper}), entry_name="helper")

        assert result.reason_details[0].code == FallbackCode.ENTRY_FUNCTION_NOT_ORCHESTRATION.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
