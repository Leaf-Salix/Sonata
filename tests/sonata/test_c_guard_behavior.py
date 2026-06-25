# Copyright (c) PyPTO Contributors.
# ---------------------------------------------------------------------------
"""v0.28 Phase C: Guard-driven behavior change.

Verifies that guard conditions affect Sonata's execution path:
- HARD guard violation → replan
- GuardSeverity correctly serializes/deserializes
"""

import pytest
from sonata.schedule import ScheduleGuard


class TestPhaseCGuardDrivenBehavior:
    """Guard conditions change Sonata's execution decision."""

    def test_hard_guard_marked_as_replan(self):
        """HARD guard → requires_replan = True."""
        guard = ScheduleGuard(kind="hard_shape", severity="hard")
        from sonata.guard import GuardSeverity
        assert guard.severity == GuardSeverity.HARD
        assert guard.severity.requires_replan is True

    def test_soft_guard_no_replan(self):
        """SOFT guard → requires_replan = False."""
        guard = ScheduleGuard(kind="shape_range", severity="soft")
        from sonata.guard import GuardSeverity
        assert guard.severity == GuardSeverity.SOFT
        assert guard.severity.requires_replan is False

    def test_invalidate_action_enum_values(self):
        """InvalidateAction enum covers all expected outcomes."""
        from sonata.guard import InvalidateAction
        assert InvalidateAction.REPLAN.value == "replan"
        assert InvalidateAction.INVALIDATE_HANDLE.value == "invalidate_handle"
        assert InvalidateAction.UPDATE_IN_PLACE.value == "update_in_place"

    def test_guard_severity_from_string(self):
        """GuardSeverity('hard') resolves via _missing_ hook."""
        from sonata.guard import GuardSeverity
        assert GuardSeverity("hard") == GuardSeverity.HARD
        assert GuardSeverity("soft") == GuardSeverity.SOFT

    def test_guard_severity_case_insensitive(self):
        """GuardSeverity accepts case variations via _missing_."""
        from sonata.guard import GuardSeverity
        assert GuardSeverity("HARD") == GuardSeverity.HARD
        assert GuardSeverity("Soft") == GuardSeverity.SOFT

    def test_unknown_severity_defaults_to_hard(self):
        """Unknown severity string defaults to HARD (safe default)."""
        from sonata.guard import GuardSeverity
        assert GuardSeverity("unknown_value") == GuardSeverity.HARD
