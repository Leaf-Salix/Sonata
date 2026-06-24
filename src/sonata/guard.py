# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Guard condition abstraction for Sonata static planning.

This module introduces a unified ``GuardCondition`` abstraction layer that
replaces the simple ``ShapeAssumption`` model from v0.1-v0.9. Key features:

- **Unified interface**: All guard types implement ``GuardCondition`` ABC
- **Severity classification**: Guards are classified as ``soft`` or ``hard``
- **Extensibility**: New guard types (e.g., ``TopologyGuard``, ``StorageGuard``)
  can be added as subclasses without modifying existing code
- **Fingerprint integration**: Guard conditions are included in canonical payload
  for cache key computation

## Usage Example

```python
from sonata.guard import GuardCondition, GuardSeverity, ShapeAssumption

# Create a shape guard (backward compatible)
shape_guard = ShapeAssumption(symbol="batch_size", dims=(32,), severity=GuardSeverity.HARD)

# Create custom guard subclass
class CustomGuard(GuardCondition):
    def __init__(self, symbol: str, value: int, threshold: int):
        super().__init__(symbol=symbol, severity=GuardSeverity.SOFT)
        self.value = value
        self.threshold = threshold
    
    def evaluate(self, runtime_values: dict) -> bool:
        return runtime_values.get(self.symbol, 0) <= self.threshold

custom = CustomGuard("temp_limit", 80, 100)
```

## Design Rationale

See ``reports/v0.10-plan.md`` and ``reports/roadmap_history.md#2026-06-01--v010-guard-condition-abstract-route``
for detailed design decisions and trade-offs.

**Key principles**:
1. Backward compatible: ``ShapeAssumption`` remains usable as-is
2. Conservative default: all guards default to ``HARD`` severity
3. O(1) evaluation: guard checks must be cheap runtime operations
4. Schema versioning: ``GUARD_CONDITION_SCHEMA_VERSION`` tracks ABI changes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Iterable, Optional
import json
import logging
import warnings

_log = logging.getLogger(__name__)


class GuardSeverity(StrEnum):
    """Classification of guard invalidation severity.

    Members:
        SOFT: Parameter-level change that can be handled by in-place update
            of plan handle. Does not require full replan.
        HARD: Structure-level change that requires rebuilding the plan handle
            or full replanning from Score.
    """

    SOFT = "soft"
    HARD = "hard"

    @property
    def requires_replan(self) -> bool:
        """Return True if this severity requires full replanning."""
        return self == self.HARD

    @classmethod
    def _missing_(cls, value):
        """Backward-compatible lookup: accept legacy string values.

        Handles case variations and direct string construction from
        serialized JSON (GuardSeverity("soft") → GuardSeverity.SOFT).
        """
        for member in cls:
            if member.value.lower() == str(value).lower():
                return member
        return cls.HARD  # safe default


# Pre-defined references for backward compat (now just aliases)
GUARD_SEVERITY_SOFT = GuardSeverity.SOFT
GUARD_SEVERITY_HARD = GuardSeverity.HARD


@dataclass(frozen=True)
class GuardCondition(ABC):
    """Abstract base class for all guard conditions in Sonata.
    
    A guard condition represents a runtime validity constraint on a Sonata
    Score or PlanHandle. When the guard condition is violated at runtime,
    the corresponding plan must be invalidated or updated according to
    its severity classification.
    
    Attributes:
        symbol: Human-readable identifier for the guard (e.g., "batch_size", 
            "tensor_shape_dim0"). Used for logging and debugging.
        severity: Classification of invalidation severity. Determines whether
            guard violation triggers in-place update (SOFT) or full replan (HARD).
    
    Subclassing Guide:
        To add a new guard type:
        
        1. Subclass ``GuardCondition``
        2. Implement ``evaluate(runtime_values)`` method
        3. Optionally override ``to_dict()`` / ``from_dict()`` for serialization
        4. Add tests covering construction, evaluation, and edge cases
    
    Example:
        >>> class MyCustomGuard(GuardCondition):
        ...     def __init__(self, symbol: str, min_val: int, max_val: int):
        ...         super().__init__(symbol=symbol, severity=GuardSeverity.SOFT)
        ...         self.min_val = min_val
        ...         self.max_val = max_val
        ...     
        ...     def evaluate(self, runtime_values: dict) -> bool:
        ...         val = runtime_values.get(self.symbol, 0)
        ...         return self.min_val <= val <= self.max_val
    """
    
    symbol: str
    severity: GuardSeverity
    
    @abstractmethod
    def evaluate(self, runtime_values: dict[str, Any]) -> bool:
        """Evaluate whether the guard condition is satisfied at runtime.
        
        Args:
            runtime_values: Dictionary mapping symbol names to runtime values.
                Must contain all symbols referenced by this guard.
        
        Returns:
            True if the guard condition is satisfied, False otherwise.
        
        Performance Requirement:
            This method MUST be O(1) - no expensive computations, no I/O,
            no complex data structure traversals. Guard evaluation happens
            on every cache lookup, so performance is critical.
        """
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize guard condition to dictionary.
        
        Default implementation only serializes base fields (symbol, severity).
        Subclasses should override to include additional fields.
        
        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "symbol": self.symbol,
            "severity": self.severity.value,  # Return string value, not object
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardCondition":
        """Deserialize guard condition from dictionary.
        
        Default implementation creates a generic ``ShapeAssumption`` instance
        when no specific subclass is specified. Subclasses should override
        to provide proper deserialization logic.
        
        Args:
            data: Dictionary from ``to_dict()`` output
        
        Returns:
            Deserialized guard condition instance
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Default fallback to ShapeAssumption for backward compatibility
        from .score import ShapeAssumption
        
        if "symbol" not in data:
            raise ValueError("Guard condition must have 'symbol' field")
        if "severity" not in data:
            raise ValueError("Guard condition must have 'severity' field")
        
        severity = GuardSeverity(data["severity"])
        return ShapeAssumption(symbol=data["symbol"], dims=(), severity=severity)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on symbol and severity."""
        if not isinstance(other, GuardCondition):
            return False
        return self.symbol == other.symbol and self.severity == other.severity
    
    def __hash__(self) -> int:
        """Hash based on symbol and severity."""
        return hash((self.symbol, self.severity))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol!r}, severity={self.severity!r})"


@dataclass(frozen=True)
class ShapeAssumption(GuardCondition):
    """Static shape assumption as a guard condition.
    
    This is the original v0.1-v0.9 guard type, now refactored to inherit
    from ``GuardCondition``. Maintains full backward compatibility while
    enabling future extensions.
    
    Attributes:
        symbol: Variable name being constrained (e.g., "batch_size")
        dims: Tuple of dimension sizes. Empty tuple means scalar.
            Negative dimensions indicate dynamic shapes.
        severity: Defaults to ``GuardSeverity.HARD`` because shape changes
            typically require full replanning.
    
    Example:
        >>> guard = ShapeAssumption(symbol="batch_size", dims=(32, 64))
        >>> guard.evaluate({"batch_size": 32})  # True (dims match)
        >>> guard.evaluate({"batch_size": 64})  # False (dims changed)
    """
    
    symbol: str
    dims: tuple[int, ...] = field(default_factory=tuple)
    severity: GuardSeverity = GUARD_SEVERITY_HARD
    
    _warning_emitted: bool = field(default=False, repr=False, init=False, compare=False, hash=False)
    
    def __post_init__(self):
        """Emit deprecation warning on first instantiation."""
        if not self._warning_emitted:
            object.__setattr__(self, '_warning_emitted', True)
            warnings.warn(
                "ShapeAssumption is deprecated since v0.10 and will be removed in v0.11.\n"
                "\n"
                "Migration guide:\n"
                "  OLD: ShapeAssumption(symbol='N', dims=(128,))\n"
                "  NEW: ShapeAssumption(symbol='N', dims=(128,), severity=GUARD_SEVERITY_HARD)\n"
                "\n"
                "See: https://github.com/hw-native-sys/pypto-sonata/blob/main/docs/user-guide/guards-migration.md\n"
                "for detailed migration instructions.",
                DeprecationWarning,
                stacklevel=2
            )
    
    def evaluate(self, runtime_values: dict[str, Any]) -> bool:
        """Check if runtime tensor shape matches assumed shape.
        
        Args:
            runtime_values: Dictionary containing tensor shapes keyed by symbol.
                Expected format: {"batch_size": (32, 64), ...}
        
        Returns:
            True if runtime shape equals assumed dims, False otherwise.
        """
        runtime_shape = runtime_values.get(self.symbol)
        if runtime_shape is None:
            return False
        
        # Convert to tuple if needed
        if isinstance(runtime_shape, list):
            runtime_shape = tuple(runtime_shape)
        
        return runtime_shape == self.dims
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize ShapeAssumption with dims field."""
        base = super().to_dict()
        base["type"] = "ShapeAssumption"
        base["dims"] = self.dims
        return base
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShapeAssumption":
        """Deserialize ShapeAssumption from dictionary."""
        return cls(
            symbol=data["symbol"],
            dims=tuple(data.get("dims", [])),
            severity=GuardSeverity(data.get("severity", "hard")),
        )
    
    def __repr__(self) -> str:
        return f"ShapeAssumption(symbol={self.symbol!r}, dims={self.dims!r}, severity={self.severity!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapeAssumption):
            return False
        return (self.symbol == other.symbol
                and self.dims == other.dims
                and self.severity == other.severity)

    def __hash__(self) -> int:
        return hash((self.symbol, self.dims, self.severity))


# Schema version for guard condition ABI
GUARD_CONDITION_SCHEMA_VERSION = 1


class InvalidateAction(Enum):
    """Action to take when a guard condition is invalidated.
    
    Attributes:
        REPLAN: Full replanning required. Score and PlanHandle both invalid.
            Triggered by hard guard violations that change computation structure.
        INVALIDATE_HANDLE: Only PlanHandle invalidated, Score remains valid.
            Triggered by soft guard violations that can be handled by rebuilding plan.
        UPDATE_IN_PLACE: Guard violation can be resolved without full rebuild.
            Triggered by parameter-level changes that don't affect structure.
    """
    
    REPLAN = "replan"
    INVALIDATE_HANDLE = "invalidate_handle"
    UPDATE_IN_PLACE = "update_in_place"
    
    def requires_full_replan(self) -> bool:
        """Return True if this action requires full replanning."""
        return self == InvalidateAction.REPLAN
    
    def is_conservative(self) -> bool:
        """Return True for conservative actions (REPLAN or INVALIDATE_HANDLE)."""
        return self in (InvalidateAction.REPLAN, InvalidateAction.INVALIDATE_HANDLE)


class GuardEvaluator:
    """Runtime evaluator for guard conditions.
    
    Evaluates whether runtime values satisfy guard conditions.
    Returns (satisfied: bool, action: InvalidateAction) tuple.
    
    Usage:
        evaluator = GuardEvaluator()
        satisfied, action = evaluator.evaluate(guard, runtime_values)
        if not satisfied:
            if action.requires_full_replan():
                # Trigger full replan from Score extraction
            else:
                # Just invalidate plan handle, keep Score cached
    """
    
    def evaluate(self, guard: GuardCondition, runtime_values: dict[str, Any]) -> tuple[bool, InvalidateAction]:
        """Evaluate a single guard condition against runtime values.
        
        Args:
            guard: Guard condition to evaluate
            runtime_values: Dictionary of runtime values keyed by symbol
            
        Returns:
            Tuple of (satisfied: bool, action: InvalidateAction)
            - If satisfied=True, action indicates what would happen if violated
            - If satisfied=False, action indicates required recovery action
        """
        try:
            satisfied = guard.evaluate(runtime_values)
            action = InvalidateAction.REPLAN if guard.severity.requires_replan else InvalidateAction.INVALIDATE_HANDLE
            return satisfied, action
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            # On evaluation error, treat as violation with conservative action
            _log.debug("[guard] evaluate failed: %s", exc)
            return False, InvalidateAction.REPLAN
    
    def evaluate_all(
        self, 
        guards: Iterable[GuardCondition], 
        runtime_values: dict[str, Any]
    ) -> tuple[bool, list[tuple[GuardCondition, bool, InvalidateAction]]]:
        """Evaluate multiple guard conditions.
        
        Args:
            guards: Iterable of guard conditions to evaluate
            runtime_values: Dictionary of runtime values
            
        Returns:
            Tuple of (all_satisfied: bool, results: list of (guard, satisfied, action))
            - all_satisfied=True means all guards passed
            - results contains detailed per-guard evaluation
        """
        results = []
        all_satisfied = True
        
        for guard in guards:
            satisfied, action = self.evaluate(guard, runtime_values)
            results.append((guard, satisfied, action))
            if not satisfied:
                all_satisfied = False
        
        return all_satisfied, results

    def evaluate_region(
        self,
        node: "RegionTreeNode",
        runtime_values: dict[str, Any],
    ) -> "GuardStatus":
        """Evaluate all guards in a region subtree.

        Collects shape assumptions (which are GuardConditions) from the
        node's Score and all descendant Scores, evaluates them, and returns
        a GuardStatus.  Short-circuits on the first HARD violation.

        Args:
            node: RegionTreeNode to evaluate.
            runtime_values: Runtime values to check guards against.

        Returns:
            GuardStatus.ALL_SATISFIED, PARTIAL_FAILED, or ALL_FAILED.
        """
        from .plan_handle import GuardStatus

        guards: list[GuardCondition] = []

        def _collect(n: "RegionTreeNode") -> None:
            if n.score is not None:
                guards.extend(n.score.shape_assumptions)
            for child in n.children:
                _collect(child)

        _collect(node)

        if not guards:
            return GuardStatus.ALL_SATISFIED

        any_failed = False
        for guard in guards:
            satisfied, action = self.evaluate(guard, runtime_values)
            if not satisfied:
                if action == InvalidateAction.REPLAN:
                    return GuardStatus.ALL_FAILED
                any_failed = True

        return GuardStatus.PARTIAL_FAILED if any_failed else GuardStatus.ALL_SATISFIED

    def select_region_guards(
        self,
        node: "RegionTreeNode",
        selector: "GuardSelector | None" = None,
        *,
        density_threshold: int = 50,
    ) -> list[GuardCondition]:
        """Select top-K guards from a region subtree.

        Collects shape assumptions from the node's Score and all
        descendants, applies the selector (default:
        EntryParamGuardSelector), and warns if guard density exceeds
        ``density_threshold``.

        Args:
            node: Root of the region subtree.
            selector: Guard selection strategy.  Defaults to
                EntryParamGuardSelector (no top_k limit).
            density_threshold: Warn when total guard count exceeds this.

        Returns:
            Selected guard conditions.
        """
        guards: list[GuardCondition] = []

        def _collect(n: "RegionTreeNode") -> None:
            if n.score is not None:
                guards.extend(n.score.shape_assumptions)
            for child in n.children:
                _collect(child)

        _collect(node)

        exceeds, count = check_guard_density(guards, warning_threshold=density_threshold)
        if exceeds:
            warnings.warn(
                f"Region guard density {count} exceeds threshold {density_threshold}. "
                "Consider using GuardSelector.top_k to reduce over-guarding.",
                UserWarning,
                stacklevel=2,
            )

        if selector is None:
            selector = EntryParamGuardSelector()
        return selector.select(guards)


class GuardInvalidator:
    """Strategy class for handling guard invalidation.
    
    Implements different invalidation strategies based on severity:
    - Hard guards → REPLAN (full replanning from Score extraction)
    - Soft guards → INVALIDATE_HANDLE (rebuild PlanHandle only)
    
    Usage:
        invalidator = GuardInvalidator()
        action = invalidator.invalidate(guard, reason="shape mismatch")
        if action == InvalidateAction.REPLAN:
            # Trigger full replan
    """
    
    def __init__(self, default_action: InvalidateAction = InvalidateAction.REPLAN):
        """Initialize GuardInvalidator.
        
        Args:
            default_action: Default action for unknown guard types
        """
        self.default_action = default_action
    
    def invalidate(self, guard: GuardCondition, reason: str = "") -> InvalidateAction:
        """Determine invalidation action for a guard violation.
        
        Args:
            guard: Guard condition that was violated
            reason: Human-readable reason for violation
            
        Returns:
            InvalidateAction indicating required recovery strategy
        """
        # Use severity-based policy
        if guard.severity.requires_replan:  # property, not method
            return InvalidateAction.REPLAN
        else:
            return InvalidateAction.INVALIDATE_HANDLE
    
    def invalidate_all(
        self, 
        results: list[tuple[GuardCondition, bool, InvalidateAction]]
    ) -> InvalidateAction:
        """Determine overall action from multiple guard evaluations.
        
        Policy: If any hard guard violated → REPLAN, else INVALIDATE_HANDLE.
        
        Args:
            results: List of (guard, satisfied, action) tuples from GuardEvaluator
            
        Returns:
            Conservative action covering all violations
        """
        has_hard_violation = False
        
        for guard, satisfied, action in results:
            if not satisfied:
                if guard.severity.requires_replan:  # property, not method
                    has_hard_violation = True
                    break
        
        return InvalidateAction.REPLAN if has_hard_violation else InvalidateAction.INVALIDATE_HANDLE


# ============================================================================
# Phase 3: Over-guarding Mitigation (C1-C5)
# ============================================================================

class GuardSelector(ABC):
    """Selects a subset of guards to minimize over-guarding.
    
    Over-guarding occurs when too many guard conditions make fingerprints
    overly fragile (e.g., 100 shape assumptions for a 100-parameter graph).
    This interface allows selecting the most important Top-K guards.
    
    Usage:
        selector = EntryParamGuardSelector(top_k=10)
        critical_guards = selector.select(guards, runtime_info)
    """
    
    @abstractmethod
    def select(
        self, 
        guards: Iterable[GuardCondition],
        runtime_info: dict[str, Any] | None = None
    ) -> list[GuardCondition]:
        """Select the most important guards from a set.
        
        Args:
            guards: All available guard conditions
            runtime_info: Optional runtime context (e.g., call frequency, 
                parameter usage patterns)
        
        Returns:
            Subset of guards deemed most important (Top-K)
        """
        pass


@dataclass(frozen=True)
class EntryParamGuardSelector(GuardSelector):
    """Default strategy: prioritize entry function parameters.
    
    Rationale: Entry function parameters (graph inputs) are more likely
    to vary across invocations than intermediate tensors. Prioritizing
    them provides better cache hit rates while maintaining safety.
    
    Attributes:
        top_k: Maximum number of guards to select. If None, select all.
        priority_symbols: Optional list of symbol names to prioritize.
            These will be selected before others.
    
    Usage:
        selector = EntryParamGuardSelector(top_k=10, priority_symbols=["batch_size", "seq_len"])
        critical_guards = selector.select(all_guards)
    """
    
    top_k: int | None = None
    priority_symbols: tuple[str, ...] = field(default_factory=tuple)
    
    def select(
        self, 
        guards: Iterable[GuardCondition],
        runtime_info: dict[str, Any] | None = None
    ) -> list[GuardCondition]:
        """Select guards by priority order: entry params → others.
        
        Args:
            guards: All available guard conditions
            runtime_info: Ignored in current implementation (reserved for future)
        
        Returns:
            Top-K guards based on priority
        """
        guards_list = list(guards)
        
        # First, select priority symbols
        selected = [g for g in guards_list if g.symbol in self.priority_symbols]
        
        # Then, add remaining guards in original order
        remaining = [g for g in guards_list if g.symbol not in self.priority_symbols]
        selected.extend(remaining)
        
        # Apply top-k limit
        if self.top_k is not None and len(selected) > self.top_k:
            selected = selected[:self.top_k]
        
        return selected


def check_guard_density(
    guards: Iterable[GuardCondition],
    warning_threshold: int = 50,
    logger: Optional[callable] = None
) -> tuple[bool, int]:
    """Check if guard density exceeds warning threshold.
    
    Used to detect potential over-guarding scenarios.
    
    Args:
        guards: All guard conditions
        warning_threshold: Number above which to emit warning (default 50)
        logger: Optional logging function(warning_threshold, message)
    
    Returns:
        Tuple of (exceeds_threshold: bool, count: int)
    """
    count = sum(1 for _ in guards)
    exceeds = count > warning_threshold
    
    if exceeds and logger:
        logger(
            warning_threshold,
            f"Guard density {count} exceeds threshold {warning_threshold}. "
            "Consider using GuardSelector to reduce over-guarding."
        )
    
    return exceeds, count


# ============================================================================
# Migration Utilities (Phase 6 F1-F2)
# ============================================================================

def shape_assumption_to_guard_condition(
    sa: "ShapeAssumption",
    severity: GuardSeverity | None = None
) -> GuardCondition:
    """Convert ShapeAssumption to GuardCondition.
    
    Since ShapeAssumption is already a GuardCondition subclass, this function
    primarily serves as a migration aid with deprecation warnings.
    
    Args:
        sa: ShapeAssumption instance to convert
        severity: Override severity (default: use sa.severity)
    
    Returns:
        GuardCondition subclass (the ShapeAssumption instance itself, or new instance if severity overridden)
    
    Raises:
        TypeError: If sa is not a ShapeAssumption
        DeprecationWarning: Always emitted to encourage migration to direct GuardCondition usage
    """
    import warnings
    from .score import ShapeAssumption as SA_Type
    
    if not isinstance(sa, SA_Type):
        raise TypeError(f"Expected ShapeAssumption, got {type(sa).__name__}")
    
    # Emit deprecation warning
    warnings.warn(
        "ShapeAssumption is deprecated since v0.10 and will be removed in v0.11. "
        "Use GuardCondition subclasses directly. "
        "This converter is provided for backward compatibility only.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # If severity is overridden, create a new instance with the new severity
    if severity is not None and sa.severity != severity:
        return SA_Type(symbol=sa.symbol, dims=sa.dims, severity=severity)
    
    # Return the ShapeAssumption as-is (it's already a GuardCondition)
    return sa


def deprecated_shape_assumption(
    symbol: str,
    dims: tuple[int, ...] = (),
    *,
    severity: GuardSeverity = GUARD_SEVERITY_HARD
) -> "ShapeAssumption":
    """Create a deprecated ShapeAssumption with clear deprecation warning.
    
    This helper function provides a clear migration path by emitting a
    detailed deprecation warning when users instantiate ShapeAssumption.
    
    Args:
        symbol: The symbol name (e.g., "batch_size")
        dims: The dimension values (e.g., (32, 64))
        severity: Guard severity (default: HARD)
    
    Returns:
        ShapeAssumption instance (with deprecation warning)
    
    Example:
        >>> import warnings
        >>> warnings.simplefilter("always")
        >>> sa = deprecated_shape_assumption("N", (128,))
        DeprecationWarning: ShapeAssumption is deprecated...
    """
    import warnings
    from .score import ShapeAssumption as SA_Type
    
    warnings.warn(
        "ShapeAssumption is deprecated since v0.10 and will be removed in v0.11.\n"
        "\n"
        "Migration guide:\n"
        "  OLD: ShapeAssumption(symbol='N', dims=(128,))\n"
        "  NEW: ShapeAssumption(symbol='N', dims=(128,), severity=GUARD_SEVERITY_HARD)\n"
        "\n"
        "See: https://github.com/hw-native-sys/pypto-sonata/blob/main/docs/user-guide/guards-migration.md\n"
        "for detailed migration instructions.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return SA_Type(symbol=symbol, dims=dims, severity=severity)


__all__ = [
    "GuardCondition",
    "GuardSeverity",
    "GUARD_SEVERITY_SOFT",
    "GUARD_SEVERITY_HARD",
    "ShapeAssumption",
    "GUARD_CONDITION_SCHEMA_VERSION",
    "InvalidateAction",
    "GuardEvaluator",
    "GuardInvalidator",
    "GuardSelector",
    "EntryParamGuardSelector",
    "check_guard_density",
    "shape_assumption_to_guard_condition",
    "deprecated_shape_assumption",
]
