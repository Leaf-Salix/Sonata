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
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class GuardSeverity:
    """Classification of guard invalidation severity.
    
    Attributes:
        value: String representation of the severity level.
        soft: Parameter-level change that can be handled by in-place update
            of plan handle. Does not require full replan.
        hard: Structure-level change that requires rebuilding the plan handle
            or full replanning from Score.
    """
    
    value: str
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"GuardSeverity({self.value!r})"
    
    @property
    def requires_replan(self) -> bool:
        """Return True if this severity requires full replanning."""
        return self.value == "hard"


# Pre-defined instances for convenience
GUARD_SEVERITY_SOFT = GuardSeverity("soft")
GUARD_SEVERITY_HARD = GuardSeverity("hard")


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
            "severity": self.severity,
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
    
    dims: tuple[int, ...] = field(default_factory=tuple)
    severity: GuardSeverity = GuardSeverity.HARD
    
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
        if data.get("type") != "ShapeAssumption":
            # Fallback for old format without type field
            return cls(
                symbol=data["symbol"],
                dims=tuple(data.get("dims", [])),
                severity=GuardSeverity(data.get("severity", "hard")),
            )
        
        return cls(
            symbol=data["symbol"],
            dims=tuple(data.get("dims", [])),
            severity=GuardSeverity(data.get("severity", "hard")),
        )
    
    def __repr__(self) -> str:
        return f"ShapeAssumption(symbol={self.symbol!r}, dims={self.dims!r}, severity={self.severity!r})"


# Schema version for guard condition ABI
GUARD_CONDITION_SCHEMA_VERSION = 1

__all__ = [
    "GuardCondition",
    "GuardSeverity",
    "ShapeAssumption",
    "GUARD_CONDITION_SCHEMA_VERSION",
]
