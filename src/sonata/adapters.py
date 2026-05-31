# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Multi-adapter registry for PyPTO pipeline stages.

Each adapter represents a specific PyPTO pipeline stage that can produce
normalized facts for Sonata Score construction.  The registry tracks
available adapters, their capabilities, and validates PlanHandle adapter
references.
"""

from dataclasses import dataclass, field
from typing import Any

from .plan_handle import PlanHandle


@dataclass(frozen=True)
class AdapterCapability:
    """Declares what an adapter stage can provide."""

    static_shapes: bool = True
    storage_keys: bool = True
    arg_directions: bool = True
    dependency_kinds: bool = False
    control_flow_regions: bool = False
    runtime_scopes: bool = False


@dataclass(frozen=True)
class AdapterDescriptor:
    """Metadata for one PyPTO pipeline adapter stage."""

    name: str
    version: int = 1
    capabilities: AdapterCapability = field(default_factory=AdapterCapability)
    description: str = ""
    certified_dump: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {
                "static_shapes": self.capabilities.static_shapes,
                "storage_keys": self.capabilities.storage_keys,
                "arg_directions": self.capabilities.arg_directions,
                "dependency_kinds": self.capabilities.dependency_kinds,
                "control_flow_regions": self.capabilities.control_flow_regions,
                "runtime_scopes": self.capabilities.runtime_scopes,
            },
            "description": self.description,
            "certified_dump": self.certified_dump,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterDescriptor":
        caps = data.get("capabilities", {})
        return cls(
            name=data["name"],
            version=data.get("version", 1),
            capabilities=AdapterCapability(
                static_shapes=caps.get("static_shapes", True),
                storage_keys=caps.get("storage_keys", True),
                arg_directions=caps.get("arg_directions", True),
                dependency_kinds=caps.get("dependency_kinds", False),
                control_flow_regions=caps.get("control_flow_regions", False),
                runtime_scopes=caps.get("runtime_scopes", False),
            ),
            description=data.get("description", ""),
            certified_dump=data.get("certified_dump", ""),
        )


POST_SIMPLIFY = AdapterDescriptor(
    name="post_simplify",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
    ),
    description="Post-Simplify adapter after CollectCommGroups, before MaterializeRuntimeScopes.",
    certified_dump="after_collect_comm_groups_simplify",
)

PRE_RUNTIME = AdapterDescriptor(
    name="pre_runtime",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
        dependency_kinds=True,
    ),
    description="Pre-runtime adapter with dependency kind awareness.",
    certified_dump="pre_runtime_scope",
)

POST_SIMPLIFY_WITH_SCOPE = AdapterDescriptor(
    name="post_simplify_with_scope",
    version=1,
    capabilities=AdapterCapability(
        static_shapes=True,
        storage_keys=True,
        arg_directions=True,
        runtime_scopes=True,
    ),
    description="Post-Simplify adapter that also extracts RuntimeScopeStmt facts.",
    certified_dump="after_collect_comm_groups_simplify_with_scope",
)


class AdapterRegistry:
    """Registry of available PyPTO adapter stages."""

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterDescriptor] = {}

    def register(self, adapter: AdapterDescriptor) -> None:
        """Register an adapter. Raises ValueError on duplicate name."""
        if adapter.name in self._adapters:
            raise ValueError(f"adapter already registered: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> AdapterDescriptor | None:
        """Look up an adapter by name. Returns None if not found."""
        return self._adapters.get(name)

    def names(self) -> frozenset[str]:
        """Return all registered adapter names."""
        return frozenset(self._adapters)

    def adapters(self) -> tuple[AdapterDescriptor, ...]:
        """Return all registered adapters."""
        return tuple(self._adapters.values())

    def entry_count(self) -> int:
        """Return the number of registered adapters."""
        return len(self._adapters)

    def select(self, **required_capabilities: bool) -> tuple[AdapterDescriptor, ...]:
        """Return adapters that satisfy all required capability flags.

        Only ``True`` values in ``required_capabilities`` act as filters.
        """
        result = []
        for adapter in self._adapters.values():
            caps = adapter.capabilities
            match = True
            for key, value in required_capabilities.items():
                if value and not getattr(caps, key, False):
                    match = False
                    break
            if match:
                result.append(adapter)
        return tuple(result)

    def validate_plan_handle(self, plan_handle: PlanHandle) -> bool:
        """Return whether the PlanHandle's source_adapter is registered."""
        return plan_handle.source_adapter in self._adapters

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapters": {
                name: adapter.to_dict()
                for name, adapter in sorted(self._adapters.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterRegistry":
        registry = cls()
        for adapter_data in data.get("adapters", {}).values():
            registry.register(AdapterDescriptor.from_dict(adapter_data))
        return registry


def default_registry() -> AdapterRegistry:
    """Return a registry pre-populated with the standard Sonata adapters."""
    registry = AdapterRegistry()
    registry.register(POST_SIMPLIFY)
    registry.register(PRE_RUNTIME)
    registry.register(POST_SIMPLIFY_WITH_SCOPE)
    return registry


__all__ = [
    "AdapterCapability",
    "AdapterDescriptor",
    "AdapterRegistry",
    "POST_SIMPLIFY",
    "POST_SIMPLIFY_WITH_SCOPE",
    "PRE_RUNTIME",
    "default_registry",
]
