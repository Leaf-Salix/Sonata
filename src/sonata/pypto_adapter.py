# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Certified PyPTO IR facts used by Sonata eligibility.

The adapter intentionally uses structural Python-visible fields instead of
importing PyPTO IR classes. Its job is to project PyPTO-like IR into the small
set of facts Sonata v0.1 needs, keeping pass-stage details out of core Score
construction.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CERTIFIED_DUMP_AFTER_COLLECT_COMM_GROUPS_SIMPLIFY = "after_collect_comm_groups_simplify"
DEFAULT_CERTIFIED_DUMP = CERTIFIED_DUMP_AFTER_COLLECT_COMM_GROUPS_SIMPLIFY


class PyPTOAdapterContractError(ValueError):
    """Raised when a PyPTO adapter is asked to normalize an out-of-scope IR."""


@dataclass(frozen=True)
class NormalizedCallFact:
    """One ordinary orchestration call observed by a PyPTO input adapter."""

    node: Any
    callee_name: str
    args: tuple[Any, ...]
    arg_names: tuple[str, ...]
    arg_directions: tuple[str, ...]
    arg_storage_keys: tuple[Any | None, ...]
    core_type: str


@dataclass(frozen=True)
class NormalizedDependencyFact:
    """Reserved extension point for future dependency provenance."""

    producer: int
    consumer: int
    source: str | None = None
    hazard: str | None = None
    provenance: Any | None = None


@dataclass(frozen=True)
class NormalizedFunctionFact:
    """Facts for one selected orchestration root."""

    node: Any
    name: str | None
    is_orchestration: bool
    params: tuple[Any, ...]
    calls: tuple[NormalizedCallFact, ...]


@dataclass(frozen=True)
class NormalizedTaskFacts:
    """Minimal Sonata-facing facts extracted from a PyPTO-like input."""

    source: Any
    certified_dump: str
    functions: tuple[NormalizedFunctionFact, ...]
    dependency_facts: tuple[NormalizedDependencyFact, ...] = ()


class PostSimplifyPyPTOInputAdapter:
    """Project a certified post-Simplify PyPTO shape into Sonata facts.

    This adapter operates on the named certified dump produced after
    CollectCommGroups and the final Simplify pass, before
    MaterializeRuntimeScopes.  If a different stage boundary is needed,
    create a separate adapter class rather than generalizing this one.
    """

    certified_dump = DEFAULT_CERTIFIED_DUMP
    control_flow_kinds = frozenset({"ForStmt", "IfStmt", "WhileStmt"})
    unsupported_kinds = frozenset({"RuntimeScopeStmt", "Submit", "SpmdScopeStmt", "ManualScopeStmt"})
    unsupported_function_types = frozenset({"Group", "Spmd"})
    builtin_op_prefixes = ("tile.", "tensor.", "system.", "array.")

    def __init__(self, node: Any, *, entry_name: str | None = None):
        self.node = node
        self.entry_name = entry_name

    def normalize(
        self,
        storage_key_resolver: Any | None = None,
        *,
        require_certified: bool = False,
    ) -> NormalizedTaskFacts:
        """Return normalized facts for the selected orchestration roots."""
        errors = self.certified_contract_errors() if require_certified else self.out_of_scope_errors()
        if errors:
            raise PyPTOAdapterContractError("; ".join(errors))

        core_types = self.function_core_types()
        functions = tuple(
            NormalizedFunctionFact(
                node=root,
                name=self.function_name(root),
                is_orchestration=True,
                params=tuple(getattr(root, "params", ())),
                calls=self._extract_calls(root, core_types, storage_key_resolver),
            )
            for root in self.extraction_roots()
        )
        return NormalizedTaskFacts(
            source=self.node,
            certified_dump=self.certified_dump,
            functions=functions,
        )

    def extraction_roots(self) -> tuple[Any, ...]:
        """Return selected orchestration roots."""
        functions = getattr(self.node, "functions", None)
        if not isinstance(functions, dict):
            if self.entry_name is not None and getattr(self.node, "name", None) != self.entry_name:
                return ()
            func_type_name = self.function_type_name(self.node)
            if func_type_name is not None and func_type_name != "Orchestration":
                return ()
            return (self.node,)

        roots: list[Any] = []
        for func in functions.values():
            if self.is_orchestration(func) and (
                self.entry_name is None or getattr(func, "name", None) == self.entry_name
            ):
                roots.append(func)
        return tuple(roots)

    def is_orchestration(self, func: Any) -> bool:
        """Return whether ``func`` is an Orchestration function."""
        return self.function_type_name(func) == "Orchestration"

    def function_core_types(self) -> dict[str, str]:
        """Return callee name -> Sonata core type for known functions."""
        functions = getattr(self.node, "functions", None)
        if not isinstance(functions, dict):
            return {}

        core_types: dict[str, str] = {}
        for func in functions.values():
            name = self.function_name(func)
            if name is not None:
                core_types[name] = self.core_type_from_function(func)
        return core_types

    def _extract_calls(
        self,
        root: Any,
        core_types: dict[str, str],
        storage_key_resolver: Any | None,
    ) -> tuple[NormalizedCallFact, ...]:
        calls: list[NormalizedCallFact] = []
        for child in self.walk(root):
            if self.kind(child) != "Call":
                continue
            call_name = self.call_name(child)
            if call_name is None or self.is_builtin_call(call_name):
                continue
            args = tuple(getattr(child, "args", ()))
            calls.append(
                NormalizedCallFact(
                    node=child,
                    callee_name=call_name,
                    args=args,
                    arg_names=tuple(self.arg_name(arg) for arg in args),
                    arg_directions=self.arg_directions(child),
                    arg_storage_keys=storage_key_resolver(child) if storage_key_resolver is not None else (),
                    core_type=core_types.get(call_name, "mixed"),
                )
            )
        return tuple(calls)

    def has_unsupported_function_call(self) -> str | None:
        """Return the first Group/Spmd callee name called by selected roots."""
        functions = getattr(self.node, "functions", None)
        if not isinstance(functions, dict):
            return None

        unsupported: set[str] = set()
        for func in functions.values():
            name = self.function_name(func)
            if name is not None and self.function_type_name(func) in self.unsupported_function_types:
                unsupported.add(name)
        for root in self.extraction_roots():
            for child in self.walk(root):
                if self.kind(child) != "Call":
                    continue
                call_name = self.call_name(child)
                if call_name in unsupported:
                    return call_name
        return None

    def root_out_of_scope_error(self) -> str | None:
        """Return an out-of-scope reason for a direct non-orchestration function root."""
        functions = getattr(self.node, "functions", None)
        if isinstance(functions, dict):
            return None
        func_type_name = self.function_type_name(self.node)
        if func_type_name is None or func_type_name == "Orchestration":
            return None
        name = self.function_name(self.node) or "<anonymous>"
        if func_type_name in self.unsupported_function_types:
            return f"{func_type_name} function root is out of scope for Sonata v0.1 PyPTO adapter: {name}"
        return f"{func_type_name} function root is not an Orchestration function: {name}"

    def out_of_scope_errors(self) -> tuple[str, ...]:
        """Return v0.1 adapter scope violations that must not normalize silently."""
        errors: list[str] = []
        root_error = self.root_out_of_scope_error()
        if root_error is not None:
            errors.append(root_error)
        unsupported_call = self.has_unsupported_function_call()
        if unsupported_call is not None:
            errors.append(f"Group/Spmd callee is out of scope for Sonata v0.1: {unsupported_call}")
        for root in self.extraction_roots():
            for child in self.walk(root):
                child_kind = self.kind(child)
                if child_kind in self.unsupported_kinds:
                    errors.append(f"{child_kind} is out of scope for Sonata v0.1 PyPTO adapter")
        return tuple(dict.fromkeys(errors))

    def certified_contract_errors(self) -> tuple[str, ...]:
        """Return contract violations for the named certified PyPTO dump."""
        errors = list(self.out_of_scope_errors())
        for root in self.extraction_roots():
            for call in self._extract_calls(root, self.function_core_types(), None):
                if not call.arg_directions:
                    errors.append(f"Call {call.callee_name} has no arg_directions in certified dump")
                elif len(call.arg_directions) != len(call.args):
                    errors.append(
                        f"Call {call.callee_name} arg_directions size {len(call.arg_directions)} "
                        f"does not match args size {len(call.args)}"
                    )
        return tuple(dict.fromkeys(errors))

    @classmethod
    def walk(cls, node: Any) -> Iterable[Any]:
        """Yield ``node`` and recursively walk common IR-like child fields."""
        seen: dict[int, Any] = {}
        stack = [node]
        child_fields = (
            "functions",
            "body",
            "then_body",
            "else_body",
            "branches",
            "stmts",
            "statements",
            "seq",
            "args",
            "value",
            "expr",
            "condition",
        )

        while stack:
            current = stack.pop()
            if current is None or isinstance(current, (str, bytes, int, float, bool)):
                continue
            ident = id(current)
            if ident in seen:
                continue
            seen[ident] = current
            yield current

            if isinstance(current, dict):
                stack.extend(current.values())
                continue
            if isinstance(current, (list, tuple)):
                stack.extend(reversed(current))
                continue

            for field in child_fields:
                if hasattr(current, field):
                    stack.append(getattr(current, field))

    @staticmethod
    def kind(node: Any) -> str:
        return type(node).__name__

    @staticmethod
    def function_name(func: Any) -> str | None:
        name = getattr(func, "name", None)
        return name if isinstance(name, str) else None

    @staticmethod
    def function_type_name(func: Any) -> str | None:
        func_type = getattr(func, "func_type", None)
        name = getattr(func_type, "name", None)
        return name if isinstance(name, str) else None

    @classmethod
    def core_type_from_function(cls, func: Any) -> str:
        func_type_name = cls.function_type_name(func)
        if func_type_name == "AIC":
            return "aic"
        if func_type_name == "AIV":
            return "aiv"
        return "mixed"

    @staticmethod
    def call_name(node: Any) -> str | None:
        op = getattr(node, "op", None)
        if isinstance(op, str):
            return op
        name = getattr(op, "name", None)
        if isinstance(name, str):
            return name
        op_name = getattr(node, "op_name", None)
        if isinstance(op_name, str):
            return op_name
        return None

    @classmethod
    def is_builtin_call(cls, call_name: str) -> bool:
        return call_name.startswith(cls.builtin_op_prefixes)

    @classmethod
    def arg_name(cls, node: Any) -> str:
        if isinstance(node, str):
            return node
        name_hint = getattr(node, "name_hint", None)
        if isinstance(name_hint, str):
            return name_hint
        name = getattr(node, "name", None)
        if isinstance(name, str):
            return name
        return cls.kind(node)

    @classmethod
    def arg_directions(cls, call: Any) -> tuple[str, ...]:
        directions = getattr(call, "arg_directions", None)
        if directions is None:
            attrs = getattr(call, "attrs", None)
            if isinstance(attrs, dict):
                directions = attrs.get("arg_directions")
        if not directions:
            return ()
        return tuple(cls.direction_name(direction) for direction in directions)

    @staticmethod
    def direction_name(direction: Any) -> str:
        name = getattr(direction, "name", None)
        if isinstance(name, str):
            return name
        value = getattr(direction, "value", None)
        if isinstance(value, str):
            return value
        return str(direction)


__all__ = [
    "CERTIFIED_DUMP_AFTER_COLLECT_COMM_GROUPS_SIMPLIFY",
    "DEFAULT_CERTIFIED_DUMP",
    "NormalizedCallFact",
    "NormalizedDependencyFact",
    "NormalizedFunctionFact",
    "NormalizedTaskFacts",
    "PostSimplifyPyPTOInputAdapter",
    "PyPTOAdapterContractError",
]
