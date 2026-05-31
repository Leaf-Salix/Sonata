"""Tests for the multi-adapter registry module."""

import pytest

from sonata.adapters import (
    AdapterCapability,
    AdapterDescriptor,
    AdapterRegistry,
    POST_SIMPLIFY,
    POST_SIMPLIFY_WITH_SCOPE,
    PRE_RUNTIME,
    default_registry,
)
from sonata.plan_handle import PlanHandle
from sonata.score import RuntimeTarget, Score, Task


def _make_score() -> Score:
    return Score(
        name="test",
        runtime_target=RuntimeTarget(),
        tasks=(Task(task_id=0, func_id=0, core_type="aic", name="f"),),
    )


class TestAdapterDescriptor:
    def test_construct(self):
        desc = AdapterDescriptor(name="test_adapter", version=2, description="test")
        assert desc.name == "test_adapter"
        assert desc.version == 2

    def test_frozen(self):
        desc = AdapterDescriptor(name="x")
        with pytest.raises(AttributeError):
            desc.name = "y"

    def test_to_dict_roundtrip(self):
        desc = POST_SIMPLIFY
        data = desc.to_dict()
        restored = AdapterDescriptor.from_dict(data)
        assert restored.name == desc.name
        assert restored.version == desc.version
        assert restored.capabilities == desc.capabilities
        assert restored.description == desc.description
        assert restored.certified_dump == desc.certified_dump

    def test_capabilities_default(self):
        caps = AdapterCapability()
        assert caps.static_shapes is True
        assert caps.dependency_kinds is False
        assert caps.control_flow_regions is False


class TestPredefinedAdapters:
    def test_post_simplify(self):
        assert POST_SIMPLIFY.name == "post_simplify"
        assert POST_SIMPLIFY.capabilities.static_shapes is True
        assert POST_SIMPLIFY.capabilities.dependency_kinds is False

    def test_pre_runtime(self):
        assert PRE_RUNTIME.name == "pre_runtime"
        assert PRE_RUNTIME.capabilities.dependency_kinds is True

    def test_post_simplify_with_scope(self):
        assert POST_SIMPLIFY_WITH_SCOPE.name == "post_simplify_with_scope"
        assert POST_SIMPLIFY_WITH_SCOPE.capabilities.runtime_scopes is True

    def test_all_distinct(self):
        names = {POST_SIMPLIFY.name, PRE_RUNTIME.name, POST_SIMPLIFY_WITH_SCOPE.name}
        assert len(names) == 3


class TestAdapterRegistry:
    def test_register_and_get(self):
        registry = AdapterRegistry()
        desc = AdapterDescriptor(name="custom")
        registry.register(desc)
        assert registry.get("custom") is desc

    def test_get_missing(self):
        registry = AdapterRegistry()
        assert registry.get("nonexistent") is None

    def test_duplicate_raises(self):
        registry = AdapterRegistry()
        registry.register(AdapterDescriptor(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(AdapterDescriptor(name="dup"))

    def test_names(self):
        registry = AdapterRegistry()
        registry.register(AdapterDescriptor(name="a"))
        registry.register(AdapterDescriptor(name="b"))
        assert registry.names() == frozenset({"a", "b"})

    def test_entry_count(self):
        registry = AdapterRegistry()
        assert registry.entry_count() == 0
        registry.register(AdapterDescriptor(name="x"))
        assert registry.entry_count() == 1

    def test_adapters(self):
        registry = AdapterRegistry()
        registry.register(AdapterDescriptor(name="a"))
        registry.register(AdapterDescriptor(name="b"))
        adapters = registry.adapters()
        assert len(adapters) == 2
        assert {a.name for a in adapters} == {"a", "b"}


class TestDefaultRegistry:
    def test_has_predefined_adapters(self):
        registry = default_registry()
        assert registry.entry_count() == 3
        assert registry.get("post_simplify") is not None
        assert registry.get("pre_runtime") is not None
        assert registry.get("post_simplify_with_scope") is not None

    def test_roundtrip(self):
        registry = default_registry()
        data = registry.to_dict()
        restored = AdapterRegistry.from_dict(data)
        assert restored.entry_count() == 3
        assert restored.names() == registry.names()


class TestAdapterSelect:
    def test_select_by_capability(self):
        registry = default_registry()
        result = registry.select(dependency_kinds=True)
        assert len(result) == 1
        assert result[0].name == "pre_runtime"

    def test_select_static_shapes(self):
        registry = default_registry()
        result = registry.select(static_shapes=True)
        assert len(result) == 3

    def test_select_runtime_scopes(self):
        registry = default_registry()
        result = registry.select(runtime_scopes=True)
        assert len(result) == 1
        assert result[0].name == "post_simplify_with_scope"

    def test_select_no_match(self):
        registry = AdapterRegistry()
        registry.register(AdapterDescriptor(
            name="basic", capabilities=AdapterCapability(static_shapes=False)
        ))
        result = registry.select(static_shapes=True)
        assert len(result) == 0

    def test_select_multiple_capabilities(self):
        registry = default_registry()
        result = registry.select(dependency_kinds=True, static_shapes=True)
        assert len(result) == 1
        assert result[0].name == "pre_runtime"


class TestValidatePlanHandle:
    def test_valid_adapter(self):
        registry = default_registry()
        score = _make_score()
        plan = PlanHandle.from_score(score, source_adapter="post_simplify")
        assert registry.validate_plan_handle(plan) is True

    def test_invalid_adapter(self):
        registry = default_registry()
        score = _make_score()
        plan = PlanHandle.from_score(score, source_adapter="unknown_stage")
        assert registry.validate_plan_handle(plan) is False

    def test_empty_registry(self):
        registry = AdapterRegistry()
        score = _make_score()
        plan = PlanHandle.from_score(score)
        assert registry.validate_plan_handle(plan) is False
