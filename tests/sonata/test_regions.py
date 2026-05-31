"""Tests for v0.4 region extraction and region-level eligibility."""

from types import SimpleNamespace

from sonata.fallback import FallbackCode
from sonata.regions import (
    REGION_DYNAMIC,
    REGION_STATIC,
    Region,
    RegionMap,
    check_region_eligibility,
    extract_regions,
)


def _make_stmt(kind: str, **attrs):
    obj = type(kind, (), {"__init__": lambda self: None})()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _make_func(body: list, **attrs):
    ns = SimpleNamespace(body=body, **attrs)
    return ns


class TestExtractRegions:
    def test_all_static(self):
        body = [_make_stmt("Call"), _make_stmt("AssignStmt"), _make_stmt("Call")]
        node = _make_func(body)
        rm = extract_regions(node)
        assert len(rm.regions) == 1
        assert rm.regions[0].is_static
        assert rm.regions[0].node_count == 3

    def test_all_dynamic(self):
        body = [_make_stmt("ForStmt"), _make_stmt("IfStmt")]
        node = _make_func(body)
        rm = extract_regions(node)
        assert len(rm.regions) == 2
        assert all(r.is_dynamic for r in rm.regions)
        assert rm.regions[0].control_flow_kind == "ForStmt"
        assert rm.regions[1].control_flow_kind == "IfStmt"

    def test_mixed_static_dynamic(self):
        body = [
            _make_stmt("Call"),
            _make_stmt("Call"),
            _make_stmt("ForStmt"),
            _make_stmt("Call"),
        ]
        node = _make_func(body)
        rm = extract_regions(node)
        assert len(rm.regions) == 3
        assert rm.regions[0].is_static
        assert rm.regions[0].node_count == 2
        assert rm.regions[1].is_dynamic
        assert rm.regions[1].control_flow_kind == "ForStmt"
        assert rm.regions[2].is_static
        assert rm.regions[2].node_count == 1

    def test_static_ratio(self):
        body = [
            _make_stmt("Call"),
            _make_stmt("ForStmt"),
            _make_stmt("Call"),
        ]
        node = _make_func(body)
        rm = extract_regions(node)
        assert rm.static_ratio() == 2 / 3

    def test_empty_body(self):
        node = SimpleNamespace()
        rm = extract_regions(node)
        assert len(rm.regions) == 1
        assert rm.regions[0].is_static

    def test_runtime_scope_is_dynamic(self):
        body = [_make_stmt("Call"), _make_stmt("RuntimeScopeStmt")]
        node = _make_func(body)
        rm = extract_regions(node)
        assert len(rm.regions) == 2
        assert rm.regions[1].is_dynamic
        assert rm.regions[1].control_flow_kind == "RuntimeScopeStmt"

    def test_while_stmt_is_dynamic(self):
        body = [_make_stmt("WhileStmt")]
        node = _make_func(body)
        rm = extract_regions(node)
        assert rm.regions[0].is_dynamic
        assert rm.regions[0].control_flow_kind == "WhileStmt"

    def test_alternating_static_dynamic(self):
        body = [
            _make_stmt("Call"),
            _make_stmt("ForStmt"),
            _make_stmt("Call"),
            _make_stmt("IfStmt"),
            _make_stmt("Call"),
        ]
        node = _make_func(body)
        rm = extract_regions(node)
        assert len(rm.regions) == 5
        kinds = [r.kind for r in rm.regions]
        assert kinds == [REGION_STATIC, REGION_DYNAMIC, REGION_STATIC,
                         REGION_DYNAMIC, REGION_STATIC]


class TestRegionMap:
    def test_all_static(self):
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_STATIC),
        ))
        assert rm.all_static()
        assert not rm.all_dynamic()

    def test_all_dynamic(self):
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_DYNAMIC),
        ))
        assert rm.all_dynamic()
        assert not rm.all_static()

    def test_static_regions(self):
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_DYNAMIC),
            Region(region_id=2, kind=REGION_STATIC),
        ))
        assert len(rm.static_regions()) == 2
        assert len(rm.dynamic_regions()) == 1

    def test_empty(self):
        rm = RegionMap()
        assert rm.static_ratio() == 0.0


class TestRegion:
    def test_is_static(self):
        r = Region(region_id=0, kind=REGION_STATIC)
        assert r.is_static
        assert not r.is_dynamic

    def test_is_dynamic(self):
        r = Region(region_id=0, kind=REGION_DYNAMIC, control_flow_kind="ForStmt")
        assert r.is_dynamic
        assert not r.is_static

    def test_fallback_reason(self):
        from sonata.score import FallbackReason
        reason = FallbackReason(
            code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
            message="ForStmt region falls back",
        )
        r = Region(region_id=0, kind=REGION_DYNAMIC, fallback_reason=reason)
        assert r.fallback_reason is not None
        assert r.fallback_reason.code == "control_flow_not_supported"

    def test_node_count(self):
        r = Region(region_id=0, kind=REGION_STATIC, nodes=(1, 2, 3))
        assert r.node_count == 3


class TestCheckRegionEligibility:
    def test_all_dynamic_rejects(self):
        body = [_make_stmt("ForStmt")]
        node = _make_func(body, name="test_func")
        result = check_region_eligibility(node)
        assert not result.eligible

    def test_dynamic_fallback_reasons(self):
        body = [_make_stmt("ForStmt"), _make_stmt("IfStmt")]
        node = _make_func(body, name="test_func")
        result = check_region_eligibility(node)
        assert not result.eligible
        codes = [r.code for r in result.reason_details]
        assert FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value in codes
