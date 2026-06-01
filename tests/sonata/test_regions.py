"""Tests for v0.4 region extraction and region-level eligibility."""

from types import SimpleNamespace

import pytest

from sonata.fallback import FallbackCode
from sonata.regions import (
    REGION_DYNAMIC,
    REGION_STATIC,
    Region,
    RegionMap,
    RegionTreeNode,
    RegionTree,
    build_region_tree,
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


# ============================================================================
# v0.11 Phase 1 A: RegionTree Tests
# ============================================================================

class TestRegionTreeNode:
    """Tests for RegionTreeNode data structure (Phase 1 A1)."""
    
    def test_leaf_node(self):
        region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=region)
        assert node.is_leaf
        assert not node.children
    
    def test_node_with_children(self):
        static_region = Region(region_id=0, kind=REGION_STATIC)
        dynamic_region = Region(region_id=1, kind=REGION_DYNAMIC)
        
        child = RegionTreeNode(region=dynamic_region)
        parent = RegionTreeNode(region=static_region, children=(child,))
        
        assert not parent.is_leaf
        assert len(parent.children) == 1
        assert parent.children[0].region.region_id == 1
    
    def test_is_static_subtree(self):
        # All static subtree
        static1 = Region(region_id=0, kind=REGION_STATIC)
        static2 = Region(region_id=1, kind=REGION_STATIC)
        child = RegionTreeNode(region=static2)
        parent = RegionTreeNode(region=static1, children=(child,))
        assert parent.is_static_subtree
        
        # Subtree with dynamic child
        dynamic = Region(region_id=2, kind=REGION_DYNAMIC)
        child_dyn = RegionTreeNode(region=dynamic)
        parent_mixed = RegionTreeNode(region=static1, children=(child_dyn,))
        assert not parent_mixed.is_static_subtree
    
    def test_fingerprint_computation(self):
        """Test Phase 1 A3: fingerprint computation for RegionTreeNode."""
        region = Region(region_id=42, kind=REGION_STATIC, nodes=(1, 2, 3))
        node = RegionTreeNode(region=region)
        
        fp = node.fingerprint
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex digest
        
        # Same region should produce same fingerprint
        node2 = RegionTreeNode(region=region)
        assert node.fingerprint == node2.fingerprint
        
        # Different regions should produce different fingerprints
        region2 = Region(region_id=43, kind=REGION_STATIC)
        node3 = RegionTreeNode(region=region2)
        assert node.fingerprint != node3.fingerprint
    
    def test_update_score(self):
        from sonata.score import Score, RuntimeTarget
        
        region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=region)
        
        rt = RuntimeTarget(runtime="host_build_graph", function_name="test", aicpu_thread_num=1)
        score = Score(name="test", runtime_target=rt, tasks=(), dependencies=(), shape_assumptions=())
        new_node = node.update_score(score)
        
        assert new_node.score is score
        assert new_node.fingerprint != node.fingerprint  # fingerprint changed


class TestRegionTree:
    """Tests for RegionTree data structure (Phase 1 A1-A2)."""
    
    def test_single_node_tree(self):
        region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=region)
        tree = RegionTree(root=node)
        
        assert tree.root is node
        assert len(tree.all_nodes) == 1
        assert tree.static_ratio == 1.0
    
    def test_tree_with_children(self):
        static_region = Region(region_id=0, kind=REGION_STATIC)
        dynamic_region = Region(region_id=1, kind=REGION_DYNAMIC)
        
        child = RegionTreeNode(region=dynamic_region)
        root = RegionTreeNode(region=static_region, children=(child,))
        tree = RegionTree(root=root)
        
        all_nodes = tree.all_nodes
        assert len(all_nodes) == 2
        assert tree.root in all_nodes
        assert child in all_nodes
    
    def test_static_subtrees(self):
        """Test finding maximal static subtrees."""
        # Create: static(0) -> [dynamic(1), dynamic(2)]
        dyn1 = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        dyn2 = RegionTreeNode(region=Region(region_id=2, kind=REGION_DYNAMIC))
        static = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC), 
                               children=(dyn1, dyn2))
        tree = RegionTree(root=static)
        
        subtrees = tree.static_subtrees()
        assert len(subtrees) == 1
        assert subtrees[0].region.region_id == 0
    
    def test_dynamic_nodes(self):
        """Test finding all dynamic nodes."""
        static = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC))
        dyn1 = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        dyn2 = RegionTreeNode(region=Region(region_id=2, kind=REGION_DYNAMIC))
        
        root = RegionTreeNode(region=Region(region_id=3, kind=REGION_STATIC),
                             children=(static, dyn1, dyn2))
        tree = RegionTree(root=root)
        
        dyn_nodes = tree.dynamic_nodes()
        assert len(dyn_nodes) == 2
        assert all(n.region.is_dynamic for n in dyn_nodes)
    
    def test_static_ratio(self):
        """Test static ratio calculation."""
        # 3 static, 1 dynamic = 75% static
        s1 = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC))
        s2 = RegionTreeNode(region=Region(region_id=1, kind=REGION_STATIC))
        s3 = RegionTreeNode(region=Region(region_id=2, kind=REGION_STATIC))
        d1 = RegionTreeNode(region=Region(region_id=3, kind=REGION_DYNAMIC))
        
        root = RegionTreeNode(region=Region(region_id=4, kind=REGION_STATIC),
                             children=(s1, s2, s3, d1))
        tree = RegionTree(root=root)
        
        assert tree.static_ratio == 4 / 5


class TestBuildRegionTree:
    """Tests for build_region_tree function (Phase 1 A2)."""
    
    def test_empty_region_map(self):
        rm = RegionMap()
        tree = build_region_tree(rm)
        
        assert tree.root.region.is_static
        assert tree.root.region.region_id == 0
        assert tree.root.is_leaf
    
    def test_single_region(self):
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_STATIC),
        ))
        tree = build_region_tree(rm)
        
        assert tree.root.region.region_id == 0
        assert tree.root.is_leaf
    
    def test_static_then_dynamic(self):
        """Test grouping: static(0) + dynamic(1) -> static with dynamic child."""
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_DYNAMIC),
        ))
        tree = build_region_tree(rm)
        
        assert tree.root.region.is_static
        assert tree.root.region.region_id == 0
        assert len(tree.root.children) == 1
        assert tree.root.children[0].region.is_dynamic
    
    def test_multiple_dynamic_under_one_static(self):
        """Test multiple consecutive dynamic regions under one static parent."""
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_DYNAMIC),
            Region(region_id=2, kind=REGION_DYNAMIC),
            Region(region_id=3, kind=REGION_STATIC),
        ))
        tree = build_region_tree(rm)
        
        # Root should be static(0) with dynamic children + subsequent static as sibling
        assert tree.root.region.region_id == 0
        assert tree.root.region.is_static
        assert len(tree.root.children) == 3
        assert tree.root.children[0].region.is_dynamic
        assert tree.root.children[1].region.is_dynamic
        assert tree.root.children[2].region.is_static  # static(3) is sibling
    
    def test_alternating_pattern(self):
        """Test alternating static/dynamic pattern."""
        rm = RegionMap(regions=(
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_DYNAMIC),
            Region(region_id=2, kind=REGION_STATIC),
            Region(region_id=3, kind=REGION_DYNAMIC),
        ))
        tree = build_region_tree(rm)
        
        # First static should have first dynamic as child
        assert tree.root.region.region_id == 0
        assert tree.root.children[0].region.region_id == 1
        # Second static should be sibling of first dynamic
        assert tree.root.children[1].region.region_id == 2


# ============================================================================
# v0.11 Phase 1 B-C: Per-Region Eligibility and Cache Tests  
# ============================================================================

class TestPerRegionEligibility:
    """Tests for per-region eligibility checking (Phase 1 B)."""
    
    def test_partial_eligibility_metadata(self):
        """Test that partial eligibility includes region metadata."""
        body = [
            _make_stmt("Call"),
            _make_stmt("Call"),
            _make_stmt("ForStmt"),  # Dynamic region
            _make_stmt("Call"),
        ]
        node = _make_func(body, name="test_func")
        result = check_region_eligibility(node)
        
        assert result.eligible  # Has static regions
        assert hasattr(result, 'metadata')
        assert result.metadata is not None
        
        # Check region metadata
        assert 'region_count' in result.metadata
        assert 'static_region_count' in result.metadata
        assert 'dynamic_region_count' in result.metadata
        assert 'static_ratio' in result.metadata
        assert 'static_subtrees' in result.metadata
        
        # Verify counts
        assert result.metadata['region_count'] == 3
        assert result.metadata['static_region_count'] == 2
        assert result.metadata['dynamic_region_count'] == 1
        assert result.metadata['static_ratio'] == 2 / 3
    
    def test_per_region_scores_structure(self):
        """Test that per-region scores are stored in metadata."""
        body = [
            _make_stmt("Call"),
            _make_stmt("Call"),
        ]
        node = _make_func(body, name="test_func")
        result = check_region_eligibility(node)
        
        assert 'per_region_scores' in result.metadata
        assert isinstance(result.metadata['per_region_scores'], dict)
        assert len(result.metadata['per_region_scores']) > 0

    def test_mixed_graph_eligibility(self):
        """Test mixed graph: static root with dynamic children is partially eligible."""
        body = [
            _make_stmt("Call"),
            _make_stmt("ForStmt"),   # dynamic
            _make_stmt("Call"),
        ]
        node = _make_func(body, name="mixed_func")
        result = check_region_eligibility(node)

        # Partially eligible: has static regions
        assert result.eligible
        assert result.metadata is not None

        # Root (region 0) is static, has dynamic child → mixed
        statuses = result.metadata['region_statuses']
        assert statuses['region_0'] == 'mixed'
        assert statuses['region_1'] == 'dynamic'
        assert statuses['region_2'] == 'static'

        # Should have per-region scores for static subtrees
        assert len(result.metadata['per_region_scores']) > 0
        # Dynamic region should have a fallback reason
        assert result.metadata['dynamic_region_count'] == 1


class TestRegionStatus:
    """Tests for RegionTreeNode.region_status property."""

    def test_static_leaf(self):
        node = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC))
        assert node.region_status == "static"

    def test_dynamic_leaf(self):
        node = RegionTreeNode(region=Region(region_id=0, kind=REGION_DYNAMIC))
        assert node.region_status == "dynamic"

    def test_static_with_dynamic_children_is_mixed(self):
        dyn = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        root = RegionTreeNode(
            region=Region(region_id=0, kind=REGION_STATIC),
            children=(dyn,),
        )
        assert root.region_status == "mixed"

    def test_static_with_static_children_is_static(self):
        child = RegionTreeNode(region=Region(region_id=1, kind=REGION_STATIC))
        root = RegionTreeNode(
            region=Region(region_id=0, kind=REGION_STATIC),
            children=(child,),
        )
        assert root.region_status == "static"

    def test_static_with_mixed_children_is_mixed(self):
        dyn = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        sta = RegionTreeNode(region=Region(region_id=2, kind=REGION_STATIC))
        root = RegionTreeNode(
            region=Region(region_id=0, kind=REGION_STATIC),
            children=(dyn, sta),
        )
        assert root.region_status == "mixed"
    """Tests for store_region_tree function (Phase 1 C1-C3)."""
    
    def test_fingerprint_mappings(self):
        """Test that store_region_tree returns correct fingerprint mappings."""
        from sonata.regions import store_region_tree
        
        # Create a simple tree: static(0) -> [dynamic(1), dynamic(2)]
        dyn1 = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        dyn2 = RegionTreeNode(region=Region(region_id=2, kind=REGION_DYNAMIC))
        root = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC),
                             children=(dyn1, dyn2))
        tree = RegionTree(root=root)
        
        # Call store_region_tree (returns mappings even without actual cache)
        mappings = store_region_tree(tree, {}, None)
        
        # Should have mappings for root and children
        assert "root" in mappings
        assert "root.child[0]" in mappings
        assert "root.child[1]" in mappings
        
        # All values should be fingerprints (hex strings)
        for path, fp in mappings.items():
            assert isinstance(fp, str)
            assert len(fp) == 64
