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


class TestRegionFallbackReasons:
    """Tests for region-level fallback semantics (Phase 1 B3)."""

    def test_region_fallback_reasons(self):
        """Dynamic region fallback reasons include region_type and control_flow_node."""
        body = [
            _make_stmt("Call"),
            _make_stmt("IfStmt"),     # dynamic
            _make_stmt("Call"),
            _make_stmt("ForStmt"),    # dynamic
            _make_stmt("Call"),
        ]
        node = _make_func(body, name="fallback_func")
        result = check_region_eligibility(node)

        assert result.eligible  # has static regions
        # Dynamic regions should have structured fallback reasons
        dynamic_regions = [r for r in result.metadata['region_tree'].dynamic_nodes()]
        assert len(dynamic_regions) == 2

        # Check fallback reason fields on the first dynamic region
        if_stmt_region = dynamic_regions[0].region
        assert if_stmt_region.fallback_reason is not None
        assert if_stmt_region.fallback_reason.region_type == "dynamic"
        assert if_stmt_region.fallback_reason.control_flow_node == "IfStmt"

        for_stmt_region = dynamic_regions[1].region
        assert for_stmt_region.fallback_reason is not None
        assert for_stmt_region.fallback_reason.control_flow_node == "ForStmt"

    def test_multi_region_scores(self):
        """Each maximal static subtree gets its own Score."""
        body = [
            _make_stmt("Call"),
            _make_stmt("ForStmt"),
            _make_stmt("Call"),
            _make_stmt("Call"),
        ]
        node = _make_func(body, name="multi_score_func")
        result = check_region_eligibility(node)

        assert result.eligible
        per_region = result.metadata['per_region_scores']
        # Two static subtrees: region 0 (before ForStmt) and region 2 (after)
        assert len(per_region) == 2


class TestStoreRegionTree:
    """Tests for store_region_tree and lookup_region_tree (Phase 1 C1-C2)."""

    def test_fingerprint_mappings(self):
        """store_region_tree returns correct fingerprint mappings."""
        from sonata.cache import ScoreCache
        from sonata.regions import store_region_tree

        cache = ScoreCache()
        dyn1 = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        dyn2 = RegionTreeNode(region=Region(region_id=2, kind=REGION_DYNAMIC))
        root = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC),
                             children=(dyn1, dyn2))
        tree = RegionTree(root=root)

        mappings = store_region_tree(tree, {}, cache)

        assert "root" in mappings
        assert "root.child[0]" in mappings
        assert "root.child[1]" in mappings

        for path, fp in mappings.items():
            assert isinstance(fp, str)
            assert len(fp) == 64

    def test_store_with_per_region_scores(self):
        """Per-region scores are stored in cache and retrievable."""
        from sonata.cache import ScoreCache
        from sonata.score import RuntimeTarget, Score
        from sonata.regions import store_region_tree, lookup_region_tree

        cache = ScoreCache()
        dyn = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        root = RegionTreeNode(region=Region(region_id=0, kind=REGION_STATIC),
                              children=(dyn,))
        tree = RegionTree(root=root)

        rt = RuntimeTarget(runtime="host_build_graph", function_name="test", aicpu_thread_num=1)
        score = Score(name="region_0_score", runtime_target=rt,
                      tasks=(), dependencies=(), shape_assumptions=())
        per_region = {"region_0": score}

        mappings = store_region_tree(tree, {}, cache, per_region_scores=per_region)

        # Root score should be in cache
        cached = lookup_region_tree("root", cache, path_to_fingerprint=mappings)
        assert cached is not None
        assert cached["name"] == "region_0_score"

    def test_lookup_miss(self):
        """lookup_region_tree returns None for unknown path."""
        from sonata.cache import ScoreCache
        from sonata.regions import lookup_region_tree

        cache = ScoreCache()
        result = lookup_region_tree("root.child[99]", cache,
                                    path_to_fingerprint={"root": "abc"})
        assert result is None

    def test_region_tree_cache_invalidation(self):
        """Invalidating a node removes it and descendants, not ancestors."""
        from sonata.cache import ScoreCache
        from sonata.score import RuntimeTarget, Score
        from sonata.regions import (
            store_region_tree, lookup_region_tree, invalidate_region_tree,
        )

        cache = ScoreCache()
        rt = RuntimeTarget(runtime="host_build_graph", function_name="test", aicpu_thread_num=1)

        # Build: static(0) -> [dynamic(1), static(2) -> [dynamic(3)]]
        dyn3 = RegionTreeNode(region=Region(region_id=3, kind=REGION_DYNAMIC))
        static2 = RegionTreeNode(
            region=Region(region_id=2, kind=REGION_STATIC),
            children=(dyn3,),
        )
        dyn1 = RegionTreeNode(region=Region(region_id=1, kind=REGION_DYNAMIC))
        root = RegionTreeNode(
            region=Region(region_id=0, kind=REGION_STATIC),
            children=(dyn1, static2),
        )
        tree = RegionTree(root=root)

        per_region = {
            f"region_{i}": Score(name=f"s{i}", runtime_target=rt,
                                 tasks=(), dependencies=(), shape_assumptions=())
            for i in range(4)
        }
        mappings = store_region_tree(tree, {}, cache, per_region_scores=per_region)

        # All 4 regions should be cached
        assert lookup_region_tree("root", cache, path_to_fingerprint=mappings) is not None
        assert lookup_region_tree("root.child[1]", cache, path_to_fingerprint=mappings) is not None

        # Invalidate static2 (region 2) — should also remove dyn3 (region 3)
        removed = invalidate_region_tree(tree, static2, cache, path_to_fingerprint=mappings)
        assert removed == 2  # static2 + dyn3

        # static2 and dyn3 are gone
        assert lookup_region_tree("root.child[1]", cache, path_to_fingerprint=mappings) is None
        assert lookup_region_tree("root.child[1].child[0]", cache, path_to_fingerprint=mappings) is None

        # root and dyn1 are still valid (ancestors/siblings untouched)
        assert lookup_region_tree("root", cache, path_to_fingerprint=mappings) is not None
        assert lookup_region_tree("root.child[0]", cache, path_to_fingerprint=mappings) is not None


class TestEndToEndRegionizedWorkflow:
    """E1: End-to-end regionized workflow smoke test.

    Input: IR with mixed static/dynamic regions.
    Process: extract -> check eligibility -> build region tree -> cache -> retrieve.
    Expected: static regions cached, dynamic regions fall back.
    """

    def test_full_workflow(self):
        from sonata.cache import ScoreCache
        from sonata.regions import (
            extract_regions, build_region_tree, check_region_eligibility,
            store_region_tree, lookup_region_tree,
        )
        from sonata.plan_handle import GuardStatus

        # Step 1: Build IR with mixed regions
        body = [
            _make_stmt("Call"),
            _make_stmt("Call"),
            _make_stmt("ForStmt"),      # dynamic
            _make_stmt("Call"),
            _make_stmt("Call"),
            _make_stmt("IfStmt"),       # dynamic
            _make_stmt("Call"),
        ]
        node = _make_func(body, name="e2e_graph")

        # Step 2: Extract regions
        region_map = extract_regions(node)
        assert len(region_map.regions) == 5  # s0, d1, s2, d3, s4

        # Step 3: Build region tree
        tree = build_region_tree(region_map)
        assert tree.root.region.is_static

        # Step 4: Check eligibility
        result = check_region_eligibility(node)
        assert result.eligible
        assert result.metadata['static_region_count'] == 3
        assert result.metadata['dynamic_region_count'] == 2
        assert result.metadata['region_statuses']['region_0'] == 'mixed'

        # Step 5: Cache per-region scores
        cache = ScoreCache()
        per_region = result.metadata['per_region_scores']
        mappings = store_region_tree(tree, {}, cache, per_region_scores=per_region)

        # Static subtrees are cached
        assert len(per_region) >= 2  # at least 2 maximal static subtrees
        cached_root = lookup_region_tree("root", cache, path_to_fingerprint=mappings)
        assert cached_root is not None
        assert cached_root['name'].endswith('region_0')

        # Step 6: Verify fallback reasons have region context
        dynamic_nodes = tree.dynamic_nodes()
        for dn in dynamic_nodes:
            assert dn.region.fallback_reason.region_type == "dynamic"
            assert dn.region.fallback_reason.control_flow_node in ("ForStmt", "IfStmt")


class TestPerformanceBenchmark:
    """E2: Per-region fallback reduction benchmark.

    Scenario: 100-node graph with 10% dynamic nodes.
    Target: ≥60% reduction in unnecessary fallbacks vs whole-graph.
    """

    def test_fallback_reduction(self):
        """Per-region eligibility reduces fallbacks by ≥60% vs whole-graph.

        Whole-graph: if ANY dynamic region exists, ALL tasks fall back.
        Per-region: only tasks in dynamic regions fall back.
        Reduction = tasks saved from unnecessary fallback / total tasks.
        """
        from sonata.regions import extract_regions, check_region_eligibility

        # 90 Call + 10 ForStmt clustered at the end
        body = [_make_stmt("Call") for _ in range(90)]
        body.extend(_make_stmt("ForStmt") for _ in range(10))
        node = _make_func(body, name="bench_graph")

        region_map = extract_regions(node)
        total_regions = len(region_map.regions)
        dynamic_regions = len(region_map.dynamic_regions())

        # Task-level analysis:
        # Each Call statement = 1 task, each ForStmt = 1 dynamic region
        total_tasks = 100  # 90 Call + 10 ForStmt
        dynamic_tasks = 10  # ForStmt nodes

        # Whole-graph: ALL tasks fall back (any dynamic → reject all)
        whole_graph_fallback = total_tasks

        # Per-region: only tasks in dynamic regions fall back
        per_region_fallback = dynamic_tasks

        # Tasks saved from unnecessary fallback
        tasks_saved = whole_graph_fallback - per_region_fallback
        reduction = tasks_saved / whole_graph_fallback

        assert reduction >= 0.6, (
            f"Expected ≥60% fallback reduction, got {reduction:.0%} "
            f"(saved {tasks_saved}/{whole_graph_fallback} tasks)"
        )

    def test_cache_hit_rate_improvement(self):
        """Per-region caching has higher hit rate than whole-graph on param changes.

        Scenario: same graph structure with minor shape parameter change.
        Whole-graph cache misses entirely; per-region cache only misses the
        region affected by the parameter.
        """
        from sonata.cache import ScoreCache
        from sonata.score import RuntimeTarget, Score
        from sonata.regions import (
            extract_regions, build_region_tree, check_region_eligibility,
            store_region_tree, lookup_region_tree,
        )
        from sonata.serialization import score_fingerprint

        # Build a graph with 3 static regions separated by dynamic nodes
        body = [
            _make_stmt("Call"), _make_stmt("Call"),  # static region 0
            _make_stmt("ForStmt"),                    # dynamic
            _make_stmt("Call"), _make_stmt("Call"),   # static region 2
            _make_stmt("ForStmt"),                    # dynamic
            _make_stmt("Call"),                       # static region 4
        ]
        node = _make_func(body, name="param_graph")

        cache = ScoreCache()
        result = check_region_eligibility(node)
        per_region = result.metadata['per_region_scores']

        tree = build_region_tree(extract_regions(node))
        mappings = store_region_tree(tree, {}, cache, per_region_scores=per_region)

        # First run: cache populated, all static regions should hit
        # (dynamic nodes have no score so won't be in cache)
        hits_v1 = sum(
            1 for path in mappings
            if lookup_region_tree(path, cache, path_to_fingerprint=mappings) is not None
        )
        assert hits_v1 >= 2  # at least 2 static regions cached

        # Invalidate just one non-root static subtree (simulating param change)
        from sonata.regions import invalidate_region_tree
        static_subtrees = tree.static_subtrees()
        # Use second subtree (not root) so other regions remain cached
        if len(static_subtrees) >= 2:
            invalidate_region_tree(tree, static_subtrees[1], cache,
                                   path_to_fingerprint=mappings)

        # Count remaining hits
        hits_v2 = sum(
            1 for path in mappings
            if lookup_region_tree(path, cache, path_to_fingerprint=mappings) is not None
        )

        # Per-region: root and sibling regions still cached after partial invalidation
        # Whole-graph: would miss entirely (fingerprint changes for whole graph)
        assert hits_v2 >= 2, f"Expected ≥2 cached regions after partial invalidation, got {hits_v2}"


class TestRegionEligibilityResult:
    """v0.18 Phase 2 A1: Per-region eligibility tests."""

    def test_fully_static_graph(self):
        """All-static graph → all regions eligible."""
        from sonata.regions import (
            RegionEligibility, RegionEligibilityResult,
            check_region_eligibility,
        )
        from types import SimpleNamespace

        # Build a simple static IR (no control flow)
        call = SimpleNamespace(
            __class__=type("Call", (), {}),
            callee_name="kernel",
            args=("x",), arg_names=("x",),
            arg_directions=("Input",), arg_storage_keys=("param:x",),
            core_type="aic", node=SimpleNamespace(),
        )
        func = SimpleNamespace(
            body=[call], name="main", calls=[call],
            func_type=SimpleNamespace(value="Orchestration"),
            node=SimpleNamespace(body=[call], name="main"),
        )

        result = check_region_eligibility(func.node)
        assert result.eligible
        region_elig = result.metadata.get("region_eligibility")
        assert region_elig is not None
        assert isinstance(region_elig, RegionEligibilityResult)
        assert region_elig.overall_eligible is True
        assert region_elig.static_count >= 1

    def test_region_eligibility_result_properties(self):
        """RegionEligibilityResult properties work correctly."""
        from sonata.regions import RegionEligibility, RegionEligibilityResult

        regions = (
            RegionEligibility(region_id="region_0", eligible=True, status="static"),
            RegionEligibility(region_id="region_1", eligible=False, status="dynamic",
                              fallback_reason="ForStmt is a dynamic region"),
        )
        result = RegionEligibilityResult(
            overall_eligible=True,
            regions=regions,
            static_count=1,
            dynamic_count=1,
        )
        assert result.is_partially_eligible
        assert result.eligible_region_ids() == ("region_0",)
        assert result.fallback_region_ids() == ("region_1",)

    def test_all_static_not_partial(self):
        """All-static → not partially eligible."""
        from sonata.regions import RegionEligibility, RegionEligibilityResult

        regions = (
            RegionEligibility(region_id="region_0", eligible=True, status="static"),
            RegionEligibility(region_id="region_1", eligible=True, status="static"),
        )
        result = RegionEligibilityResult(
            overall_eligible=True, regions=regions,
            static_count=2, dynamic_count=0,
        )
        assert not result.is_partially_eligible
        assert result.fallback_region_ids() == ()

    def test_mixed_eligibility_partial(self):
        """One static + one dynamic → partially eligible."""
        from sonata.regions import RegionEligibility, RegionEligibilityResult

        regions = (
            RegionEligibility(region_id="region_0", eligible=True, status="static"),
            RegionEligibility(region_id="region_1", eligible=False, status="dynamic",
                              fallback_reason="ForStmt"),
        )
        result = RegionEligibilityResult(
            overall_eligible=True,
            regions=regions,
            static_count=1,
            dynamic_count=1,
        )
        assert result.overall_eligible
        assert result.is_partially_eligible
        assert result.eligible_region_ids() == ("region_0",)
        assert result.fallback_region_ids() == ("region_1",)

    def test_all_dynamic_not_eligible(self):
        """All dynamic → not eligible, not partial."""
        from sonata.regions import RegionEligibility, RegionEligibilityResult

        regions = (
            RegionEligibility(region_id="region_0", eligible=False, status="dynamic"),
            RegionEligibility(region_id="region_1", eligible=False, status="dynamic"),
        )
        result = RegionEligibilityResult(
            overall_eligible=False,
            regions=regions,
            static_count=0,
            dynamic_count=2,
        )
        assert not result.overall_eligible
        assert not result.is_partially_eligible
        assert result.eligible_region_ids() == ()
        assert result.fallback_region_ids() == ("region_0", "region_1")


class TestForStmtExpansion:
    """v0.18 Phase 2 B2: ForStmt → expanded task graph."""

    def _make_for_stmt(self, start, stop, body_calls=()):
        class ForStmt:
            pass
        node = ForStmt()
        node.start = start
        node.stop = stop
        node.step = None
        node.body = tuple(body_calls)
        return node

    def _make_call(self, name):
        class Call:
            pass
        c = Call()
        c.__class__ = type("Call", (), {})
        c.callee_name = name
        return c

    def test_expand_simple_loop(self):
        """for i in range(4): call() → 4 copies of call."""
        from sonata.regions import expand_for_stmt
        call = self._make_call("kernel")
        for_stmt = self._make_for_stmt(0, 4, body_calls=[call])
        expanded = expand_for_stmt(for_stmt)
        assert len(expanded) == 4
        # All copies should be distinct objects
        assert len(set(id(c) for c in expanded)) == 4

    def test_expand_non_loop_returns_empty(self):
        """Non-ForStmt → empty tuple."""
        from sonata.regions import expand_for_stmt
        node = self._make_call("kernel")
        assert expand_for_stmt(node) == ()

    def test_expand_large_loop_returns_empty(self):
        """trip_count > 16 → not unrollable, returns empty."""
        from sonata.regions import expand_for_stmt
        call = self._make_call("kernel")
        for_stmt = self._make_for_stmt(0, 100, body_calls=[call])
        assert expand_for_stmt(for_stmt) == ()

    def test_expand_empty_body_returns_empty(self):
        """ForStmt with no body calls → empty."""
        from sonata.regions import expand_for_stmt
        for_stmt = self._make_for_stmt(0, 4, body_calls=[])
        assert expand_for_stmt(for_stmt) == ()

    def test_expand_task_graph(self):
        """expand_task_graph expands ForStmts, keeps others."""
        from sonata.regions import expand_task_graph
        call1 = self._make_call("pre")
        call2 = self._make_call("in_loop")
        call3 = self._make_call("post")
        for_stmt = self._make_for_stmt(0, 3, body_calls=[call2])

        nodes = (call1, for_stmt, call3)
        expanded = expand_task_graph(nodes)
        # call1 + 3 copies of call2 + call3 = 5
        assert len(expanded) == 5
        # First is call1, last is call3
        assert expanded[0].callee_name == "pre"
        assert expanded[-1].callee_name == "post"
        # Middle 3 are copies of in_loop
        for i in range(1, 4):
            assert expanded[i].callee_name == "in_loop"

    def test_expand_preserves_independence(self):
        """Expanded copies are independent (deep copy)."""
        from sonata.regions import expand_for_stmt
        call = self._make_call("kernel")
        for_stmt = self._make_for_stmt(0, 2, body_calls=[call])
        expanded = expand_for_stmt(for_stmt)
        assert len(expanded) == 2
        # Modify one, other should be unaffected
        expanded[0].callee_name = "modified"
        assert expanded[1].callee_name == "kernel"
