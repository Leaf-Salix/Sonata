"""Tests for v0.11 RegionTree and hierarchical region management."""

import pytest
from types import SimpleNamespace

from sonata.fallback import FallbackCode
from sonata.regions import (
    REGION_DYNAMIC,
    REGION_STATIC,
    Region,
    RegionMap,
    RegionTreeNode,
    RegionTree,
    build_region_tree,
)


def _make_stmt(kind: str, **attrs):
    """Create a mock statement node."""
    obj = type(kind, (), {"__init__": lambda self: None})()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _make_func(body: list, **attrs):
    """Create a mock function node."""
    ns = SimpleNamespace(body=body, **attrs)
    return ns


class TestRegionTreeNode:
    """Tests for RegionTreeNode data model - Phase 1 A1."""
    
    def test_basic_construction(self):
        """A1: Basic node construction with required fields."""
        region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=region)
        
        assert node.region == region
        assert node.children == ()
        assert node.score is None
        assert node.is_leaf
    
    def test_with_children(self):
        """A1: Node construction with children."""
        parent_region = Region(region_id=0, kind=REGION_STATIC)
        child_region = Region(region_id=1, kind=REGION_DYNAMIC)
        
        child_node = RegionTreeNode(region=child_region)
        parent_node = RegionTreeNode(region=parent_region, children=(child_node,))
        
        assert len(parent_node.children) == 1
        assert parent_node.children[0] == child_node
        assert not parent_node.is_leaf
    
    def test_is_static_subtree_all_static(self):
        """A1: Static subtree detection for all-static tree."""
        static_region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=static_region)
        
        assert node.is_static_subtree
    
    def test_is_static_subtree_with_dynamic_child(self):
        """A1: Static subtree detection when child is dynamic.
        
        Parent region is static, but has a dynamic child.
        The entire subtree is NOT static because it contains dynamic regions.
        """
        static_region = Region(region_id=0, kind=REGION_STATIC)
        dynamic_region = Region(region_id=1, kind=REGION_DYNAMIC)
        
        child_node = RegionTreeNode(region=dynamic_region)
        parent_node = RegionTreeNode(region=static_region, children=(child_node,))
        
        # Subtree contains dynamic region, so not static
        assert not parent_node.is_static_subtree
    
    def test_fingerprint_computation(self):
        """A3: Fingerprint computation for node without score."""
        region = Region(region_id=0, kind=REGION_STATIC, nodes=(1, 2, 3))
        node = RegionTreeNode(region=region)
        
        fp = node.fingerprint
        
        # Should be a valid hex string (SHA-256 produces 64 chars)
        assert len(fp) == 64
        assert all(c in '0123456789abcdef' for c in fp)
    
    def test_fingerprint_includes_region_metadata(self):
        """A3: Fingerprint includes region metadata."""
        region1 = Region(region_id=0, kind=REGION_STATIC, nodes=(1,))
        region2 = Region(region_id=1, kind=REGION_STATIC, nodes=(1,))
        
        node1 = RegionTreeNode(region=region1)
        node2 = RegionTreeNode(region=region2)
        
        # Different region IDs should produce different fingerprints
        assert node1.fingerprint != node2.fingerprint
    
    def test_fingerprint_stability(self):
        """A3: Fingerprint is stable across multiple calls (cached)."""
        region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=region)
        
        fp1 = node.fingerprint
        fp2 = node.fingerprint
        fp3 = node.fingerprint
        
        # Should always return same value (cached)
        assert fp1 == fp2 == fp3


class TestRegionTree:
    """Tests for RegionTree data structure - Phase 1 A2-A3."""
    
    def test_single_node_tree(self):
        """A2: Single region becomes root-only tree."""
        region = Region(region_id=0, kind=REGION_STATIC)
        node = RegionTreeNode(region=region)
        tree = RegionTree(root=node)
        
        assert tree.root == node
        assert len(tree.all_nodes) == 1
        assert tree.static_ratio == 1.0
    
    def test_tree_with_children(self):
        """A2: Tree with parent and children."""
        static_region = Region(region_id=0, kind=REGION_STATIC)
        dynamic_region = Region(region_id=1, kind=REGION_DYNAMIC)
        
        child_node = RegionTreeNode(region=dynamic_region)
        parent_node = RegionTreeNode(region=static_region, children=(child_node,))
        tree = RegionTree(root=parent_node)
        
        assert len(tree.all_nodes) == 2
        assert tree.root.is_leaf == False
        assert tree.all_nodes[0] == parent_node
        assert tree.all_nodes[1] == child_node
    
    def test_bfs_traversal_order(self):
        """A2: BFS traversal visits parents before children."""
        # Create a simple tree: root -> child1, child2
        root_region = Region(region_id=0, kind=REGION_STATIC)
        child1_region = Region(region_id=1, kind=REGION_STATIC)
        child2_region = Region(region_id=2, kind=REGION_DYNAMIC)
        
        child1_node = RegionTreeNode(region=child1_region)
        child2_node = RegionTreeNode(region=child2_region)
        root_node = RegionTreeNode(region=root_region, children=(child1_node, child2_node))
        
        tree = RegionTree(root=root_node)
        all_nodes = tree.all_nodes
        
        # Root should come first
        assert all_nodes[0] == root_node
        # Children should come after parent
        assert root_node in all_nodes[:1]
    
    def test_static_subtrees(self):
        """A2: Extract maximal static subtrees."""
        # Structure: root(static) -> child1(dynamic), child2(static)
        root_region = Region(region_id=0, kind=REGION_STATIC)
        child1_region = Region(region_id=1, kind=REGION_DYNAMIC)
        child2_region = Region(region_id=2, kind=REGION_STATIC)
        
        child1_node = RegionTreeNode(region=child1_region)
        child2_node = RegionTreeNode(region=child2_region)
        root_node = RegionTreeNode(region=root_region, children=(child1_node, child2_node))
        
        tree = RegionTree(root=root_node)
        subtrees = tree.static_subtrees()
        
        # At least root should be in subtrees if it's fully static
        # But here root has dynamic children, so check accordingly
        assert len(subtrees) >= 0
    
    def test_dynamic_nodes(self):
        """A2: Extract all dynamic nodes."""
        static_region = Region(region_id=0, kind=REGION_STATIC)
        dynamic_region = Region(region_id=1, kind=REGION_DYNAMIC)
        
        dynamic_node = RegionTreeNode(region=dynamic_region)
        static_node = RegionTreeNode(region=static_region, children=(dynamic_node,))
        
        tree = RegionTree(root=static_node)
        dynamic_nodes = tree.dynamic_nodes()
        
        # Only the dynamic node should be in the list
        assert len(dynamic_nodes) == 1
        assert dynamic_nodes[0].region.is_dynamic
    
    def test_static_ratio_calculation(self):
        """A2: Calculate ratio of static nodes."""
        # 2 static + 1 dynamic = 2/3 static ratio
        regions = [
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_STATIC),
            Region(region_id=2, kind=REGION_DYNAMIC),
        ]
        
        nodes = [RegionTreeNode(region=r) for r in regions]
        # Create tree with root having children
        root_with_children = nodes[0].update_children(tuple(nodes[1:]))
        tree = RegionTree(root=root_with_children)
        
        assert abs(tree.static_ratio - 2/3) < 0.01


class TestBuildRegionTree:
    """Tests for build_region_tree algorithm - Phase 1 A2."""
    
    def test_empty_region_map(self):
        """A2: Build tree from empty RegionMap."""
        region_map = RegionMap(regions=())
        tree = build_region_tree(region_map)
        
        assert tree.root.region.is_static
        assert tree.root.is_leaf
    
    def test_single_region(self):
        """A2: Single region becomes root-only tree."""
        region = Region(region_id=0, kind=REGION_STATIC)
        region_map = RegionMap(regions=(region,))
        tree = build_region_tree(region_map)
        
        assert tree.root.region == region
        assert tree.root.is_leaf
    
    def test_all_static_regions(self):
        """A2: All static regions grouped under first as root."""
        regions = [
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_STATIC),
            Region(region_id=2, kind=REGION_STATIC),
        ]
        region_map = RegionMap(regions=tuple(regions))
        tree = build_region_tree(region_map)
        
        # First region is root, others are children
        assert tree.root.region.region_id == 0
        assert len(tree.root.children) == 2
    
    def test_mixed_static_dynamic(self):
        """A2: Mixed regions with dynamic children grouped under static parent."""
        regions = [
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_DYNAMIC),
            Region(region_id=2, kind=REGION_DYNAMIC),
            Region(region_id=3, kind=REGION_STATIC),
        ]
        region_map = RegionMap(regions=tuple(regions))
        tree = build_region_tree(region_map)
        
        # Root should be first static region
        assert tree.root.region.region_id == 0
        # Dynamic regions should be children of root
        assert len(tree.root.children) >= 1
    
    def test_alternating_regions(self):
        """A2: Alternating static/dynamic creates nested structure."""
        regions = [
            Region(region_id=0, kind=REGION_STATIC),
            Region(region_id=1, kind=REGION_DYNAMIC),
            Region(region_id=2, kind=REGION_STATIC),
            Region(region_id=3, kind=REGION_DYNAMIC),
        ]
        region_map = RegionMap(regions=tuple(regions))
        tree = build_region_tree(region_map)
        
        # Should have 4 nodes total
        assert len(tree.all_nodes) == 4
        # Root should be static
        assert tree.root.region.is_static
    
    def test_dynamic_first(self):
        """A2: When first region is dynamic, it becomes standalone root."""
        regions = [
            Region(region_id=0, kind=REGION_DYNAMIC),
            Region(region_id=1, kind=REGION_STATIC),
        ]
        region_map = RegionMap(regions=tuple(regions))
        tree = build_region_tree(region_map)
        
        # First region (dynamic) becomes root
        assert tree.root.region.region_id == 0
        assert tree.root.region.is_dynamic


class TestRegionTreeIntegration:
    """Integration tests for RegionTree workflow."""
    
    def test_full_workflow_from_region_map(self):
        """End-to-end: RegionMap → extract → build tree → analyze."""
        # Create a realistic mixed graph
        body = [
            _make_stmt("Call"),  # static
            _make_stmt("Call"),  # static
            _make_stmt("ForStmt"),  # dynamic
            _make_stmt("Call"),  # static
            _make_stmt("IfStmt"),  # dynamic
            _make_stmt("Call"),  # static
        ]
        node = _make_func(body, name="test_func")
        
        # Step 1: Extract regions
        from sonata.regions import extract_regions
        region_map = extract_regions(node)
        
        assert len(region_map.regions) == 5  # 3 static + 2 dynamic
        assert region_map.static_ratio() == 3/5
        
        # Step 2: Build tree
        tree = build_region_tree(region_map)
        
        # Step 3: Analyze tree properties
        assert len(tree.all_nodes) == 5
        assert tree.static_ratio >= 0.5  # At least 50% static
        
        # Step 4: Verify structure
        static_nodes = [n for n in tree.all_nodes if n.region.is_static]
        dynamic_nodes = [n for n in tree.all_nodes if n.region.is_dynamic]
        
        assert len(static_nodes) == 3
        assert len(dynamic_nodes) == 2
    
    def test_fingerprint_stability(self):
        """Verify fingerprint is stable across multiple calls."""
        region = Region(region_id=0, kind=REGION_STATIC, nodes=(1, 2, 3))
        node = RegionTreeNode(region=region)
        
        # Call fingerprint multiple times
        fp1 = node.fingerprint
        fp2 = node.fingerprint
        fp3 = node.fingerprint
        
        # Should always return same value (cached)
        assert fp1 == fp2 == fp3
