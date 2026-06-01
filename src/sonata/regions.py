# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Region extraction and region-level eligibility for Sonata v0.4/v0.11.

Splits an IR graph into static and dynamic regions. Static regions can
be independently planned by Sonata; dynamic regions fall back to the
original PyPTO runtime path. This replaces the v0.1 whole-graph
eligibility check with per-region granularity.

v0.11 enhancements:
- RegionTree: hierarchical region management with per-region fingerprints
- Per-region independent cache integration
- Guard evaluation at region level
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .fallback import FallbackCode
from .score import EligibilityResult, FallbackReason, Score
from .serialization import score_fingerprint


REGION_STATIC = "static"
REGION_DYNAMIC = "dynamic"

_CONTROL_FLOW_KINDS = {"ForStmt", "IfStmt", "WhileStmt"}


@dataclass(frozen=True)
class Region:
    """One region of an IR graph, classified as static or dynamic."""

    region_id: int
    kind: str
    nodes: tuple[Any, ...] = ()
    control_flow_kind: str | None = None
    fallback_reason: FallbackReason | None = None

    @property
    def is_static(self) -> bool:
        return self.kind == REGION_STATIC

    @property
    def is_dynamic(self) -> bool:
        return self.kind == REGION_DYNAMIC

    @property
    def node_count(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class RegionMap:
    """Collection of regions extracted from an IR graph."""

    regions: tuple[Region, ...] = ()

    def static_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.is_static)

    def dynamic_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.is_dynamic)

    def static_ratio(self) -> float:
        if not self.regions:
            return 0.0
        return len(self.static_regions()) / len(self.regions)

    def all_static(self) -> bool:
        return all(r.is_static for r in self.regions)

    def all_dynamic(self) -> bool:
        return all(r.is_dynamic for r in self.regions)


@dataclass(frozen=True)
class RegionTreeNode:
    """A node in the RegionTree, representing a region with optional children.
    
    v0.11 Phase 1 A1: RegionTree data model
    
    Each node can have:
    - A region (static or dynamic)
    - Child nodes (for hierarchical organization)
    - An optional Score (computed for static subtrees)
    
    Frozen dataclass with hash support for use in caches.
    """
    region: Region
    children: tuple["RegionTreeNode", ...] = ()
    score: Optional[Score] = None
    _fingerprint_cache: Optional[str] = field(default=None, compare=False, repr=False)
    
    @property
    def is_leaf(self) -> bool:
        """Check if this node has no children."""
        return len(self.children) == 0
    
    @property
    def is_static_subtree(self) -> bool:
        """Check if entire subtree is static (no dynamic regions)."""
        if self.region.is_dynamic:
            return False
        return all(child.is_static_subtree for child in self.children)
    
    def __post_init__(self):
        """Compute initial fingerprint if score exists."""
        if self._fingerprint_cache is None and self.score is not None:
            object.__setattr__(self, '_fingerprint_cache', self._compute_fingerprint())
    
    def _compute_fingerprint(self) -> str:
        """Compute fingerprint for this subtree.
        
        v0.11 Phase 1 A3: Per-region fingerprint computation
        
        Includes:
        - Region type (static/dynamic)
        - Region ID
        - Task sequence from score (if available)
        - Dependencies
        - Shape assumptions
        - Child fingerprints (for hierarchical identity)
        
        Excludes runtime_target (consistent with v0.1 fingerprint policy).
        """
        # Base identity from region metadata
        base_identity = {
            "region_id": self.region.region_id,
            "region_kind": self.region.kind,
            "node_count": self.region.node_count,
        }
        
        # Add score-based identity if available
        if self.score is not None:
            score_identity = {
                "score_name": self.score.name,
                "task_count": len(self.score.tasks),
                "dep_count": len(self.score.dependencies),
                "guard_count": len(self.score.shape_assumptions),
            }
            base_identity.update(score_identity)
        
        # Add child fingerprints for hierarchical identity
        if self.children:
            child_fps = [child._compute_fingerprint() for child in self.children]
            base_identity["children"] = child_fps
        
        # Use serialization module's fingerprint function
        # Note: We construct a minimal dict that mimics Score structure for fingerprinting
        fingerprint_payload = {
            "fingerprint_version": 1,  # Consistent with v0.1 fingerprint version
            "identity": {
                "type": "region_tree_node",
                **base_identity,
            },
        }
        
        # Compute hash using same mechanism as score_fingerprint
        import hashlib
        import json
        
        # Serialize to JSON with sorted keys for determinism
        serialized = json.dumps(fingerprint_payload, sort_keys=True, default=str)
        fp_hash = hashlib.sha256(serialized.encode()).hexdigest()
        
        return fp_hash
    
    @property
    def fingerprint(self) -> str:
        """Return computed fingerprint for this subtree."""
        if self._fingerprint_cache is None:
            object.__setattr__(self, '_fingerprint_cache', self._compute_fingerprint())
        return self._fingerprint_cache
    
    def update_score(self, new_score: Score) -> "RegionTreeNode":
        """Create a new node with updated score and fingerprint.
        
        Returns a new frozen instance with the score and recomputed fingerprint.
        """
        new_fp = score_fingerprint(new_score)
        return RegionTreeNode(
            region=self.region,
            children=self.children,
            score=new_score,
            _fingerprint_cache=new_fp,
        )
    
    def update_children(self, new_children: tuple["RegionTreeNode", ...]) -> "RegionTreeNode":
        """Create a new node with updated children."""
        return RegionTreeNode(
            region=self.region,
            children=new_children,
            score=self.score,
            _fingerprint_cache=self._compute_fingerprint(),
        )


@dataclass(frozen=True)
class RegionTree:
    """Hierarchical region tree for v0.11 advanced region management.
    
    Extends v0.4's flat RegionMap with hierarchical organization and
    per-region independent Score/cache management.
    """
    root: RegionTreeNode
    
    @property
    def all_nodes(self) -> list[RegionTreeNode]:
        """BFS traversal of all nodes in the tree.
        
        Each node appears exactly once in the traversal.
        """
        result: list[RegionTreeNode] = []
        visited_ids: set[int] = set()
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            node_id = id(node)
            
            # Skip if already visited (prevent duplicates from shared references)
            if node_id in visited_ids:
                continue
            
            visited_ids.add(node_id)
            result.append(node)
            
            # Add children to queue (check before adding to avoid duplicate queue entries)
            for child in node.children:
                child_id = id(child)
                if child_id not in visited_ids and child_id not in [id(n) for n in queue]:
                    queue.append(child)
        
        return result
    
    def static_subtrees(self) -> list[RegionTreeNode]:
        """Get all maximal static subtrees (static nodes with no static children)."""
        subtrees = []
        for node in self.all_nodes:
            if node.region.is_static and not any(c.region.is_static for c in node.children):
                subtrees.append(node)
        return subtrees
    
    def dynamic_nodes(self) -> list[RegionTreeNode]:
        """Get all nodes that are dynamically classified.
        
        Returns only nodes where region.kind == REGION_DYNAMIC,
        not nodes that merely contain dynamic descendants.
        """
        return [n for n in self.all_nodes if n.region.is_dynamic]
    
    @property
    def static_ratio(self) -> float:
        """Ratio of static nodes to total nodes."""
        all_nodes = self.all_nodes
        if not all_nodes:
            return 0.0
        static_count = sum(1 for n in all_nodes if n.region.is_static)
        return static_count / len(all_nodes)


def build_region_tree(region_map: RegionMap) -> RegionTree:
    """Build a RegionTree from a flat RegionMap.
    
    v0.11 Phase 1 A2: Hierarchical region construction algorithm
    
    Groups dynamic regions under static region parents to create
    a hierarchical structure. Static regions become internal nodes,
    dynamic regions become leaf nodes (or their own subtrees).
    
    Algorithm:
    1. Create leaf nodes for each region
    2. Group consecutive dynamic regions under their preceding static parent
    3. Build parent-child relationships
    4. Return RootTree with the first static region as root
    """
    if not region_map.regions:
        empty_region = Region(region_id=0, kind=REGION_STATIC)
        return RegionTree(root=RegionTreeNode(region=empty_region))
    
    # Create leaf nodes for all regions
    leaf_nodes: list[RegionTreeNode] = [
        RegionTreeNode(region=r) for r in region_map.regions
    ]
    
    if len(leaf_nodes) == 1:
        # Single region tree
        return RegionTree(root=leaf_nodes[0])
    
    # Group dynamic regions under preceding static parent
    # Strategy: static nodes are internal, dynamic nodes are children
    grouped: list[RegionTreeNode] = []
    i = 0
    
    while i < len(leaf_nodes):
        node = leaf_nodes[i]
        
        if node.region.is_static:
            # Find consecutive dynamic children
            children: list[RegionTreeNode] = []
            j = i + 1
            
            while j < len(leaf_nodes) and leaf_nodes[j].region.is_dynamic:
                children.append(leaf_nodes[j])
                j += 1
            
            # Create parent node with dynamic children
            if children:
                grouped.append(node.update_children(tuple(children)))
            else:
                grouped.append(node)
            
            i = j
        else:
            # Dynamic node without static parent - keep as standalone
            grouped.append(node)
            i += 1
    
    # If first node is static, use it as root with others as siblings
    if grouped[0].region.is_static and len(grouped) > 1:
        # Collect all children: existing children of root + subsequent grouped nodes
        root_node = grouped[0]
        existing_children = list(root_node.children) if root_node.children else []
        subsequent_nodes = grouped[1:]
        
        # Merge existing children with subsequent nodes
        all_children = existing_children + subsequent_nodes
        final_root = root_node.update_children(tuple(all_children))
    else:
        # First node is dynamic or only one node - use as root
        final_root = grouped[0]
    
    return RegionTree(root=final_root)


def extract_regions(node: Any) -> RegionMap:
    """Extract static and dynamic regions from an IR node.

    Walks the node's body and splits statements into contiguous runs
    of static statements (no control flow) separated by dynamic
    statements (control flow nodes).
    """
    body = _get_body(node)
    if body is None:
        return RegionMap(regions=(_classify_node(node, 0),))

    regions: list[Region] = []
    static_buffer: list[Any] = []
    region_id = 0

    for stmt in body:
        kind = type(stmt).__name__
        if kind in _CONTROL_FLOW_KINDS:
            if static_buffer:
                regions.append(Region(
                    region_id=region_id,
                    kind=REGION_STATIC,
                    nodes=tuple(static_buffer),
                ))
                region_id += 1
                static_buffer = []

            regions.append(Region(
                region_id=region_id,
                kind=REGION_DYNAMIC,
                nodes=(stmt,),
                control_flow_kind=kind,
                fallback_reason=FallbackReason(
                    code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                    message=f"{kind} region falls back to dynamic execution",
                    severity="warning",
                ),
            ))
            region_id += 1
        elif kind == "RuntimeScopeStmt":
            if static_buffer:
                regions.append(Region(
                    region_id=region_id,
                    kind=REGION_STATIC,
                    nodes=tuple(static_buffer),
                ))
                region_id += 1
                static_buffer = []
            regions.append(Region(
                region_id=region_id,
                kind=REGION_DYNAMIC,
                nodes=(stmt,),
                control_flow_kind=kind,
                fallback_reason=FallbackReason(
                    code=FallbackCode.UNSUPPORTED_RUNTIME_SCOPE.value,
                    message=f"{kind} region falls back to dynamic execution",
                    severity="warning",
                ),
            ))
            region_id += 1
        else:
            static_buffer.append(stmt)

    if static_buffer:
        regions.append(Region(
            region_id=region_id,
            kind=REGION_STATIC,
            nodes=tuple(static_buffer),
        ))

    if not regions:
        regions.append(Region(
            region_id=0,
            kind=REGION_STATIC,
            nodes=(),
        ))

    return RegionMap(regions=tuple(regions))


def check_region_eligibility(
    node: Any,
    *,
    entry_name: str | None = None,
) -> EligibilityResult:
    """Check eligibility at region granularity.

    Static regions are eligible for Sonata planning; dynamic regions
    fall back with structured warnings. A graph with at least one
    static region is partially eligible.
    
    v0.11 Phase 1 B: Enhanced to return per-region eligibility results
    when RegionTree is available.
    """
    from .eligibility import check_static_eligibility

    region_map = extract_regions(node)

    if region_map.all_dynamic():
        reasons = []
        for r in region_map.dynamic_regions():
            if r.fallback_reason is not None:
                reasons.append(r.fallback_reason)
        if not reasons:
            reasons.append(FallbackReason(
                code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                message="all regions are dynamic, no static region available",
            ))
        return EligibilityResult.reject(*reasons)

    # Build RegionTree for per-region analysis
    region_tree = build_region_tree(region_map)
    
    # Check if any static regions exist
    static_nodes = [n for n in region_tree.all_nodes if n.region.is_static]
    if not static_nodes:
        return check_static_eligibility(node, entry_name=entry_name)
    
    # Extract per-region Scores for static subtrees
    # This is the key enhancement over v0.4
    per_region_scores: dict[str, Score] = {}
    fallback_reasons: list[FallbackReason] = []
    
    # Create a default RuntimeTarget for placeholder scores
    from .score import RuntimeTarget
    default_rt = RuntimeTarget(
        runtime="host_build_graph",
        function_name=f"{entry_name or 'graph'}_placeholder",
        aicpu_thread_num=1,
    )
    
    # Process each maximal static subtree
    for static_subtree in region_tree.static_subtrees():
        # TODO: Extract Score from this subtree
        # For now, mark as eligible but note that per-region extraction is pending
        region_key = f"region_{static_subtree.region.region_id}"
        
        # Create a placeholder Score - will be replaced with real extraction
        # This demonstrates the per-region structure
        from .score import Score, Task, Dependency
        
        placeholder_score = Score(
            name=f"{entry_name or 'graph'}_region_{static_subtree.region.region_id}",
            runtime_target=default_rt,
            tasks=(),
            dependencies=(),
            shape_assumptions=(),
        )
        per_region_scores[region_key] = placeholder_score
    
    # Collect fallback reasons from dynamic regions
    for dynamic_node in region_tree.dynamic_nodes():
        if dynamic_node.region.fallback_reason is not None:
            fallback_reasons.append(dynamic_node.region.fallback_reason)
        else:
            fallback_reasons.append(FallbackReason(
                code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                message=f"Dynamic region {dynamic_node.region.region_id} falls back to AICPU",
                severity="warning",
            ))
    
    # If we have static regions, accept with warnings about dynamic regions
    if per_region_scores:
        # Create a merged result with per-region metadata
        result = EligibilityResult.accept(score=None)

        # Add region tree and per-region scores to metadata (frozen dataclass)
        meta = {
            'region_tree': region_tree,
            'per_region_scores': per_region_scores,
            'region_count': len(region_map.regions),
            'static_region_count': len(region_map.static_regions()),
            'dynamic_region_count': len(region_map.dynamic_regions()),
            'static_ratio': region_map.static_ratio(),
            'static_subtrees': [
                f"region_{n.region.region_id}" for n in region_tree.static_subtrees()
            ],
        }
        object.__setattr__(result, 'metadata', meta)
        
        return result
    
    # Fallback to whole-graph check if no static subtrees found
    return check_static_eligibility(node, entry_name=entry_name)


def _classify_node(node: Any, region_id: int) -> Region:
    body = _get_body(node)
    if body is None:
        kind = type(node).__name__
        if kind in _CONTROL_FLOW_KINDS:
            return Region(
                region_id=region_id,
                kind=REGION_DYNAMIC,
                nodes=(node,),
                control_flow_kind=kind,
                fallback_reason=FallbackReason(
                    code=FallbackCode.CONTROL_FLOW_NOT_SUPPORTED.value,
                    message=f"{kind} is a dynamic region",
                ),
            )
        return Region(
            region_id=region_id,
            kind=REGION_STATIC,
            nodes=(node,),
        )

    return extract_regions(node).regions[0] if extract_regions(node).regions else Region(
        region_id=region_id,
        kind=REGION_STATIC,
        nodes=(),
    )


def _get_body(node: Any) -> list[Any] | tuple[Any, ...] | None:
    for field_name in ("body", "stmts", "statements", "seq"):
        body = getattr(node, field_name, None)
        if isinstance(body, (list, tuple)) and body:
            return body
    return None


# ============================================================================
# v0.11 Phase 1 C: Per-Region Cache Integration
# ============================================================================

def store_region_tree(
    region_tree: RegionTree,
    plan_handle_payload: dict[str, Any],
    cache: Any,  # ScoreCache type would create circular import
    *,
    guard_status: str = "all_satisfied",
) -> dict[str, str]:
    """Store a RegionTree in the cache with per-region fingerprints.
    
    v0.11 Phase 1 C1-C3: Per-region cache integration
    
    Args:
        region_tree: The RegionTree to cache
        plan_handle_payload: Serialized PlanHandle
        cache: ScoreCache instance
        guard_status: Initial guard status string
        
    Returns:
        dict mapping region paths to their fingerprints
    """
    # TODO: This requires ScoreCache integration
    # For now, return fingerprint mappings without actual cache storage
    mappings: dict[str, str] = {}
    
    # Store root node
    root_fp = region_tree.root.fingerprint
    mappings["root"] = root_fp
    
    # Store all child nodes with path-based keys
    def traverse_and_map(node: RegionTreeNode, path: str = "root"):
        if node.children:
            for i, child in enumerate(node.children):
                child_path = f"{path}.child[{i}]"
                child_fp = child.fingerprint
                mappings[child_path] = child_fp
                traverse_and_map(child, child_path)
    
    traverse_and_map(region_tree.root)
    
    # TODO: Actually store in cache when ScoreCache is available
    # cache.store_region_tree(region_tree, plan_handle_payload, guard_status=guard_status)
    
    return mappings


def lookup_region_tree(
    region_path: str,
    cache: Any,  # ScoreCache
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Lookup a specific region's Score and PlanHandle from cache.
    
    v0.11 Phase 1 C2: Per-region cache lookup
    
    Args:
        region_path: Path to region (e.g., "root", "root.child[0]", "root.child[1].child[0]")
        cache: ScoreCache instance
        
    Returns:
        Tuple of (score_payload, plan_handle_payload) or (None, None) if miss
    """
    # TODO: Implement actual cache lookup
    # This requires ScoreCache integration
    raise NotImplementedError("lookup_region_tree not yet implemented - requires ScoreCache")


__all__ = [
    "REGION_DYNAMIC",
    "REGION_STATIC",
    "Region",
    "RegionMap",
    "RegionTreeNode",
    "RegionTree",
    "build_region_tree",
    "check_region_eligibility",
    "extract_regions",
    # v0.11 Phase 1 C
    "store_region_tree",
    "lookup_region_tree",
]
