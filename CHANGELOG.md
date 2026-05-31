# Changelog

All notable changes to Sonata are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering follows the Sonata project convention: `v0.X` milestones
are developed on feature branches and merged to main upon completion.

---

## [v0.9] -- In Progress

**Theme**: Deliverable -- upstream PR readiness.

### Added

- User-facing documentation: README with architecture diagram, quick start,
  and key concepts; CONTRIBUTING guide with development setup and workflow.
- Project changelog covering v0.1 through v0.9.

### Changed

- Package metadata updated for upstream PR submission.

---

## [v0.8] -- API Stability

**Theme**: Version info, deprecation utilities, schema version queries, and
public API audit.

### Added

- `sonata.version` module:
  - `SONATA_VERSION` and `VERSION_INFO` constants for programmatic version
    identification.
  - `version_string()` returning the version with optional library label.
  - `deprecated()` decorator for marking functions as deprecated with
    structured warnings (since, replacement, message).
  - `DeprecatedField` descriptor for deprecating class-level fields with
    first-access warnings.
  - `schema_versions()` aggregating all schema version constants
    (`SCORE_SCHEMA_VERSION`, `FINGERPRINT_VERSION`,
    `ELIGIBILITY_RESULT_SCHEMA_VERSION`, `PLAN_HANDLE_SCHEMA_VERSION`,
    `RUNTIME_CONTRACT_VERSION`, `CACHE_SCHEMA_VERSION`).
  - `public_api()` returning all public symbol names from `sonata.__all__`.
  - `module_api()` returning public symbols grouped by source module.
- Deprecation warning on `Score.runtime_target` field access (migrated to
  `PlanHandle.runtime_target` in v0.2).

### Tests

- `test_version.py` covering version info, deprecation decorators, schema
  version aggregation, and public API audit.

---

## [v0.7] -- Multi-Adapter Support

**Theme**: Adapter registry for multiple PyPTO pipeline stages with
capability-based selection.

### Added

- `sonata.adapters` module:
  - `AdapterCapability` dataclass declaring per-adapter feature flags
    (static_shapes, storage_keys, arg_directions, dependency_kinds,
    control_flow_regions, runtime_scopes).
  - `AdapterDescriptor` dataclass carrying adapter metadata (name, version,
    capabilities, description, certified_dump) with bidirectional dict
    serialization.
  - `AdapterRegistry` class with register, lookup, capability-based select,
    PlanHandle source validation, and dict serialization.
  - Three predefined adapters:
    - `POST_SIMPLIFY` -- post-Simplify stage after CollectCommGroups.
    - `PRE_RUNTIME` -- pre-runtime stage with dependency kind awareness.
    - `POST_SIMPLIFY_WITH_SCOPE` -- post-Simplify with RuntimeScopeStmt
      extraction.
  - `default_registry()` factory returning a pre-populated registry.

### Tests

- `test_adapters.py` covering capability declarations, registry operations,
  capability-based selection, PlanHandle validation, and serialization
  round-trips.

---

## [v0.6] -- Deserialization

**Theme**: JSON/dict to frozen dataclass round-trip deserialization.

### Added

- `sonata.deserialization` module:
  - `DeserializationError` exception for malformed payloads.
  - `score_from_dict()` / `score_from_json()` reconstructing `Score` objects.
  - `plan_handle_from_dict()` / `plan_handle_from_json()` reconstructing
    `PlanHandle` objects.
  - `eligibility_result_from_dict()` / `eligibility_result_from_json()`
    reconstructing `EligibilityResult` objects.
  - Schema version validation on all deserialization entry points.
  - Strict type checking with descriptive error messages for each field.

### Tests

- `test_deserialization.py` covering complete round-trip (serialize then
  deserialize) for Score, PlanHandle, and EligibilityResult; schema version
  mismatch rejection; malformed input handling.

---

## [v0.5] -- Score Cache

**Theme**: Fingerprint-based Score and PlanHandle caching with persistence.

### Added

- `sonata.cache` module:
  - `CacheEntry` frozen dataclass carrying fingerprint, score payload,
    optional PlanHandle payload, schema/fingerprint versions, and timestamps.
  - `ScoreCache` class with:
    - `store()` / `store_plan_handle()` for caching Scores and PlanHandles.
    - `lookup()` / `lookup_plan_handle()` with schema version validation.
    - `contains()`, `invalidate()`, `invalidate_all()` for cache management.
    - `stats()` returning hit/miss/entry-count statistics.
    - `to_dict()` / `from_dict()` for in-memory serialization.
    - `save()` / `load()` for JSON file persistence.
  - `cached_score()` convenience function that looks up or builds a Score,
    returning `(score, fingerprint, was_cached)`.
  - `CACHE_SCHEMA_VERSION` constant.

### Tests

- `test_cache.py` covering store/lookup, invalidation, persistence,
  statistics, PlanHandle association, and the `cached_score` helper.

---

## [v0.4] -- Regionized DAG

**Theme**: Region extraction and per-region eligibility for graphs with
dynamic control flow.

### Added

- `sonata.regions` module:
  - `Region` frozen dataclass with region_id, kind (static/dynamic),
    nodes, optional control_flow_kind, and fallback_reason.
  - `RegionMap` collection with `static_regions()`, `dynamic_regions()`,
    `static_ratio()`, `all_static()`, `all_dynamic()` queries.
  - `extract_regions()` splitting an IR graph into contiguous static and
    dynamic regions. Control flow nodes (ForStmt, IfStmt, WhileStmt) and
    RuntimeScopeStmt create dynamic region boundaries.
  - `check_region_eligibility()` checking eligibility at region granularity.
    Graphs with at least one static region are partially eligible.
  - `REGION_STATIC` and `REGION_DYNAMIC` constants.

### Tests

- `test_regions.py` covering extraction from pure static, pure dynamic, and
  mixed graphs; region boundary classification; per-region eligibility.

---

## [v0.3] -- Storage & Alias Model

**Theme**: Complete storage effect model with alias analysis, liveness,
memory planning, and classified dependency kinds.

### Added

- `sonata.alias` module:
  - `AliasRelation` dataclass with key_a, key_b, and relation kind.
  - `analyze_aliases()` resolving alias / view / inplace / disjoint
    relationships from declarations.
  - `ALIAS_DISJOINT`, `ALIAS_ALIAS`, `ALIAS_VIEW`, `ALIAS_INPLACE` constants.
- `sonata.liveness` module:
  - `BufferLifetime` dataclass with storage_key, birth, death intervals.
  - `StorageConflict` dataclass for overlapping lifetime pairs.
  - `compute_lifetimes()` computing buffer lifetime intervals from task
    args, directions, and storage keys.
  - `find_conflicts()` identifying buffer pairs with overlapping lifetimes.
- `sonata.memory_plan` module:
  - `BufferAllocation` dataclass with storage_key, offset, size.
  - `MemoryPlan` dataclass with allocations and peak_memory.
  - `plan_memory()` implementing greedy-first-fit-by-size 1D strip-packing
    allocation respecting lifetime conflicts.
- `sonata.dependencies` enhancements:
  - `build_dataflow_dependencies()` building conservative RAW/WAW/WAR edges
    from task arg directions and storage keys.
  - `build_ordering_dependencies()` for pure ordering constraints.
  - `build_mixed_dependencies()` combining dataflow and ordering edges.
  - `Dependency.kind` field classifying edges: `data`, `storage`, `war`,
    `ordering`.
  - `supports_dataflow_dependencies()` and
    `dataflow_dependency_fallback_code()` for graceful policy fallback.
- `sonata.directions` module:
  - Direction normalization and classification: `READ_DIRECTIONS`,
    `WRITE_DIRECTIONS`, `IGNORED_DIRECTIONS`, `MEMORY_DIRECTIONS`.

### Tests

- `test_alias.py`, `test_liveness.py`, `test_memory_plan.py`,
  `test_dependency_kind.py`, `test_directions.py`.

---

## [v0.2] -- Runtime Contract

**Theme**: PlanHandle, FuncRegistry, RuntimeArgBinding, and
HostBuildGraphRuntimeAdapter.

### Added

- `sonata.plan_handle` module:
  - `PlanHandle` frozen dataclass bridging Score to a specific runtime target,
    carrying score_fingerprint, runtime_target, source_adapter,
    runtime_contract_version, func_registry, arg_bindings, schema_version,
    and metadata.
  - `FuncRegistry` with name-based lookup, runtime ID binding, and
    `from_score()` factory.
  - `FuncRegistryEntry` mapping function name to sonata_func_id and
    optional runtime_func_id.
  - `RuntimeArgBinding` mapping task args from Sonata storage identity to
    runtime handles.
  - `PLAN_HANDLE_SCHEMA_VERSION` and `RUNTIME_CONTRACT_VERSION` constants.
- `sonata.runtime_adapter` module:
  - `HostBuildGraphRuntimeAdapter` generating and validating
    host-build-graph-shaped task tables and edge contracts from
    Score + PlanHandle.
  - `HostBuildGraphPlan`, `HostBuildGraphTask`, `HostBuildGraphEdge`
    dataclasses for the runtime output format.
  - `RuntimeAdapterResult` with accept/reject factory methods.
  - Structural validation: fingerprint matching, contract version checking,
    func registry consistency, arg binding completeness, edge validity.
- `sonata.serialization` enhancements:
  - `plan_handle_to_dict()` / `plan_handle_to_json()` for PlanHandle
    serialization.

### Tests

- `test_plan_handle.py`, `test_runtime_adapter.py`.

---

## [v0.1] -- Static Plan Completeness

**Theme**: Core Score model, eligibility, serialization, dependencies,
storage keys, and PyPTO adapter smoke test.

### Added

- `sonata.score` module:
  - `Score` frozen dataclass: name, runtime_target, tasks, dependencies,
    shape_assumptions, metadata. Self-validation via `validate()`.
  - `Task` frozen dataclass: task_id, func_id, core_type, args,
    arg_directions, arg_storage_keys, name.
  - `Dependency` frozen dataclass: producer, consumer, kind.
  - `ShapeAssumption` frozen dataclass: symbol, dims.
  - `RuntimeTarget` frozen dataclass: runtime, function_name,
    aicpu_thread_num, config_comment.
  - `EligibilityResult` with accept/reject/accept_with_warnings factories,
    structured `FallbackReason` details, and severity levels.
  - `FallbackReason` with stable code, message, and severity.
- `sonata.eligibility` module:
  - `check_static_eligibility()` performing conservative static checks on
    PyPTO IR nodes via structural introspection.
  - Checks: root kind, control flow, runtime scope, tensor.read,
    entry function, storage coverage, dataflow direction completeness.
- `sonata.serialization` module:
  - `score_to_dict()` / `score_to_json()` for deterministic serialization.
  - `eligibility_result_to_dict()` for eligibility result serialization.
  - `score_fingerprint()` producing stable SHA-256 digests.
  - `SCORE_SCHEMA_VERSION`, `ELIGIBILITY_RESULT_SCHEMA_VERSION`,
    `FINGERPRINT_VERSION` constants.
- `sonata.dependencies` module:
  - `build_dependencies()` with named policy dispatch.
  - `build_sequential_dependencies()` chaining tasks in extraction order.
  - `DEPENDENCY_POLICY_SEQUENTIAL_V0` and `DEPENDENCY_POLICY_DATAFLOW_V0`.
- `sonata.storage` module:
  - Structural storage-key extraction from PyPTO IR assignments.
  - `collect_storage_keys()`, `arg_storage_keys()`,
    `propagate_call_output_storage()`.
  - `STORAGE_COVERAGE_WARN_THRESHOLD` and
    `STORAGE_COVERAGE_REJECT_THRESHOLD` constants.
- `sonata.audit` module:
  - `build_score_metadata()` and `build_task_storage_metadata()` for
    audit/explanatory metadata on Scores.
- `sonata.fallback` module:
  - `FallbackCode` enum with stable reason codes for eligibility rejection,
    validation failure, and runtime adapter errors.
- `sonata.pypto_adapter` module:
  - `PostSimplifyPyPTOInputAdapter` projecting PyPTO IR into Sonata facts
    via structural Python-visible fields (no C++ import).
  - `NormalizedCallFact`, `NormalizedFunctionFact`, `NormalizedTaskFacts`
    dataclasses for adapter output.
  - `PyPTOAdapterContractError` for out-of-scope IR.
- `sonata.__init__` package root with `__all__` exporting all public symbols.

### Tests

- `test_score.py`, `test_eligibility.py`, `test_serialization.py`,
  `test_dependencies.py`, `test_storage.py`, `test_audit.py`,
  `test_fallback.py`, `test_pypto_adapter.py`.
- `tests/pypto_cases/test_eligibility.py` for PyPTO-backed compatibility.

---

## [v0.0] -- Baseline

Initial project structure with Score model skeleton, eligibility stubs,
basic serialization, dependency analysis, storage alias placeholders, audit
metadata, and test suite layout. Single commit: `Initialize Sonata project`.
