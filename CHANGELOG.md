# Changelog

All notable changes to Sonata are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering follows the Sonata project convention: `v0.X` milestones
are developed on feature branches and merged to main upon completion.

---

## [v0.19] -- CI/CD + Production Hardening

### Added
- GitHub Actions workflow for automated test regression.
- Makefile with `test-sonata`, `test-integration`, `test-st`, `test-all` targets.
- Pipeline edge case tests (empty score, None score, empty dependencies).
- Large graph performance benchmarks (200 calls, linear scaling).
- API documentation for `runtime_hook` and `profile` modules.

### Fixed
- Inbox cleanup: runtime-strategy-gap resolved, roadmap_undecided #3/#19 resolved.

---

## [v0.18] -- Runtime Integration + Region Extension + Profile DB

### Added
- **Runtime Hook** (`sonata.runtime_hook`):
  - `apply_sonata_runtime_hints()` optional pre-dispatch hook in PyPTO `execute_compiled()`.
  - Auto-discovers `sonata_plan.json` — no explicit `execute_with_sonata()` needed.
  - `user_block_dim` distinguishes user-supplied from RUNTIME_CONFIG block_dim.
- **Per-Region Eligibility** (`sonata.regions`):
  - `RegionEligibilityResult` and `RegionEligibility` dataclasses.
  - `check_region_eligibility()` populates per-region breakdown.
  - Region-level guards are independent.
- **Static-Enumerable Control Flow** (`sonata.eligibility`):
  - `_is_unrollable_for_stmt()`: constant-trip-count ForStmt detection (<=16).
  - `expand_for_stmt()` / `expand_task_graph()`: loop body expansion.
- **Profile Database** (`sonata.profile`):
  - `OperatorProfile` dataclass with Welford's incremental mean/std.
  - `ProfileDatabase` with record/lookup/save/load.
  - `compute_scheduling_instructions()` accepts optional `profile_db`.
  - `collect_task_timings()` for post-execution timing collection.
- **Memory Planning** in `sonata_analyze()`:
  - `sonata_plan.json` now includes `memory_plan` with `peak_memory` and `allocations`.

### Changed
- ST conftest no longer monkeypatches `execute_compiled` / `execute_on_device`.
- Runner hook is the single runtime integration path.

---

## [v0.17] -- Guard Hardening + StorageEffect + Bug Fixes

### Added
- **Guard Statistics** in `sonata_plan.json`:
  - `guard_stats` with `shape_assumption_count`, `unique_symbols`, `guard_density`.
  - Warning when `guard_density > 8` (TorchDynamo reference).
- **STALE Guard Status**:
  - `GuardStatus.STALE` for two-level invalidation (plan handle invalid, Score valid).
  - `GuardDetail` dataclass for per-guard evaluation info.
  - `check_guards_at_runtime()` returns structured results with STALE semantics.
- **StorageEffect Model** (`sonata.score`):
  - `StorageEffect` dataclass (`buffer_id`, `kind`).
  - `Task.storage_effects` optional field.
  - `derive_storage_effects()` from arg_directions + arg_storage_keys.
  - Serialization/deserialization round-trip support.

### Fixed
- `ShapeAssumption.__eq__/__hash__` now includes `dims` (was comparing only symbol+severity).
- `plan_handle_from_dict()` now reads `guard_status` and `critical_guards`.
- `_compute_fingerprint()` uses `score_fingerprint()` instead of counts.
- `GreedySolver.solve()` respects `device_memory_limit`.
- `score_to_dict()` severity serialization uses `getattr(.value)`.

### Performance
- IR tree walk cache in eligibility (`_walk_cache`).
- Performance benchmarks at qwen3 scale.

---

## [v0.16] -- Broader pypto-lib Coverage

### Added
- SPMD eligibility via region-based fallback.
- LLM model analysis (qwen3=49 tasks, deepseek=46 tasks).
- End-to-end simpler execution verification.

---

## [v0.15] -- Deeper Runtime Integration

### Added
- Memory offset injection (`write_memory_hints`).
- Region-aware scheduling (block_dim by region type).
- Guard cache invalidation.

---

## [v0.14] -- Runtime Integration + pypto-lib Validation

### Added
- Sonata influences runtime via `execute_on_device` block_dim.
- pypto-lib example + LLM model integration tests.

---

## [v0.13] -- Performance & Dependency Kind

### Added
- `DependencyKind` enum in `dependencies.py`.
- Certified IR cache.
- Performance benchmarks.

---

## [v0.12] -- Deep PyPTO Pipeline Integration

### Added
- Sonata-integrated st tests (D0-D7).
- `sonata_plan.json` schema and serialization.
- Region-aware execution dispatcher.

---

## [v0.11] -- Region Tree & Memory Optimization

### Added
- `RegionTree` with per-region fingerprints.
- Memory planning (conflict matrix + GreedySolver).
- Real PyPTO IR integration via `PostSimplifyPyPTOInputAdapter`.

---

## [v0.10] -- Guard Condition Abstraction

**Theme**: Unified guard condition system with soft/hard severity classification,
over-guarding mitigation, and Score cache integration.

### Added

- **GuardCondition Abstraction** (`sonata.guard` module):
  - `GuardCondition` ABC as unified interface for all guard types.
  - `ShapeAssumption` refactored as `GuardCondition` subclass (backward compatible).
  - `GuardSeverity` enum with `GUARD_SEVERITY_HARD` and `GUARD_SEVERITY_SOFT`.
  - `GuardEvaluator` interface for runtime guard evaluation.
  - `GuardInvalidator` strategy implementation with severity-based policy.
  - `InvalidateAction` enum: `REPLAN`, `INVALIDATE_HANDLE`, `UPDATE_IN_PLACE`.
  - `GuardSelector` ABC and `EntryParamGuardSelector` Top-K selection strategy.
  - `check_guard_density()` utility for over-guarding detection (threshold: 50 guards).
  
- **Score Cache Enhancement** (`sonata.cache` module):
  - `CacheEntry.guard_status` field tracking guard satisfaction state.
  - `ScoreCache.lookup()` validates guard status, treats violations as cache misses.
  - `ScoreCache.lookup_plan_handle()` includes guard validation.
  - `ScoreCache.contains()` checks guard status before returning True.
  - Backward compatible: existing cache entries load with default `ALL_SATISFIED`.
  - Serialization updated to support `guard_status` in JSON persistence.
  
- **PlanHandle Guard Integration** (`sonata.plan_handle` module):
  - `PlanHandle.guard_status` field for runtime guard evaluation state.
  - `PlanHandle.critical_guards` field tracking critical guard conditions.
  - `GuardStatus` enum: `ALL_SATISFIED`, `PARTIAL_FAILED`, `ALL_FAILED`.
  - `PlanHandle.from_score()` initializes guard status to `ALL_SATISFIED`.
  
- **Migration Path Tools**:
  - `shape_assumption_to_guard_condition()` conversion function (with deprecation warning).
  - `deprecated_shape_assumption()` helper with detailed migration guidance.
  - User-facing migration guide: `docs/user-guide/guards-migration.md` (~430 lines).
  - Compatibility test suite: `test_migration_compatibility.py` (4 tests).
  
- **Performance Benchmark**:
  - `bench_cache_guard_overhead.py`: Measures guard checking overhead (< 5% confirmed).
  - Results saved to `benchmarks/results/cache_guard_overhead_results.json`.

### Changed

- **Deprecation Warnings**:
  - `ShapeAssumption` now emits `DeprecationWarning` when used without explicit severity.
  - Migration guidance included in warning message with link to user guide.
  - `Score.runtime_target` deprecation warnings active throughout codebase.

- **Cache Behavior**:
  - Conservative invalidation: both `PARTIAL_FAILED` and `ALL_FAILED` treated as cache misses.
  - Rationale: Safety first, prevents subtle bugs from partial violations.

### Tests

- `test_guard.py`: 39 comprehensive tests covering Phase 1-4 features.
- `test_cache.py`: 30 tests including guard-aware lookup scenarios.
- `test_plan_handle.py`: Updated to include guard status fields.
- `test_migration_compatibility.py`: 4 tests for ShapeAssumption → GuardCondition conversion.

### Performance

- Guard checking overhead benchmark results:
  - Baseline (ALL_SATISFIED): ~3.9 µs per lookup
  - ALL_FAILED: ~3.8 µs per lookup (-2.4% overhead)
  - PARTIAL_FAILED: ~3.5 µs per lookup (-11.7% overhead)
- Conclusion: Guard checking adds negligible performance cost (< 5%).

### Documentation

- New user guide: `guards-migration.md` (~430 lines) covering:
  - Quick start migration examples
  - Detailed step-by-step instructions
  - API reference for GuardCondition system
  - Common patterns and troubleshooting
  - Performance considerations
- Updated roadmap documentation to reflect v0.10 completion.
- Version bumped to `SONATA_VERSION = "0.10.0"`.

### Backward Compatibility

- ✅ All existing code continues to work without modifications.
- ✅ `ShapeAssumption` remains functional but emits deprecation warnings.
- ✅ Existing cache files without `guard_status` field load correctly (default: ALL_SATISFIED).
- ⚠️ **Breaking Change in v0.11**: `ShapeAssumption` will be removed; full migration required.

---

## [v0.9] -- Deliverable Milestone

**Theme**: Upstream PR readiness with complete documentation and test coverage.

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
