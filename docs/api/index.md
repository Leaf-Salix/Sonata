# Sonata API Reference

Sonata is a host-side static scheduling compiler layer for PyPTO. For static-shape subgraphs it generates a pre-computed task dependency calendar (Score) at compile time, consumed by the `host_build_graph` runtime instead of AICPU dynamic scheduling.

## Modules

| Module | Description |
|--------|-------------|
| [Score](score.md) | Core data model: Score, Task, Dependency, ShapeAssumption, RuntimeTarget, EligibilityResult, FallbackReason. |
| [Eligibility](eligibility.md) | Conservative static-eligibility checks that determine whether an IR graph can use Sonata planning. |
| [Serialization](serialization.md) | Stable JSON-like serialization and deserialization for Score, PlanHandle, and EligibilityResult. |
| [Dependencies](dependencies.md) | Dependency policy builders: sequential, dataflow (RAW/WAW/WAR), mixed, and pure ordering edges. |
| [Storage / Alias / Liveness](storage-alias.md) | Storage-key extraction, alias analysis, buffer liveness, and graph-level memory planning. |
| [Cache](cache.md) | Fingerprint-based Score and PlanHandle cache to avoid repeated eligibility checks. |
| [PlanHandle / Runtime](plan-handle.md) | Runtime artifact key types bridging Score to a specific runtime target and function registry. |
| [Adapters](adapters.md) | Multi-adapter registry for PyPTO pipeline stages with capability-based selection. |
| [Version](version.md) | Version constants, deprecation utilities, schema version introspection, and API audit helpers. |
