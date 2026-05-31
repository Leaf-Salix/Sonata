# Serialization Module

Source: `src/sonata/serialization.py`, `src/sonata/deserialization.py`

Stable JSON-like serialization and deserialization helpers for Sonata scores, plan handles, and eligibility results.

---

## Schema Version Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SCORE_SCHEMA_VERSION` | `1` | Schema version for Score serialization. |
| `ELIGIBILITY_RESULT_SCHEMA_VERSION` | `1` | Schema version for EligibilityResult serialization. |
| `FINGERPRINT_VERSION` | `1` | Version tag embedded in fingerprint payloads. |

---

## Serialization Functions

### `score_to_dict`

```python
def score_to_dict(score: Score) -> dict[str, Any]
```

Return a deterministic JSON-like dictionary for `score`. Includes `schema_version`, `name`, `runtime_target`, `tasks`, `dependencies`, `shape_assumptions`, and `metadata`.

### `score_to_json`

```python
def score_to_json(score: Score, *, indent: int | None = 2) -> str
```

Return a stable JSON string for `score`. Uses `sort_keys=True` for deterministic output.

### `score_from_dict`

```python
def score_from_dict(data: dict[str, Any]) -> Score
```

Reconstruct a `Score` from a dictionary produced by `score_to_dict`. Raises `DeserializationError` on schema mismatch or malformed data.

### `score_from_json`

```python
def score_from_json(text: str) -> Score
```

Reconstruct a `Score` from a JSON string produced by `score_to_json`.

### `score_fingerprint`

```python
def score_fingerprint(score: Score, *, include_metadata: bool = False) -> str
```

Return a stable SHA-256 fingerprint for the score computation identity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score` | `Score` | *(required)* | The score to fingerprint. |
| `include_metadata` | `bool` | `False` | When `True`, includes `Score.metadata` audit/debug data. Does not include `runtime_target`. |

### `plan_handle_to_dict`

```python
def plan_handle_to_dict(plan_handle: PlanHandle) -> dict[str, Any]
```

Return a deterministic JSON-like dictionary for `plan_handle`. Includes `schema_version`, `score_fingerprint`, `runtime_target`, `source_adapter`, `runtime_contract_version`, `func_registry`, `arg_bindings`, and `metadata`.

### `plan_handle_to_json`

```python
def plan_handle_to_json(plan_handle: PlanHandle, *, indent: int | None = 2) -> str
```

Return a stable JSON string for `plan_handle`.

### `eligibility_result_to_dict`

```python
def eligibility_result_to_dict(result: EligibilityResult) -> dict[str, Any]
```

Return a deterministic JSON-like dictionary for an eligibility result. Includes `schema_version`, `eligible`, `reasons`, `reason_details`, and optionally `score`.

---

## Deserialization Functions

Source: `src/sonata/deserialization.py`

### `plan_handle_from_dict`

```python
def plan_handle_from_dict(data: dict[str, Any]) -> PlanHandle
```

Reconstruct a `PlanHandle` from a dictionary produced by `plan_handle_to_dict`.

### `plan_handle_from_json`

```python
def plan_handle_from_json(text: str) -> PlanHandle
```

Reconstruct a `PlanHandle` from a JSON string.

### `eligibility_result_from_dict`

```python
def eligibility_result_from_dict(data: dict[str, Any]) -> EligibilityResult
```

Reconstruct an `EligibilityResult` from a dictionary.

### `eligibility_result_from_json`

```python
def eligibility_result_from_json(text: str) -> EligibilityResult
```

Reconstruct an `EligibilityResult` from a JSON string.

---

## Exceptions

### `DeserializationError`

```python
class DeserializationError(Exception)
```

Raised when a serialized payload cannot be reconstructed. Covers schema version mismatches, missing required fields, and type errors.
