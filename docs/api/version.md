# Version Module

Source: `src/sonata/version.py`

Sonata version information, deprecation utilities, schema version introspection, and API audit helpers.

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SONATA_VERSION` | `"0.8.0"` | Current Sonata version string. |
| `VERSION_INFO` | `(0, 8, 0)` | Version as a `(major, minor, patch)` tuple. |

---

## Functions

### `version_string`

```python
def version_string(*, include_label: bool = False) -> str
```

Return the Sonata version string.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_label` | `bool` | `False` | When `True`, prepends the library name: `"Sonata 0.8.0"`. |

---

### `deprecated`

```python
def deprecated(
    message: str,
    *,
    since: str = "",
    replacement: str = "",
) -> Callable
```

Decorator that marks a function as deprecated. Emits a `DeprecationWarning` on every call and preserves the original function's behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | *(required)* | Explanation of why the function is deprecated. |
| `since` | `str` | `""` | Version when the deprecation was introduced. |
| `replacement` | `str` | `""` | Name of the replacement function or API. |

Sets `__deprecated__` and `__deprecated_message__` attributes on the wrapped function.

**Example:**

```python
@deprecated("No longer needed after v0.2 refactor.", since="0.2", replacement="PlanHandle.from_score")
def old_build_plan(score):
    ...
```

---

### `schema_versions`

```python
def schema_versions() -> dict[str, int]
```

Return all Sonata schema version constants as a dictionary.

| Key | Source |
|-----|--------|
| `"score_schema"` | `SCORE_SCHEMA_VERSION` |
| `"fingerprint_version"` | `FINGERPRINT_VERSION` |
| `"eligibility_result_schema"` | `ELIGIBILITY_RESULT_SCHEMA_VERSION` |
| `"plan_handle_schema"` | `PLAN_HANDLE_SCHEMA_VERSION` |
| `"runtime_contract"` | `RUNTIME_CONTRACT_VERSION` |
| `"cache_schema"` | `CACHE_SCHEMA_VERSION` |

---

### `public_api`

```python
def public_api() -> list[str]
```

Return all public symbol names exported by the `sonata` package (from `sonata.__all__`), sorted alphabetically.

---

### `module_api`

```python
def module_api() -> dict[str, list[str]]
```

Return public symbols grouped by source module. Keys are short module names (e.g. `"score"`, `"cache"`, `"serialization"`). Values are sorted lists of public symbol names.

---

## Classes

### `DeprecatedField`

```python
class DeprecatedField:
    def __init__(
        self,
        default: Any = None,
        *,
        message: str = "",
        since: str = "",
        replacement: str = "",
    ) -> None
```

Descriptor that emits a `DeprecationWarning` on first access. Works on regular (mutable) classes. For frozen dataclasses, use the `_deprecated_field_warnings` / `__getattribute__` protocol instead (see `Score`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default` | `Any` | `None` | Default value when the field is not explicitly set. |
| `message` | `str` | `""` | Explanation of the deprecation. |
| `since` | `str` | `""` | Version when the deprecation was introduced. |
| `replacement` | `str` | `""` | Name of the replacement field. |

**Example:**

```python
class Config:
    old_option = DeprecatedField(
        default=42,
        message="Use new_option instead.",
        since="0.8",
    )
```

The warning is emitted only on the first access per instance. Subsequent accesses return the value silently.
