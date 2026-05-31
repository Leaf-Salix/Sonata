# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata version information, deprecation utilities, and API audit helpers."""

import functools
import warnings
from typing import Any, Callable

SONATA_VERSION = "0.8.0"
VERSION_INFO = (0, 8, 0)


def version_string(*, include_label: bool = False) -> str:
    """Return the Sonata version string.

    With ``include_label=True``, appends the library name:
    ``"Sonata 0.8.0"``.
    """
    if include_label:
        return f"Sonata {SONATA_VERSION}"
    return SONATA_VERSION


def deprecated(
    message: str,
    *,
    since: str = "",
    replacement: str = "",
) -> Callable:
    """Decorator that marks a function as deprecated.

    Emits a :class:`DeprecationWarning` on first call and preserves the
    original function's behavior.
    """
    def decorator(func: Callable) -> Callable:
        parts = [f"{func.__qualname__} is deprecated"]
        if since:
            parts.append(f"since v{since}")
        if message:
            parts.append(f"— {message}")
        if replacement:
            parts.append(f"Use {replacement} instead.")
        warning_text = ". ".join(parts)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(warning_text, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True
        wrapper.__deprecated_message__ = warning_text
        return wrapper
    return decorator


def schema_versions() -> dict[str, int]:
    """Return all Sonata schema version constants as a dictionary."""
    from .cache import CACHE_SCHEMA_VERSION
    from .plan_handle import PLAN_HANDLE_SCHEMA_VERSION, RUNTIME_CONTRACT_VERSION
    from .serialization import (
        ELIGIBILITY_RESULT_SCHEMA_VERSION,
        FINGERPRINT_VERSION,
        SCORE_SCHEMA_VERSION,
    )

    return {
        "score_schema": SCORE_SCHEMA_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "eligibility_result_schema": ELIGIBILITY_RESULT_SCHEMA_VERSION,
        "plan_handle_schema": PLAN_HANDLE_SCHEMA_VERSION,
        "runtime_contract": RUNTIME_CONTRACT_VERSION,
        "cache_schema": CACHE_SCHEMA_VERSION,
    }


def public_api() -> list[str]:
    """Return all public symbol names exported by the sonata package."""
    import sonata
    return sorted(sonata.__all__)


def module_api() -> dict[str, list[str]]:
    """Return public symbols grouped by source module."""
    import sonata
    result: dict[str, list[str]] = {}
    for name in sorted(sonata.__all__):
        obj = getattr(sonata, name, None)
        if obj is None:
            continue
        module = getattr(obj, "__module__", "")
        if module.startswith("sonata."):
            short = module.split(".", 1)[1]
        elif module == "sonata":
            short = "core"
        else:
            short = "reexport"
        result.setdefault(short, []).append(name)
    return result


__all__ = [
    "SONATA_VERSION",
    "VERSION_INFO",
    "deprecated",
    "module_api",
    "public_api",
    "schema_versions",
    "version_string",
]
