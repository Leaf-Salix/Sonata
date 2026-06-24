# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Operator profiling for feedback-driven scheduling.

v0.18 Phase 3: Collects runtime execution latency per operator signature
and provides lookup for timing-aware scheduling decisions.

Profile data is stored host-side (not in Score, not in fingerprint).
Cross-compilation reuse: same op signature → same profile.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperatorProfile:
    """Runtime execution profile for one operator signature.

    Attributes:
        op_signature: Unique key, e.g. "matmul_fp16_128x128x128".
        op_type: Operator type, e.g. "matmul".
        shape: Tensor shape tuple.
        dtype: Data type string, e.g. "fp16".
        core_type: Core type, e.g. "aic".
        mean_latency_us: Mean execution time in microseconds.
        std_latency_us: Population standard deviation of execution time.
        sample_count: Number of recorded samples.
        _m2: Internal Welford accumulator (sum of squared deviations).
             Not serialized; derived from std on load.
    """

    op_signature: str
    op_type: str
    shape: tuple[int, ...]
    dtype: str
    core_type: str
    mean_latency_us: float
    std_latency_us: float
    sample_count: int
    _m2: float = 0.0


def _make_signature(op_type: str, shape: tuple[int, ...], dtype: str) -> str:
    """Build a canonical signature string."""
    shape_str = "x".join(str(s) for s in shape)
    return f"{op_type}_{dtype}_{shape_str}"


class ProfileDatabase:
    """In-memory database of operator execution profiles.

    Supports record, lookup, and JSON persistence.
    Thread-safe for single-writer usage.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, OperatorProfile] = {}

    def lookup(
        self, op_type: str, shape: tuple[int, ...], dtype: str,
    ) -> OperatorProfile | None:
        """Return profile for the given op signature, or None if unknown."""
        sig = _make_signature(op_type, shape, dtype)
        return self._profiles.get(sig)

    def record(
        self,
        op_type: str,
        shape: tuple[int, ...],
        dtype: str,
        core_type: str,
        latency_us: float,
    ) -> None:
        """Record one execution sample. Updates mean/std incrementally."""
        sig = _make_signature(op_type, shape, dtype)
        existing = self._profiles.get(sig)

        if existing is None:
            self._profiles[sig] = OperatorProfile(
                op_signature=sig,
                op_type=op_type,
                shape=shape,
                dtype=dtype,
                core_type=core_type,
                mean_latency_us=latency_us,
                std_latency_us=0.0,
                sample_count=1,
                _m2=0.0,
            )
            return

        # Incremental mean/std update (Welford's algorithm)
        n = existing.sample_count + 1
        old_mean = existing.mean_latency_us
        new_mean = old_mean + (latency_us - old_mean) / n
        # Welford recurrence: new_m2 = old_m2 + (x - old_mean)(x - new_mean)
        old_m2 = existing._m2
        new_m2 = old_m2 + (latency_us - old_mean) * (latency_us - new_mean)
        new_std = math.sqrt(new_m2 / n) if n > 1 else 0.0

        self._profiles[sig] = OperatorProfile(
            op_signature=sig,
            op_type=op_type,
            shape=shape,
            dtype=dtype,
            core_type=core_type,
            mean_latency_us=new_mean,
            std_latency_us=new_std,
            sample_count=n,
            _m2=new_m2,
        )

    def all_profiles(self) -> tuple[OperatorProfile, ...]:
        """Return all stored profiles."""
        return tuple(self._profiles.values())

    def save(self, path: str | Path) -> Path:
        """Save database to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "profiles": [
                {
                    "op_signature": p.op_signature,
                    "op_type": p.op_type,
                    "shape": list(p.shape),
                    "dtype": p.dtype,
                    "core_type": p.core_type,
                    "mean_latency_us": p.mean_latency_us,
                    "std_latency_us": p.std_latency_us,
                    "sample_count": p.sample_count,
                }
                for p in self._profiles.values()
            ],
        }
        p.write_text(json.dumps(data, indent=2, sort_keys=True))
        return p

    def load(self, path: str | Path) -> None:
        """Load database from JSON file. Merges with existing data."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        for entry in data.get("profiles", []):
            n = entry["sample_count"]
            std = entry["std_latency_us"]
            # Reconstruct _m2 from population std: std = sqrt(m2/n) → m2 = std^2 * n
            m2 = (std ** 2) * n if n > 1 else 0.0
            profile = OperatorProfile(
                op_signature=entry["op_signature"],
                op_type=entry["op_type"],
                shape=tuple(entry["shape"]),
                dtype=entry["dtype"],
                core_type=entry["core_type"],
                mean_latency_us=entry["mean_latency_us"],
                std_latency_us=std,
                sample_count=n,
                _m2=m2,
            )
            self._profiles[profile.op_signature] = profile


__all__ = [
    "OperatorProfile",
    "ProfileDatabase",
]
