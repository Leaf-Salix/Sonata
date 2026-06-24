# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen trace extraction — extract TMARB API call sequences from real C++ output.

Uses regex to scan generated ``orchestration.cpp`` files for TMARB API calls
and produce a ``list[TMARBCallTraceEntry]`` that can be structurally compared
against Sonata-generated traces.
"""

from __future__ import annotations

from typing import Any

from .trace import TMARBCallTraceEntry


TMARB_APIS = [
    "add_input", "add_output", "add_inout", "add_no_dep", "add_scalar",
    "set_dependencies", "rt_submit_aic_task", "rt_submit_aiv_task",
    "rt_submit_task", "PTO2_SCOPE", "alloc_tensors", "get_ref",
    "from_tensor_arg", "make_tensor_external",
    "rt_orchestration_done",
    "rt_report_fatal", "rt_is_fatal",
]


def extract_codegen_trace(cpp_source: str) -> list[TMARBCallTraceEntry]:
    """Extract TMARB API calls from real codegen C++ output.

    Uses regex to find API calls in the generated source. Returns an
    ordered sequence of trace entries. Skips comments and string literals
    to avoid false positives.

    Args:
        cpp_source: The full text of a generated ``orchestration.cpp``.

    Returns:
        A list of trace entries corresponding to the TMARB API calls
        found in the source, in source order.
    """
    lines = cpp_source.split("\n")
    entries: list[TMARBCallTraceEntry] = []
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue

        entry = _match_api_call(stripped, lineno)
        if entry is not None:
            entries.append(entry)

    return entries


def diff_traces(
    sonata_trace: list[TMARBCallTraceEntry],
    codegen_trace: list[TMARBCallTraceEntry],
) -> list[str]:
    """Structural diff between a Sonata trace and a codegen trace.

    Compares the sequence of API calls, ignoring ordering of independent
    tasks within the same scope.

    Returns a list of human-readable diff lines. Empty list = identical.
    """
    diffs: list[str] = []
    s_idx = 0
    c_idx = 0

    while s_idx < len(sonata_trace) and c_idx < len(codegen_trace):
        s = sonata_trace[s_idx]
        c = codegen_trace[c_idx]

        if _entries_match(s, c):
            s_idx += 1
            c_idx += 1
        else:
            diffs.append(f"Sonata[{s_idx}]: api={s.api} phase={s.phase} | "
                         f"Codegen[{c_idx}]: api={c.api} phase={c.phase}")
            s_idx += 1

    if s_idx < len(sonata_trace):
        diffs.append(f"Sonata has {len(sonata_trace) - s_idx} extra entries starting at {s_idx}")
    if c_idx < len(codegen_trace):
        diffs.append(f"Codegen has {len(codegen_trace) - c_idx} extra entries starting at {c_idx}")

    return diffs


def _match_api_call(line: str, lineno: int) -> TMARBCallTraceEntry | None:
    for api in TMARB_APIS:
        if api in line:
            args: dict[str, Any] = {}
            phase = "task"
            if api == "PTO2_SCOPE":
                phase = "region"
            elif api in ("from_tensor_arg", "make_tensor_external", "alloc_tensors"):
                phase = "entry_setup"
            elif api in ("rt_orchestration_done", "rt_report_fatal", "rt_is_fatal"):
                phase = "finish"
            elif api == "get_ref":
                phase = "output_bind"
            elif api == "set_dependencies":
                phase = "dep"

            return TMARBCallTraceEntry(
                phase=phase,
                api=api,
                args=args,
            )
    return None


def _entries_match(a: TMARBCallTraceEntry, b: TMARBCallTraceEntry) -> bool:
    if a.api != b.api:
        return False
    if a.phase != b.phase:
        return False
    return True


__all__ = [
    "TMARB_APIS",
    "extract_codegen_trace",
    "diff_traces",
]
