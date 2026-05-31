# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared argument-direction normalization helpers."""

READ_DIRECTIONS = frozenset({"input", "inout"})
WRITE_DIRECTIONS = frozenset({"output", "outputexisting", "inout"})
IGNORED_DIRECTIONS = frozenset({"scalar", "nodep"})
MEMORY_DIRECTIONS = READ_DIRECTIONS | WRITE_DIRECTIONS


def normalize_direction(direction: object) -> str:
    """Return Sonata's canonical direction token."""
    return "".join(ch for ch in str(direction).lower() if ch.isalnum())


__all__ = [
    "IGNORED_DIRECTIONS",
    "MEMORY_DIRECTIONS",
    "READ_DIRECTIONS",
    "WRITE_DIRECTIONS",
    "normalize_direction",
]
