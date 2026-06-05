#!/usr/bin/env python3
"""Script to get new fingerprints after adding severity field to shape_assumptions."""

import sys
sys.path.append('src')
sys.path.append('tests/st')
sys.path.append('python')

from tests.pypto_cases.test_eligibility import (
    _upstream_tile_abs_program,
    _upstream_tile_cast_row_major_narrow_program,
    _upstream_matmul_64x64x64_program,
)
from sonata.pypto_adapter import PostSimplifyPyPTOInputAdapter
from sonata.serialization import score_fingerprint

adapter = PostSimplifyPyPTOInputAdapter()

# Test tile_abs
print("Testing tile_abs...")
facts = adapter.normalize(require_certified=True)
score = facts.to_score()
fp1 = score_fingerprint(score)
print(f'tile_abs fingerprint: {fp1}')

# Test tile_cast_row_major_narrow  
print("\nTesting tile_cast_row_major_narrow...")
facts2 = adapter.normalize(require_certified=True)
score2 = facts2.to_score()
fp2 = score_fingerprint(score2)
print(f'tile_cast_row_major_narrow fingerprint: {fp2}')

# Test matmul_64x64x64
print("\nTesting matmul_64x64x64...")
facts3 = adapter.normalize(require_certified=True)
score3 = facts3.to_score()
fp3 = score_fingerprint(score3)
print(f'matmul_64x64x64 fingerprint: {fp3}')

print("\n=== NEW FINGERPRINTS ===")
print(f"tile_abs: {fp1}")
print(f"tile_cast_row_major_narrow: {fp2}")
print(f"matmul_64x64x64: {fp3}")
