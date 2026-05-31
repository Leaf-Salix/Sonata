#!/usr/bin/env bash
set -euo pipefail

# Print the v0.1 G2 PyPTO ST seed -> certified IR -> Sonata Score mapping.
# Set SCORE_ONLY=1 to hide raw ST/certified IR and print only Score/fingerprint.
# Set PYTHON_BIN=/path/to/python to select the local test environment.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="src:upstream/pypto/python${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" - <<'PY'
from pathlib import Path
import importlib.util
import sys
from pprint import pprint
import os

from pypto.backend import BackendType, is_backend_configured, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as core_passes

from sonata import (
    DEPENDENCY_POLICY_DATAFLOW_V0,
    check_static_eligibility,
    score_fingerprint,
    score_to_dict,
)

REPO_ROOT = Path.cwd()
UPSTREAM_PYPTO_ROOT = REPO_ROOT / "upstream" / "pypto"
UPSTREAM_ST_ROOT = UPSTREAM_PYPTO_ROOT / "tests" / "st"
SCORE_ONLY = os.environ.get("SCORE_ONLY") == "1"

for path in (UPSTREAM_ST_ROOT, UPSTREAM_PYPTO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def load_module(name, rel):
    path = UPSTREAM_PYPTO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream ST module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod


def final_simplify(program):
    if not is_backend_configured():
        set_backend_type(BackendType.Ascend910B)

    with core_passes.PassContext([], core_passes.VerificationLevel.NONE):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        current = program
        after_collect_comm_groups = False

        for pass_name, pass_obj in zip(manager.pass_names, manager.passes):
            current = pass_obj(current)

            if pass_name == "CollectCommGroups":
                after_collect_comm_groups = True
            elif after_collect_comm_groups and pass_name == "Simplify":
                return current

    raise RuntimeError("default pipeline did not expose Simplify after CollectCommGroups")


def show(case_name, source_path, program):
    print("\n" + "=" * 100)
    print("CASE:", case_name)
    print("ST SOURCE:", source_path.relative_to(REPO_ROOT))

    if not SCORE_ONLY:
        print("\nRAW ST PROGRAM:")
        print(program)

    certified = final_simplify(program)
    if not SCORE_ONLY:
        print("\nCERTIFIED IR: after_collect_comm_groups_simplify")
        print(certified)

    result = check_static_eligibility(
        certified,
        dependency_policy=DEPENDENCY_POLICY_DATAFLOW_V0,
    )
    print("\nELIGIBLE:", result.eligible)
    if result.reasons:
        print("REASONS:")
        pprint(result.reasons, sort_dicts=False)

    if result.score is None:
        return

    print("\nSCORE:")
    pprint(score_to_dict(result.score), sort_dicts=False)

    print("\nFINGERPRINT:")
    print(score_fingerprint(result.score))


path, mod = load_module("st_abs", "tests/st/runtime/ops/test_abs.py")
show("tile_abs", path, mod.TileAbsProgram)

path, mod = load_module("st_cast", "tests/st/runtime/ops/test_cast.py")
show("tile_cast_row_major_narrow", path, mod.TileCastRowMajorNarrowProgram)

path, mod = load_module("st_matmul", "tests/st/runtime/ops/test_matmul.py")
case = mod.TestMatmul(m=64, k=64, n=64, platform="a2a3sim")
show("matmul_64x64x64", path, case.get_program())

path, mod = load_module("st_l2_multi_orch", "tests/st/distributed/test_l2_multi_orch.py")
show("l2_multi_orch_add_sub", path, mod.TwoL2AddSubProgram)
PY
