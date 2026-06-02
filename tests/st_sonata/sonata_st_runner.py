# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sonata st test runner — run upstream st tests through Sonata analysis.

Usage:
    python tests/st_sonata/sonata_st_runner.py tests/st/runtime/ops/test_abs.py
    python tests/st_sonata/sonata_st_runner.py tests/st/runtime/ops/ -- -v --forked
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: sonata_st_runner.py <test_path_or_dir> [-- pytest-args...]")
        return 1

    # Split at '--' separator
    args = sys.argv[1:]
    pytest_args: list[str] = []
    if "--" in args:
        idx = args.index("--")
        test_target = args[:idx]
        pytest_args = args[idx + 1:]
    else:
        test_target = args

    test_path = " ".join(test_target)

    # Paths
    project_root = Path(__file__).resolve().parents[2]  # pypto-sonata/
    upstream_root = project_root / "upstream" / "pypto"
    conftest_plugin = Path(__file__).resolve().parent / "conftest.py"
    venv_python = project_root.parent / ".venv-sonata" / "bin" / "python"

    import os
    env = os.environ.copy()
    st_sonata_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = f"upstream/pypto/tests/st:upstream/pypto/python:{st_sonata_dir}"
    if "PTOAS_ROOT" in env:
        env["PATH"] = env["PTOAS_ROOT"] + "/bin:" + env.get("PATH", "")
    if "PTO_ISA_ROOT" in env:
        env["PATH"] = env["PTO_ISA_ROOT"] + "/bin:" + env.get("PATH", "")

    # Build pytest command — run from pypto-sonata/ so rootdir is correct
    # tests/conftest.py auto-loads Sonata hooks from tests/st_sonata/
    cmd = [
        str(venv_python), "-m", "pytest",
        f"upstream/pypto/{test_path}",
        "--with-sonata",
        "--platform=a2a3sim",
    ] + pytest_args

    print(f"[sonata_st_runner] Running: {' '.join(cmd)}")
    print(f"[sonata_st_runner] Working directory: {project_root}")
    print()

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        timeout=300,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
