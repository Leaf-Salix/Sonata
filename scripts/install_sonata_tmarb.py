#!/usr/bin/env python3
"""Install sonata_tmarb into simpler's runtime discovery path.

Creates a symlink so that simpler's build system can find and compile
the sonata_tmarb runtime. Safe to run multiple times (idempotent).

Usage:
    python scripts/install_sonata_tmarb.py          # create symlink
    python scripts/install_sonata_tmarb.py --remove  # remove symlink
    python scripts/install_sonata_tmarb.py --rebuild  # symlink + rebuild simpler
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PYPTO_SONATA_ROOT = SCRIPT_DIR.parent
UPSTREAM_RUNTIME = PYPTO_SONATA_ROOT / "upstream" / "pypto" / "runtime"
SONATA_TMARB_SRC = PYPTO_SONATA_ROOT / "runtime" / "sonata_tmarb"
SYMLINK_TARGET = UPSTREAM_RUNTIME / "src" / "a2a3" / "runtime" / "sonata_tmarb"
VENV_PIP = PYPTO_SONATA_ROOT.parent / ".venv-sonata" / "bin" / "pip"


def create_symlink():
    if SYMLINK_TARGET.exists() or SYMLINK_TARGET.is_symlink():
        if SYMLINK_TARGET.is_symlink():
            existing = SYMLINK_TARGET.resolve()
            if existing == SONATA_TMARB_SRC.resolve():
                print(f"Symlink already exists and is correct: {SYMLINK_TARGET}")
                return True
            else:
                print(f"WARNING: symlink exists but points to {existing}, not {SONATA_TMARB_SRC}")
                return False
        else:
            print(f"ERROR: {SYMLINK_TARGET} exists and is not a symlink")
            return False

    # Use relative path so it works across different mount points
    rel_target = os.path.relpath(SONATA_TMARB_SRC, SYMLINK_TARGET.parent)
    SYMLINK_TARGET.symlink_to(rel_target)
    print(f"Created symlink: {SYMLINK_TARGET} -> {rel_target}")
    return True


def remove_symlink():
    if SYMLINK_TARGET.is_symlink():
        SYMLINK_TARGET.unlink()
        print(f"Removed symlink: {SYMLINK_TARGET}")
        return True
    elif SYMLINK_TARGET.exists():
        print(f"ERROR: {SYMLINK_TARGET} exists but is not a symlink — not removing")
        return False
    else:
        print(f"Symlink does not exist: {SYMLINK_TARGET}")
        return True


def rebuild_simpler():
    if not VENV_PIP.exists():
        print(f"ERROR: pip not found at {VENV_PIP}")
        return False

    print("Rebuilding simpler with sonata_tmarb...")
    cmd = [str(VENV_PIP), "install", "--no-build-isolation", "-e", str(UPSTREAM_RUNTIME)]
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(UPSTREAM_RUNTIME))
    if result.returncode != 0:
        print(f"ERROR: pip install failed with exit code {result.returncode}")
        return False

    print("simpler rebuilt successfully with sonata_tmarb runtime")
    return True


def main():
    parser = argparse.ArgumentParser(description="Install sonata_tmarb into simpler's runtime discovery path")
    parser.add_argument("--remove", action="store_true", help="Remove the symlink instead of creating it")
    parser.add_argument("--rebuild", action="store_true", help="Also rebuild simpler after creating the symlink")
    args = parser.parse_args()

    if args.remove:
        success = remove_symlink()
    else:
        success = create_symlink()
        if success and args.rebuild:
            success = rebuild_simpler()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
