# Auto-loaded by pytest at rootdir level.
# Loads Sonata st test hooks for --with-sonata flag.
# Note: --platform flag comes from upstream/pypto/tests/st/conftest.py
# which is discovered by pytest's conftest walk automatically.

import importlib.util
import os
import sys
from pathlib import Path

# Ensure st test harness is importable (needed for the monkeypatch in
# st_sonata/conftest.py to find harness.core.test_runner).
_upstream_st = Path(__file__).parent / "upstream" / "pypto" / "tests" / "st"
if _upstream_st.exists():
    sys.path.insert(0, str(_upstream_st))

_st_sonata_conftest = Path(__file__).parent / "tests" / "st_sonata" / "conftest.py"
_mod = None

if _st_sonata_conftest.exists():
    spec = importlib.util.spec_from_file_location("_sonata_st_conftest", _st_sonata_conftest)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)


def pytest_addoption(parser):
    if _mod is not None:
        _mod.pytest_addoption(parser)


def pytest_runtest_setup(item):
    if _mod is not None:
        # Re-add the path if forked child loses it
        if str(_upstream_st) not in sys.path:
            sys.path.insert(0, str(_upstream_st))
        _mod.pytest_runtest_setup(item)


def pytest_sessionfinish(session, exitstatus):
    if _mod is not None and hasattr(_mod, "pytest_sessionfinish"):
        _mod.pytest_sessionfinish(session, exitstatus)
