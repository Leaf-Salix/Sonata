# Auto-loaded by pytest at rootdir level.
# Loads Sonata st test hooks for --with-sonata flag.

import importlib.util
import sys
from pathlib import Path

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
        _mod.pytest_runtest_setup(item)


def pytest_sessionfinish(session, exitstatus):
    if _mod is not None and hasattr(_mod, "pytest_sessionfinish"):
        _mod.pytest_sessionfinish(session, exitstatus)
