"""Sonata runtime integration — registers ``sonata_tmarb`` with simpler's RuntimeBuilder.

This module monkey-patches ``simpler_setup.runtime_builder.RuntimeBuilder``
to discover the ``sonata_tmarb`` C++ runtime variant. The patch is applied
automatically on import, before any runtime binary is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("sonata.runtime")

_RUNTIME_NAME = "sonata_tmarb"


def _patch_runtime_builder() -> None:
    """Extend RuntimeBuilder to discover the sonata_tmarb runtime.

    The sonata_tmarb C++ source lives in ``runtime/sonata_tmarb/``
    relative to the project root. This patch makes it visible to
    simpler's build/discovery system without modifying upstream files.
    """
    try:
        import simpler_setup.runtime_builder as rb
    except ImportError:
        return

    # Find the sonata_tmarb directory relative to this file
    sonata_dir = Path(__file__).resolve().parent.parent.parent / "runtime" / _RUNTIME_NAME
    if not (sonata_dir / "build_config.py").exists():
        log.debug("sonata_tmarb build_config not found at %s", sonata_dir)
        return

    _orig_init = rb.RuntimeBuilder.__init__

    def _patched_init(self, platform: str, build_dir: str | Path | None = None) -> None:
        _orig_init(self, platform, build_dir)
        config_path = sonata_dir / "build_config.py"
        if config_path.exists() and _RUNTIME_NAME not in self._runtimes:
            self._runtimes[_RUNTIME_NAME] = config_path

    rb.RuntimeBuilder.__init__ = _patched_init


_patch_runtime_builder()
