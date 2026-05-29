# Sonata

Sonata is an experimental compile-time static DAG layer for PyPTO. The project
explores how to extract DAG regions from PyPTO DSL / IR before runtime
orchestration, so a runtime can consume formed graph fragments instead of
reconstructing them from dynamic hints.

## Layout

```text
src/sonata/          # Sonata package: DAG model, analysis, serialization
tests/sonata/        # PyPTO-free Sonata tests
tests/pypto_cases/   # PyPTO compatibility and extraction cases
patches/pypto/       # minimal PyPTO seam patches kept outside upstream
docs/                # Sonata project documentation
upstream/pypto/      # future PyPTO git submodule location
```

## PyPTO Boundary

PyPTO is treated as a reference, test-case source, future host, and eventual
upstream integration target. It should live under `upstream/pypto/` as a pinned
submodule, not as Sonata-owned source.

Sonata core code should stay in `src/sonata/` and avoid importing PyPTO
internals. PyPTO-specific extraction and compatibility tests belong in
`tests/pypto_cases/`.

## Local Test Commands

Pure Sonata tests:

```bash
PYTHONPATH=src python -m pytest tests/sonata
```

With a PyPTO checkout or submodule:

```bash
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/sonata tests/pypto_cases
```
