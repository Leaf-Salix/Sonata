# Sonata Layout

This project is being separated from the PyPTO fork so PyPTO can become a
pinned upstream dependency or submodule.

## Current Independent Sonata Tree

```text
src/sonata/
  audit.py
  dependencies.py
  eligibility.py
  score.py
  serialization.py
  storage.py

tests/sonata/
  test_audit.py
  test_dependencies.py
  test_eligibility.py
  test_score.py
  test_serialization.py
  test_storage.py

tests/pypto_cases/
  test_eligibility.py  # PyPTO-backed compatibility/extraction coverage

patches/pypto/
  runtime-target-seam.patch
```

## Boundary

`src/sonata/` contains the experimental Sonata package. Its core model is pure
Python and should not import PyPTO internals.

`tests/sonata/` contains tests that do not require PyPTO. Eligibility tests in
this tree use PyPTO-free structural mock objects so Sonata can validate its
analysis and schema behavior without importing PyPTO internals.

`tests/pypto_cases/` contains compatibility and extraction tests that still
need PyPTO. These should run against the future PyPTO submodule.

`patches/pypto/` stores minimal upstream-facing PyPTO changes that Sonata may
need later.

## Transitional State

`upstream/pypto/` is intentionally absent until it is added as a real
git submodule. Do not commit a copied PyPTO source tree there; use a submodule
so Sonata can pin and update PyPTO explicitly.

## Target Submodule Shape

After the new Sonata repository root is initialized, add PyPTO as a pinned
upstream submodule:

```text
upstream/
  pypto/          # git submodule pointing at upstream PyPTO
```

Then run Sonata compatibility tests with both packages on `PYTHONPATH`, for
example:

```text
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/sonata tests/pypto_cases
```

If Sonata needs a small PyPTO seam before it is upstreamed, apply the relevant
patch from `patches/pypto/` to the submodule worktree or to a temporary PyPTO
integration branch.

## Planned Directories

The following directories are intentionally not present yet:

- `examples/`: add once Sonata has a stable end-to-end example from PyPTO DSL / IR to Score JSON.
- `scripts/`: add only when repeated local commands become stable enough to share.
- `.github/workflows/`: add after the new remote repository and PyPTO submodule policy are settled.
