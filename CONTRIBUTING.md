# Contributing to Sonata

Thank you for your interest in contributing to Sonata. This guide covers
development setup, testing, code style, and contribution workflow.

## Development Setup

### Prerequisites

- Python 3.10 or later
- pip

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install in development mode

```bash
pip install -e ".[dev]"
```

This installs Sonata in editable mode along with development dependencies
(pytest, ruff).

### Verify installation

```bash
PYTHONPATH=src python -m pytest tests/sonata
```

All tests should pass with exit code 0.

## Running Tests

### Pure Sonata tests (no PyPTO dependency)

```bash
PYTHONPATH=src python -m pytest tests/sonata
```

### PyPTO compatibility tests (requires PyPTO submodule)

```bash
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/pypto_cases
```

### Full test suite

```bash
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/sonata tests/pypto_cases
```

### Test requirements

- All tests must pass before submitting a PR.
- No newly skipped tests are allowed without justification.
- If a reasonable test fails, fix the root cause. Do not weaken or bypass a
  test to make it pass.

## Code Style

### General guidelines

- Follow existing patterns in the codebase.
- Use frozen `@dataclass(frozen=True)` for all data model types.
- Add type hints to all public functions and methods.
- Keep changes surgical. Every changed line should serve the request.
- Prefer extracting structural facts over depending on volatile runtime
  behavior.

### Formatting

Sonata uses ruff for linting and formatting:

```bash
ruff check src/sonata/ tests/
ruff format --check src/sonata/ tests/
```

Configuration is in `pyproject.toml`:

- Line length: 110
- Target Python version: 3.10

### Code organization

- Sonata-owned source code lives in `src/sonata/`.
- PyPTO-independent tests live in `tests/sonata/`.
- PyPTO compatibility tests live in `tests/pypto_cases/`.
- PyPTO-facing patches live in `patches/pypto/`.
- Sonata core code must not import PyPTO internals. PyPTO-specific extraction
  and compatibility tests belong in `tests/pypto_cases/`.

### Module conventions

- Each module exports a `__all__` list.
- Public symbols are re-exported from `src/sonata/__init__.py`.
- Schema version constants are defined per module (e.g., `SCORE_SCHEMA_VERSION`,
  `PLAN_HANDLE_SCHEMA_VERSION`, `CACHE_SCHEMA_VERSION`).
- Fallback codes use the `FallbackCode` enum for stable routing keys.

## Branch Naming

Branch names follow a strict convention:

```
<agent>/<base-version>-<type>-<slug>-alpha
```

| Component | Description | Example |
|-----------|-------------|---------|
| `<agent>` | Short contributor identifier | `claude`, `codex`, `mimo` |
| `<base-version>` | Patch baseline you forked from | `v0.1.0` |
| `<type>` | Change type | `feat` or `fix` |
| `<slug>` | Short kebab-case description | `shape-guard` |

Examples:

```
claude/v0.1.0-feat-shape-guard-alpha
codex/v0.1.0-fix-fallback-reason-alpha
mimo/v0.1.1-feat-guard-fingerprint-alpha
```

### Branch rules

1. Always branch from the latest patch baseline.
2. Never self-assign a patch number (the third digit). Patch numbers are
   assigned by the maintainer when merging.
3. No direct commits to `main` or any patch baseline branch.
4. No force-push under any circumstances.

## Pull Request Requirements

### Before submitting

1. Run `PYTHONPATH=src python -m pytest tests/sonata` and confirm exit 0.
2. Ensure no unrelated changes are included.
3. Verify ruff passes: `ruff check src/sonata/ tests/`.
4. Confirm new public symbols are added to `__all__` in the relevant module
   and re-exported from `src/sonata/__init__.py`.

### PR content

- Keep the title concise (under 70 characters).
- Describe what the change does and why, not just what files changed.
- List the test commands you ran and their results.
- Do not include absolute local paths, usernames, or machine names in PR
  descriptions or commit messages.

### Review expectations

- Reviews focus on correctness, behavioral regressions, edge cases, missing
  tests, coupling risks, and license compliance.
- Address review feedback before requesting re-review.

## PyPTO Boundary

PyPTO is treated as an external upstream dependency. Key rules:

- Do not import PyPTO internals in `src/sonata/`.
- Use structural introspection (Python-visible fields) instead.
- PyPTO-specific extraction and compatibility tests belong in
  `tests/pypto_cases/`.
- Minimal PyPTO seam changes go in `patches/pypto/`.
- The PyPTO submodule lives under `upstream/pypto/`.

## License

By contributing, you agree that your contributions will be licensed under
the CANN Open Software License Agreement Version 2.0, the same license
that covers the Sonata project. See [LICENSE](LICENSE) for the full text.
