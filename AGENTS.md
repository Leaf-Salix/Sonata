# Sonata Codex Instructions

This repository is the Sonata project root. PyPTO should be treated as an
external upstream dependency under `upstream/pypto/`, ideally as a git
submodule.

## Working Rules

- Keep Sonata-owned code in `src/sonata/`.
- Keep PyPTO-independent tests in `tests/sonata/`.
- Keep PyPTO compatibility/extraction tests in `tests/pypto_cases/`.
- Keep PyPTO modifications as patches under `patches/pypto/` unless the user
  explicitly asks to edit the submodule worktree.
- Do not mix Codex automation reports into this repository; those live in the
  outer `../reports/` workspace.
- Prefer extracting structural facts from PyPTO over depending on volatile
  runtime/orchestrator behavior.

## Verification

Use Python 3.10+.

```bash
PYTHONPATH=src python -m pytest tests/sonata
PYTHONPATH=src:upstream/pypto/python python -m pytest tests/pypto_cases
```
