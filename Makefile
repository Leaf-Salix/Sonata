.PHONY: test-sonata test-integration test-st test-all lint

PYTHON ?= python
PYTHONPATH_SRC := src
PYTHONPATH_FULL := src:upstream/pypto/python

test-sonata:
	PYTHONPATH=$(PYTHONPATH_SRC) $(PYTHON) -m pytest tests/sonata -q --tb=short

test-integration:
	PYTHONPATH=$(PYTHONPATH_FULL) $(PYTHON) -m pytest tests/pypto_cases -q --tb=short

test-st:
	cd upstream/pypto && PYTHONPATH=tests/st:python:../../src $(PYTHON) -m pytest tests/st/runtime/ops/test_abs.py --with-sonata --platform=a2a3sim --forked -q

test-st-baseline:
	cd upstream/pypto && PYTHONPATH=tests/st:python:../../src $(PYTHON) -m pytest tests/st/runtime/ops/test_abs.py --platform=a2a3sim --forked -q

test-all: test-sonata test-integration

lint:
	$(PYTHON) -m ruff check src/sonata/ tests/sonata/
