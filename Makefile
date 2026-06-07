# Reproducibility tasks. Uses uv for the environment.
# Determinism: every analysis runs under PYTHONHASHSEED=0 (seed 42 inside the code).

PY := .venv/bin/python
export PYTHONHASHSEED := 0

.PHONY: help setup rehydrate reproduce figures criterion test lint format clean

help:
	@echo "make setup      , create .venv (uv) and install the package + dev tools"
	@echo "make rehydrate  , re-fetch corpus text by id from data/public/corpus_dehydrated.csv"
	@echo "make reproduce  , run every CPU analysis that produces the paper's tables"
	@echo "make figures    , regenerate the paper's figures into figures/"
	@echo "make criterion  , run the criterion tool on the synthetic demo"
	@echo "make test       , run the regression test suite (locks the paper's numbers)"
	@echo "make lint       , ruff + black --check"
	@echo "make format     , apply black + ruff --fix"
	@echo "make clean      , remove caches and generated outputs"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

# Re-fetch comment text by id (needs `requests`: uv pip install -e ".[labelling]").
# Reconstructs data/derived/*_sonnet.jsonl + data/raw/reddit/*.jsonl so `make reproduce` works.
rehydrate:
	$(PY) -m toxicity_criterion.labelling.rehydrate

reproduce:
	$(PY) -m toxicity_criterion.cli reproduce-all 2>/dev/null || reproduce-all

figures:
	$(PY) -m toxicity_criterion.figures.positive_control
	$(PY) -m toxicity_criterion.figures.reply_subgraphs
	@echo "Fig. 1: render figures/fig_diagnostic.mmd with:"
	@echo "  npx -p @mermaid-js/mermaid-cli mmdc -i figures/fig_diagnostic.mmd -o figures/fig_diagnostic.pdf -b transparent"

criterion:
	$(PY) -m toxicity_criterion.criterion.decision --demo --precision 0.92 --recall 0.32

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check toxicity_criterion tests
	$(PY) -m black --check toxicity_criterion tests

format:
	$(PY) -m black toxicity_criterion tests
	$(PY) -m ruff check --fix toxicity_criterion tests

clean:
	rm -rf $$(find . -name __pycache__ -not -path './.venv/*') .pytest_cache
	rm -f results/*.json figures/*.pdf
