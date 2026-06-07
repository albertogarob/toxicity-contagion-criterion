"""Shared pytest fixtures and skip guards.

The corpus-dependent tests need the PII-bearing Sonnet label files (gitignored). On a fresh
clone without them, those tests skip; the synthetic-data and committed-results tests always run.
"""

import json
import os

import pytest

from toxicity_criterion import config

requires_corpus = pytest.mark.skipif(
    not os.path.exists(config.SONNET_TOXIC),
    reason="Sonnet label files not present (PII; gitignored). See data/README.md.",
)


def load_result(name: str):
    """Load a committed results/<name>.json, or skip if it has not been generated yet."""
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"results/{name} not generated yet (run `make reproduce`).")
    with open(path) as f:
        return json.load(f)
