"""Central path resolution , the single source of truth for the repository layout.

Data is organized by provenance::

  <repo>/data/raw/reddit/        raw scraped corpus (PII; gitignored)
  <repo>/data/derived/          labels + validation intermediates (PII; gitignored)
      toxic_comments_sonnet.jsonl
      non_toxic_comments_sonnet.jsonl
      validation/validation_key.csv
      validation/human_labels.csv
  <repo>/data/public/           PII-free, committed artifacts
      validation/panel_{opus,gpt,gemini}.csv
      validation/validation_populations.json
  <repo>/figures/               generated PDF figures
  <repo>/results/               generated *.json result tables

See ``data/README.md`` and ``DATA_RELEASE.md`` for the sharing policy.
"""

from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PKG_DIR)

DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "reddit")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
PUBLIC_DIR = os.path.join(DATA_DIR, "public")

VAL_PRIVATE_DIR = os.path.join(DERIVED_DIR, "validation")
VAL_PUBLIC_DIR = os.path.join(PUBLIC_DIR, "validation")

FIG_DIR = os.path.join(REPO_ROOT, "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Canonical data files (Stage 2 deployed-Sonnet labels: the inputs the reproduction consumes).
SONNET_TOXIC = os.path.join(DERIVED_DIR, "toxic_comments_sonnet.jsonl")
SONNET_NONTOXIC = os.path.join(DERIVED_DIR, "non_toxic_comments_sonnet.jsonl")

# Created on import so analyses can write outputs without pre-making the directories.
for _d in (FIG_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
