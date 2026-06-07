"""Path resolution for the document-only labelling / validation scripts (Stage 2 + 3).

These scripts are included for provenance and were **not** re-run for reproduction. They read
and write PII-bearing intermediates under ``data/derived/`` (gitignored) and additionally
require API keys, a GPU, the GPT-OSS stratifier predictions, and the external HurtLex lexicon,
none of which are shipped. See ``README.md`` in this folder. All paths resolve through the
package's :mod:`toxicity_criterion.config`.
"""

from __future__ import annotations

from pathlib import Path

from .. import config

DATA = Path(config.DERIVED_DIR)  # data/derived (PII labels + private validation)
VAL = Path(config.VAL_PRIVATE_DIR)  # data/derived/validation (validation_key, human_labels)
VAL_PUBLIC = Path(config.VAL_PUBLIC_DIR)  # data/public/validation (panel_*, populations)
RAW = Path(config.RAW_DIR)  # data/raw/reddit
DOCS = Path(config.REPO_ROOT) / "docs"  # methodology docs

# Deployed Sonnet classifier labels (Stage 2 output; the inputs the reproduction consumes).
SONNET_TOXIC = Path(config.SONNET_TOXIC)
SONNET_NONTOXIC = Path(config.SONNET_NONTOXIC)

# GPT-OSS 20B stratifier predictions + fine-tune training set (Stage 2; regenerable; not shipped).
GPTOSS_TOXIC = DATA / "toxic_comments_20260106_054723-model-predictions.jsonl"
GPTOSS_NONTOXIC = DATA / "non_toxic_comments_20260106_054723-model-predections.jsonl"
FINETUNE_TRAIN = DATA / "data_balanced_18k.jsonl"
