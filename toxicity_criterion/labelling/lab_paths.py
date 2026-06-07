"""Path resolution for the document-only labelling / validation scripts (Stage 2 + 3).

These scripts are included for provenance and were **not** re-run for reproduction. They read
and write the **private** source files (real text + usernames under ``data/raw`` / ``data/derived``,
gitignored) and require API keys, a GPU, the GPT-OSS stratifier predictions, and the external
HurtLex lexicon, none of which are shipped. See ``README.md`` in this folder. Reproduction itself
uses the committed, anonymized public data resolved in :mod:`toxicity_criterion.config`.
"""

from __future__ import annotations

from pathlib import Path

from .. import config

DATA = Path(config.DERIVED_DIR)  # data/derived (private)
VAL = Path(config.VAL_PRIVATE_SRC)  # data/derived/validation (real key, intermediates)
VAL_PUBLIC = Path(config.VAL_PUBLIC_DIR)  # data/public/validation (panels, populations)
RAW = Path(config.RAW_PRIVATE_DIR)  # data/raw/reddit (real)
DOCS = Path(config.REPO_ROOT) / "docs"  # methodology docs

# Deployed Sonnet classifier labels (Stage 2 output; real text + usernames; private).
SONNET_TOXIC = Path(config.SONNET_TOXIC_PRIVATE)
SONNET_NONTOXIC = Path(config.SONNET_NONTOXIC_PRIVATE)

# GPT-OSS 20B stratifier predictions + fine-tune training set (Stage 2; regenerable; not shipped).
GPTOSS_TOXIC = DATA / "toxic_comments_20260106_054723-model-predictions.jsonl"
GPTOSS_NONTOXIC = DATA / "non_toxic_comments_20260106_054723-model-predections.jsonl"
FINETUNE_TRAIN = DATA / "data_balanced_18k.jsonl"
