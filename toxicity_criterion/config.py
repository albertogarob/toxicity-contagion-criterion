"""Central path resolution , the single source of truth for the repository layout.

Reproduction reads the **committed, public, PII-free** data (no comment text, pseudonymized
usernames); the analyses reproduce offline with no Reddit calls. The private source files (real
text + usernames) live alongside but are gitignored and used only to regenerate the public
release (``build_public_data.py``) or recover real text (``rehydrate.py``).

  data/public/                      committed (PII-free)
      corpus/toxic_comments.jsonl       id, author(pseudonym), type, parent_id, prediction (no text)
      corpus/non_toxic_comments.jsonl
      corpus/raw/<sub>_dataset.jsonl    id, author(pseudonym), type, parent_id (no text)
      corpus_dehydrated.csv             id, parent_id, subreddit, sonnet_label
      validation/                       panel_*.csv, validation_populations.json,
                                        validation_key.csv (anon: sample_id+stratum), human_labels.csv
  data/raw/  data/derived/          private source (real text + usernames; gitignored)
  figures/   results/               generated outputs

See ``data/README.md`` and ``DATA_RELEASE.md``.
"""

from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PKG_DIR)

DATA_DIR = os.path.join(REPO_ROOT, "data")
PUBLIC_DIR = os.path.join(DATA_DIR, "public")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")

# --- Reproduction inputs (committed, public, anonymized) ---
_PUB_CORPUS = os.path.join(PUBLIC_DIR, "corpus")
SONNET_TOXIC = os.path.join(_PUB_CORPUS, "toxic_comments.jsonl")
SONNET_NONTOXIC = os.path.join(_PUB_CORPUS, "non_toxic_comments.jsonl")
RAW_DIR = os.path.join(_PUB_CORPUS, "raw")
VAL_DIR = os.path.join(PUBLIC_DIR, "validation")
# disattenuation reads the (now public, anonymized) key + human labels and the panel/populations
# from the same public validation directory.
VAL_PUBLIC_DIR = VAL_DIR
VAL_PRIVATE_DIR = VAL_DIR

# --- Private source files (real text + usernames; gitignored; regeneration only) ---
RAW_PRIVATE_DIR = os.path.join(DATA_DIR, "raw", "reddit")
SONNET_TOXIC_PRIVATE = os.path.join(DERIVED_DIR, "toxic_comments_sonnet.jsonl")
SONNET_NONTOXIC_PRIVATE = os.path.join(DERIVED_DIR, "non_toxic_comments_sonnet.jsonl")
VAL_PRIVATE_SRC = os.path.join(DERIVED_DIR, "validation")

# --- Generated outputs ---
FIG_DIR = os.path.join(REPO_ROOT, "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

for _d in (FIG_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
