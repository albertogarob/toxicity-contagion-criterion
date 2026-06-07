"""JSON Lines input helpers shared across the package.

A "post" record is a dict with at least ``id``, ``parent_id``, ``author``, and a toxicity
field (``prediction`` == "toxic" for the Sonnet label files, or a boolean ``toxic``).
"""

from __future__ import annotations

import json
import os


def load_jsonl(filepath: str) -> list[dict]:
    """Load a list of records from a JSON Lines file.

    Raises ``FileNotFoundError`` if the file is missing; skips (and counts) malformed lines.
    """
    posts: list[dict] = []
    errors = 0

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError:
                    errors += 1

    if errors > 0:
        print(f"  Warning: {errors} lines could not be parsed")

    return posts


def load_corpus(toxic_path: str, nontoxic_path: str) -> list[dict]:
    """Load and concatenate the toxic and non-toxic Sonnet label files into one corpus."""
    return load_jsonl(toxic_path) + load_jsonl(nontoxic_path)
