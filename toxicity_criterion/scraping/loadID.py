import json


def load_existing_ids(path: str):
    """
    Load all existing comment IDs from a JSONL file into a set.

    Args:
        path (str): Path to the JSONL dataset file.

    Returns:
        Set[str]: A set of all unique IDs already stored.
    """
    ids = set()

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cid = obj.get("id")
                    if cid:
                        ids.add(cid)
                except json.JSONDecodeError:
                    # Skip malformed JSONL entries
                    continue

    except FileNotFoundError:
        # File doesn't exist yet; return empty set
        pass

    return ids
