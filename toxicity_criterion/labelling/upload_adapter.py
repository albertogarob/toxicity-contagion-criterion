"""Upload the GPT-OSS LoRA adapter to the HuggingFace Hub (gated; document-only).

The adapter is fine-tuned on the Reddit toxicity data, so its release is subject to the **same
institutional ethics/DPO approval** as the corpus (see ``../../DATA_RELEASE.md``). Do not run this
until that approval is in place.

Prerequisites:
  * the trained adapter directory (from ``finetune_gptoss.py``; default save dir
    ``./2026_consis_trained_fixed``), containing ``adapter_config.json`` + adapter weights;
  * ``huggingface_hub`` installed (``uv pip install -e ".[finetune]"``);
  * a HuggingFace token with write access (``huggingface-cli login`` or ``HF_TOKEN`` env var);
  * the model card ``docs/MODEL_CARD_gptoss_lora.md`` filled in.

Usage:
  python -m toxicity_criterion.labelling.upload_adapter \
      --adapter-dir ./2026_consis_trained_fixed \
      --repo-id <hf-user>/<adapter-repo> [--private] [--dry-run]
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload the GPT-OSS LoRA adapter to the HF Hub (gated).")
    ap.add_argument("--adapter-dir", required=True, help="path to the trained adapter directory")
    ap.add_argument("--repo-id", required=True, help="target HF repo, e.g. user/gptoss-tox-lora")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--dry-run", action="store_true", help="check inputs, do not upload")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.adapter_dir, "adapter_config.json")):
        raise SystemExit(
            f"No adapter_config.json in {args.adapter_dir!r}; point --adapter-dir at the LoRA "
            "output from finetune_gptoss.py."
        )

    print("Reminder: release is gated on institutional ethics/DPO approval (see DATA_RELEASE.md).")
    if args.dry_run:
        print(
            f"[dry-run] would upload {args.adapter_dir} -> {args.repo_id} "
            f"({'private' if args.private else 'public'}) + docs/MODEL_CARD_gptoss_lora.md"
        )
        return

    from huggingface_hub import HfApi, create_repo  # imported lazily so --dry-run needs no dep

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    api.upload_folder(folder_path=args.adapter_dir, repo_id=args.repo_id, repo_type="model")
    card = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "MODEL_CARD_gptoss_lora.md")
    if os.path.isfile(card):
        api.upload_file(
            path_or_fileobj=card, path_in_repo="README.md", repo_id=args.repo_id, repo_type="model"
        )
    print(f"uploaded adapter + model card to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
