# Model card , GPT-OSS 20B LoRA adapter (toxicity stratifier)

> Draft model card to accompany the released adapter (e.g. on HuggingFace Hub). Fill `[brackets]`.
> Release is **gated**: upload only after the institutional ethics/DPO approval that also governs
> the corpus (the adapter is fine-tuned on the Reddit toxicity data). See `../DATA_RELEASE.md`.

## Summary
- **Model:** LoRA adapter for `openai/gpt-oss-20b`, fine-tuned to classify UK Reddit comments as
  toxic / non-toxic.
- **Role in the paper:** **stratifier only.** Used to define the validation strata
  (the GPT-OSS × Sonnet agreement table); it is **not** the deployed labeller. The deployed
  classifier is Claude Sonnet (API). Do not treat this adapter as the paper's labeller.
- **Base model:** `openai/gpt-oss-20b` (load the base separately; this repo ships only the adapter).
- **Adapter type:** PEFT / LoRA. r=16, α=32, dropout=0.05, target modules q/k/v/o, 3 epochs,
  lr 3e-5, max_length 256. Training script: `toxicity_criterion/labelling/finetune_gptoss.py`.

## Training data
- A balanced 18,000-example subset (`data_balanced_18k.jsonl`; 9k toxic / 9k non-toxic) drawn
  from the UK Reddit corpus; held-out test 1,000 (500/500). The training data is **not** shipped
  (PII; same data-availability gate as the corpus).

## Intended use and limitations
- **Intended:** research reproduction of the validation-stratum construction in this paper.
- **Out of scope:** production moderation, decisions or enforcement against individuals,
  general-purpose toxicity detection beyond this domain.
- **Limitations / bias:** trained on a single platform, topic (UK immigration discourse), and
  two-month window; labels are imperfect (see the paper's validation: precision ~0.92, recall a
  range). May reflect annotation-rubric and base-model biases. Predictions are not authoritative
  judgements about any person.

## Evaluation
- See the paper's in-domain validation (multi-LLM silver panel + human reference) and
  `docs/VALIDATION_SET_DESIGN.md`.

## How to use
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "openai/gpt-oss-20b"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "[hf-user]/[adapter-repo]")
```

## License and contact
- Adapter license: [e.g. CC-BY-4.0 / Apache-2.0 , confirm compatibility with the base model license].
- Base model `openai/gpt-oss-20b` is under its own license; comply with it.
- Contact / takedown: [email].

## Citation
[BibTeX for the ICDM 2026 paper.]
