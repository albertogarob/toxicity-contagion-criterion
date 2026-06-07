# Model card , GPT-OSS 20B LoRA adapter (toxicity stratifier)

> Draft model card to accompany the adapter (e.g. on HuggingFace Hub). Fill `[brackets]`.
> The adapter is fine-tuned on a **public** hate-speech corpus (not the Reddit data). It is
> **not required to reproduce the paper's results** (the validation strata it produced are already
> shipped in `data/public/validation/validation_key.csv`, and the deployed labels are Sonnet's).

## Summary
- **Model:** LoRA adapter for `openai/gpt-oss-20b`, fine-tuned for binary toxic / non-toxic
  classification.
- **Role in the paper:** **stratifier only.** Its predictions defined the 2x2 GPT-OSS × Sonnet
  agreement table used to stratify the 200-item validation sample. It is **not** the deployed
  labeller (that is Claude Sonnet, API) and does **not** enter any estimate.
- **Base model:** `openai/gpt-oss-20b` (load the base separately; this repo ships only the adapter).
- **Adapter type:** PEFT / LoRA. r=16, α=32, dropout=0.05, target modules q/k/v/o, 3 epochs,
  lr 3e-5, max_length 256. Training script: `toxicity_criterion/labelling/finetune_gptoss.py`.

## Training data
- **A public HuggingFace hate-speech corpus**, 18,000 balanced examples (train split), with a
  held-out 1,000-comment test set (500 toxic / 500 non-toxic). **Not** the Reddit corpus; no PII
  from this project. Dataset: [name + HF URL + license , confirm]. Because the source is public,
  the fine-tune is **reproducible by anyone**.
- Format used by the training script: `data_balanced_18k.jsonl` (`instruction`, `input`, `output`).

## Reproducing the adapter (anyone, with a GPU)
1. Get the base model `openai/gpt-oss-20b` and the public training corpus above.
2. Produce `data_balanced_18k.jsonl` (18k balanced; instruction/input/output) and the 1k test split.
3. Run `python -m toxicity_criterion.labelling.finetune_gptoss` (SBATCH script: `finetune_gptoss.sh`).
   ~6 h on a single NVIDIA RTX 4000 Ada. All hyperparameters are fixed above.

## Intended use and limitations
- **Intended:** research reproduction of the validation-stratum construction in this paper.
- **Out of scope:** production moderation, decisions or enforcement against individuals.
- **Limitations / bias:** fine-tuned on a general public hate-speech corpus, so there is genuine
  domain shift to UK-immigration Reddit discourse (one reason the paper validates Sonnet, the
  deployed labeller, rather than relying on this model). Predictions are imperfect and are not
  authoritative judgements about any person.

## How to use
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "openai/gpt-oss-20b"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "[hf-user]/[adapter-repo]")
```

## Availability, license, contact
- The adapter weights are **available from the authors on request**. They are not needed to
  reproduce the paper (see the note at the top), and the fine-tune is reproducible by anyone from
  the public corpus + released script. (An optional `upload_adapter.py` is provided should the
  authors later publish the weights on the HuggingFace Hub.)
- Adapter license: [e.g. Apache-2.0 , confirm compatibility with the base model + training-corpus licenses].
- Base model `openai/gpt-oss-20b` is under its own license; comply with it.
- Contact: [email].

## Citation
[BibTeX for the ICDM 2026 paper.]
