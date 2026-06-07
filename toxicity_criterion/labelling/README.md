# Labelling and fine-tuning (Stage 2 + Stage 3)

These scripts are included **for provenance and reproducibility documentation**. They were
**not re-run** to verify the paper's results: they are costly, non-deterministic (LLM APIs),
and/or require a GPU, and their outputs already ship as the Stage 2 label files that the
CPU reproduction consumes (see the top-level `README.md`).

All paths resolve through `lab_paths.py` (which wraps `toxicity_criterion/config.py`), so inputs
and outputs map to the repository's `data/` tree. The PII-bearing intermediates, the GPT-OSS
stratifier predictions, the HurtLex lexicon, and API keys are **not shipped**; the scripts are
documented here so a reader can see exactly how the labels were produced.

## Dehydration / rehydration (the data-sharing path)

The corpus is shared **dehydrated** (IDs + labels, no text/usernames); see `../../DATA_RELEASE.md`.

- `build_dehydrated.py` , (authors only) build `data/public/corpus_dehydrated.csv` from the
  local raw + Sonnet-label files. Output is PII-free and committed.
- `rehydrate.py` , (anyone) re-fetch comment text by ID from Reddit's public .json endpoints and
  reconstruct `data/derived/*_sonnet.jsonl` + `data/raw/reddit/*.jsonl`. Needs `requests`
  (`uv pip install -e ".[labelling]"`). Run via `make rehydrate`. Deletions are respected.

## Requirements to actually run these

- **API keys** in a gitignored `.env`: `ANTHROPIC_API_KEY` (Sonnet + Opus panel),
  `OPENAI_API_KEY` (GPT panel), `GEMINI_API_KEY` (Gemini panel).
- **GPU** for the GPT-OSS 20B LoRA fine-tune.
- Optional deps: `uv pip install -e ".[labelling]"` (APIs) and/or `".[finetune]"` (GPU stack).
- External: the **HurtLex EN** lexicon at `data/in_domain_validation/lexicon/hurtlex_EN.tsv`
  and the GPT-OSS prediction files (regenerable; see `lab_paths.py`).

## Stage 2 , deployed Sonnet classifier (the labels used by the paper)

1. `build_corpus_dataset.py` , build `corpus_dataset_full.csv` (per-comment context: subreddit
   + reply chain + lexicon flags) from the raw corpus.
2. `classify_sonnet.py` , label each comment toxic/non-toxic with Claude Sonnet in-context
   under the fixed rubric (`../../../docs/ANNOTATION_GUIDELINES.md`). Needs `ANTHROPIC_API_KEY`.
3. `build_sonnet_labels.py` , join the labels with the raw corpus to emit
   `toxic_comments_sonnet.jsonl` / `non_toxic_comments_sonnet.jsonl` (the CPU pipeline's inputs).

## Stage 2 , GPT-OSS 20B stratifier (validation strata only, not the deployed labeller)

- `finetune_gptoss.py` (+ `finetune_gptoss.sh` SLURM job) , LoRA fine-tune
  (r=16, α=32, dropout 0.05, q/k/v/o, 3 epochs, lr 3e-5). Set `GPTOSS_MODEL_PATH` /
  `FINETUNE_TRAIN` env vars to point at the base model and training set.

## Stage 3 , in-domain validation (silver panel + human)

1. `build_validation_app.py` , build the agreement-stratified 200-item sample, the annotator
   HTML app, `validation_key.csv`, and `validation_populations.json`.
2. `build_validation_sample.py` , alignment-guarded sampler driving the above.
3. `classify_panel.py` , label the sample with the 3-LLM silver panel (Opus 4.8 / GPT-5.5 /
   Gemini 2.5 Pro). Needs all three API keys. Emits `panel_{opus,gpt,gemini}.csv`.
4. `score_panel.py` , panel agreement (Fleiss/Cohen κ), consensus, and HT-reweighted metrics.
5. `score_validation.py` , score against the human gold once `human_labels.csv` exists.

The committed `panel_*.csv` and `validation_populations.json` (in `data/in_domain_validation/`)
are the PII-free outputs of this stage; `contagion_disattenuation.py` reads them to derive the
classifier (precision, recall) used in the criterion's Step 2.
