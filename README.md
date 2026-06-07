# Testing for Toxicity Contagion Before Applying Network Amplifier-Ranking Methods: A UK Reddit Case Study

Reproducibility repository for the paper submitted to the **IEEE ICDM 2026 Applied Track**.

## Overview

The paper's lead contribution is a **contagion-based method-selection criterion**: a cheap
pre-check that decides whether the toxicity in a corpus is contagious/cascading, and therefore
whether network-based amplifier-ranking methods are worth using at all. The criterion is
validated by a generative positive control (it detects planted cascades and reports their
absence on real data) and by a case study whose head-to-head outcome confirms its
recommendation:

> The criterion reports "no contagion, no cascades" on UK Reddit immigration discourse,
> predicts network methods will not pay off, and that is confirmed: a producer-count baseline
> beats centrality and influence maximization on a reach-controlled validation.

Corpus: 35,176 comments (after cleaning) from five UK subreddits
(r/unitedkingdom, r/askuk, r/ukpolitics, r/ukvisa, r/skilledworkervisauk), collected Nov–Dec 2025
by querying Reddit's public per-subreddit comment listings (the unauthenticated `.json` web
endpoints, e.g. `reddit.com/r/<sub>/comments.json`; not the official Reddit Data API), labelled
toxic/non-toxic (1.76% toxic).

## Repository layout

```
toxicity_criterion/            installable package
├── config.py                  single source of truth for the data/figures/results layout
├── io.py  forest.py  stats.py shared utilities (loaders, reply-forest, statistics)
├── graph.py  influence.py     reply-graph construction; Independent Cascade + CELF++
├── criterion/                 the method-selection criterion
│   ├── local_effect.py        Step 1 , parent->child assortativity + RR vs null
│   ├── disattenuation.py      Step 2 , correct for classifier error (+ generative control)
│   ├── cascade_structure.py   Step 3 , component size + cascade depth vs null
│   ├── engagement_baseline.py engagement-vs-transmission foil
│   └── decision.py            run_criterion() + decision rule (the deployable tool)
├── validation/                reach-controlled head-to-head (Tables II, III)
│   ├── reach_controlled_counts.py / reach_controlled_rates.py
│   ├── toxicity_aware.py / diffusion.py
│   └── incremental_spearman.py / incremental_value.py
├── robustness/                recall-degradation stress tests
├── figures/                   positive_control.py (Fig 3), reply_subgraphs.py (Figs 2, 4)
├── labelling/                 Stage 2/3 labelling + fine-tune (API/GPU; document-only)
├── scraping/                  Stage 1 Reddit collection (document-only)
└── cli.py                     console entrypoints
data/        corpus data , withheld pending institutional approval (gitignored), see data/README.md
docs/        methodology docs       figures/  results/     tests/  regression suite
pyproject.toml  Makefile  requirements.txt
```

## Setup (uv)

```bash
make setup          # uv venv --python 3.12 .venv  &&  uv pip install -e ".[dev]"
# or manually:
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

The analyses are deterministic (`SEED = 42`, `PYTHONHASHSEED = 0`; the Makefile sets the hash
seed) and run **on CPU in minutes** once the corpus is present locally. No API key or GPU is
needed for the analyses themselves.

```bash
make reproduce      # run every analysis that produces the paper's tables -> results/
make figures        # regenerate Figs 2-4 -> figures/
make test           # regression tests that lock the paper's numbers
make lint           # ruff + black --check
```

## Data , not shared yet (pending institutional approval)

**No corpus data is shared in this repository at this time** , not even a dehydrated, text-free
id+label file. Three coauthors are at a university in the Netherlands, so any data release
(including dehydrated IDs, which remain re-identifiable on rehydration and so are personal data
under the GDPR) must first be cleared by the university privacy officer (FG/DPO) and
research-ethics committee. Until then the repo ships only code, aggregate `results/*.json`, and
the anonymized figures.

The sharing **mechanism is already built** and will be enabled on approval: a dehydrated dataset
(`build_dehydrated.py` -> `data/public/corpus_dehydrated.csv`) plus a rehydration script
(`make rehydrate`) that re-fetches text by ID from Reddit's public `.json` endpoints and honours deletions. See
[`DATA_RELEASE.md`](DATA_RELEASE.md) (availability + ethics) and [`data/README.md`](data/README.md).

> Reproducing the tables therefore currently requires access to the corpus from the authors
> (under the institutional approval / PC confidentiality). The committed `results/*.json` provide
> the reference values, and `tests/` assert them.

## Reproducing the paper's results

Run from the repo root with the hash seed pinned. Each module has a `run()` API and a
`python -m` entrypoint; `make reproduce` runs them all.

| Paper artifact | Command | Key reproduced values |
|---|---|---|
| **Table I**, Step 1 + clusters | `python -m toxicity_criterion.criterion.local_effect` | RR 3.35×, r=+0.046 (null 0.000±0.008, p≈0.003), largest component 7 (null 7.5±2.0, p=0.65) |
| **Table I**, Step 2 (disattenuation) | `python -m toxicity_criterion.criterion.disattenuation` | closed-form r_true ∈ [0.09, 0.31]; generative 0.07–0.19 |
| **Table I**, Step 3 (cascade depth) | `python -m toxicity_criterion.criterion.cascade_structure` | deepest chain 3 (null 2.02±0.18); 582/17/1 singletons/pairs/triple |
| **Fig. 3** (positive control) | `python -m toxicity_criterion.figures.positive_control` | `figures/fig_poscontrol.pdf`; components rise to 40–50 as RR grows |
| Engagement-baseline foil | `python -m toxicity_criterion.criterion.engagement_baseline` | toxic 0.52 vs 0.43 replies; 38% vs 34%; 94% isolated; 5.9% reply-toxic |
| **Table II** (count columns) + graph | `python -m toxicity_criterion.validation.reach_controlled_counts` | 13,688 nodes / 12,018 edges; ToxicCount 2.2/2.3, IM 1.2/1.7 |
| **Table II** (rate columns) | `python -m toxicity_criterion.validation.reach_controlled_rates` | ToxicCount 7.3/7.1, IM 2.9/3.1, PageRank 4.3/3.6, Random 3.6/3.0 |
| **Table III**, topology/producer/circular | `python -m toxicity_criterion.validation.toxicity_aware` | Closeness 4.5/3.9, Betweenness 4.5/3.4, ToxDegree† 12.2/11.5, ToxSub-PR† 16.4/14.2 |
| **Table III**, diffusion rows | `python -m toxicity_criterion.validation.diffusion` | PPR 10.4/9.6 (n=7), Neighbor 9.4/8.2 (n=9), Diffused 9.0/8.0 (n=10) |
| Incremental-value (§toxaware) | `python -m toxicity_criterion.validation.incremental_spearman` | Volume ρ=0.35, ToxicCount 0.26, network 0.16–0.19; ΔCI spans 0 |
| Incremental-value (hybrid/stratum) | `python -m toxicity_criterion.validation.incremental_value` | no hybrid beats plain toxic-count |
| **Fig. 2 & Fig. 4** | `python -m toxicity_criterion.figures.reply_subgraphs` | `figures/fig_examples.pdf`, `figures/fig_reach.pdf` (comp 7, chain 3, reach 61) |
| Recall robustness (random + clustered) | `python -m toxicity_criterion.robustness.recall_degradation` | producer-count wins up to 50% toxic-label loss |
| Recall robustness (adversarial) | `python -m toxicity_criterion.robustness.recall_degradation_adversarial` | robust to clustered; flips only under extreme hub-targeted dropout |
| **Fig. 1** (method dataflow) | `npx -p @mermaid-js/mermaid-cli mmdc -i figures/fig_diagnostic.mmd -o figures/fig_diagnostic.pdf -b transparent` | `figures/fig_diagnostic.pdf` |

> Notes
> - Table II rates and Table III topology/producer rows are the same reach-controlled numbers,
>   split across `reach_controlled_*` and `toxicity_aware`/`diffusion` for the two tables.
> - The `†` Table III rows (toxic-graph weighted degree / PageRank) are kept to *exhibit, not
>   endorse* a label-leakage tautology, as the paper explains.

## The criterion as a reusable tool

[`criterion/decision.py`](toxicity_criterion/criterion/decision.py) packages the whole criterion
(Steps 1–3 + decision rule) so a practitioner can run the pre-check on **any** corpus: give it a
reply forest and the classifier's `(precision, recall)`, and it outputs producer-count vs network
methods. Installed as `criterion-run`; depends only on `networkx`; needs no private data.

```bash
criterion-run --demo --precision 0.92 --recall 0.32              # null demo -> producer-count
criterion-run --demo --plant-rr 16 --precision 0.92 --recall 0.32 # planted -> network methods
criterion-run --input forest.jsonl --precision 0.92 --recall 0.32 0.77   # real forest, recall band
```

On this paper's corpus it reproduces the case-study reading exactly (r=+0.046, RR 3.35×,
corrected r_true ≈ 0.07–0.16, component 7 at chance, chain 3, 94% isolated → **producer-count
suffices**). Thresholds are documented and overridable.

## Tests

`make test` runs a pytest regression suite that locks the paper's reported values:
`tests/test_decision.py` (synthetic; always runs), `tests/test_criterion.py` (Table I; skips if
the PII corpus is absent), and `tests/test_validation.py` (Tables II/III, from committed
`results/*.json`).

## API / GPU stages (document-only)

`toxicity_criterion/labelling/` (Sonnet labelling, 3-LLM silver panel, GPT-OSS LoRA fine-tune)
and `toxicity_criterion/scraping/` are **not re-run** for reproduction: they are costly,
non-deterministic, and their outputs already ship as the Stage 2 label files. Install their
deps with `uv pip install -e ".[labelling]"` and/or `".[finetune]"`. See
[`toxicity_criterion/labelling/README.md`](toxicity_criterion/labelling/README.md).

**GPT-OSS LoRA adapter (stratifier).** A LoRA adapter on `openai/gpt-oss-20b` was used only to
define the validation strata (not the deployed labeller). Its model card is
[`docs/MODEL_CARD_gptoss_lora.md`](docs/MODEL_CARD_gptoss_lora.md) and `upload_adapter.py` pushes
it to the HuggingFace Hub. The adapter is trained on the corpus, so its upload is **gated on the
same institutional approval as the data** (see `DATA_RELEASE.md`) and is not yet published.

## Determinism and run parameters

`SEED = 42`; permutation/degradation/contagion pin `PYTHONHASHSEED = 0`. Permutation null
`B = 300`; recall degradation 20 perturbations × 6 fractions; influence maximization mean over
5 runs (`R = 200`). Original runs: CPU experiments on a commodity laptop (minutes); fine-tune on
an NVIDIA RTX 4000 Ada (6 h 21 m); panel ~10–15 min API wall-clock.

## Documentation

- [`docs/CONTAGION_DISATTENUATION.md`](docs/CONTAGION_DISATTENUATION.md) , the criterion (disattenuation + generative control + cascade tests)
- [`docs/RECALL_ROBUSTNESS.md`](docs/RECALL_ROBUSTNESS.md) , recall reality + degradation robustness
- [`docs/VALIDATION_SET_DESIGN.md`](docs/VALIDATION_SET_DESIGN.md) , validation design + estimator
- [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md) , labelling rubric
- [`docs/MODEL_CARD_gptoss_lora.md`](docs/MODEL_CARD_gptoss_lora.md) , the GPT-OSS LoRA stratifier adapter

## Authors

- Fatemeh Akrami (University of Twente)
- Alexandru Ilies (University of Twente)
- Alberto Garcia-Robledo (SECIHTI-CentroGeo)
- Mahboobeh Zangiabady (University of Twente)

## License

MIT (code). Please cite the paper if you reference this work.
