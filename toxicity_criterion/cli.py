"""Console entrypoints for the package.

Installed as commands by ``pyproject.toml``::

  criterion-run    run the method-selection criterion (see --help; wraps criterion.decision)
  reproduce-all    run every CPU analysis that produces the paper's tables
  make-figures     regenerate the paper's figures into figures/

All analyses are deterministic; run under ``PYTHONHASHSEED=0`` (the Makefile does this for you).
"""

from __future__ import annotations

import importlib

# (module, heading) pairs run in order by reproduce-all. Each module exposes main().
_ANALYSES = [
    ("toxicity_criterion.criterion.local_effect", "Criterion Step 1 , local effect (Table I)"),
    ("toxicity_criterion.criterion.disattenuation", "Criterion Step 2 , disattenuation (Table I)"),
    ("toxicity_criterion.criterion.cascade_structure", "Criterion Step 3 , cascade structure (Table I)"),
    ("toxicity_criterion.criterion.engagement_baseline", "Engagement-baseline foil"),
    ("toxicity_criterion.validation.reach_controlled_counts", "Table II , reach-controlled counts"),
    ("toxicity_criterion.validation.reach_controlled_rates", "Table II , reach-controlled rates"),
    ("toxicity_criterion.validation.toxicity_aware", "Table III , toxicity-aware centralities"),
    ("toxicity_criterion.validation.diffusion", "Table III , diffusion rows"),
    ("toxicity_criterion.validation.incremental_spearman", "Incremental value , Spearman"),
    ("toxicity_criterion.validation.incremental_value", "Incremental value , hybrid/stratum"),
    ("toxicity_criterion.robustness.recall_degradation", "Recall robustness , random + clustered"),
    ("toxicity_criterion.robustness.recall_degradation_adversarial", "Recall robustness , adversarial"),
]

_FIGURES = [
    ("toxicity_criterion.figures.positive_control", "Fig. 3 , positive control"),
    ("toxicity_criterion.figures.reply_subgraphs", "Fig. 2 + Fig. 4 , reply subgraphs"),
]


def _run(modules) -> None:
    for mod, heading in modules:
        print("\n" + "=" * 78 + f"\n{heading}\n" + "=" * 78)
        importlib.import_module(mod).main()


def reproduce_all() -> None:
    """Run every CPU analysis that produces the paper's reported tables."""
    _run(_ANALYSES)


def make_figures() -> None:
    """Regenerate the paper's figures into ``figures/``."""
    _run(_FIGURES)


def criterion_run() -> None:
    """Thin wrapper so the criterion tool is available as the ``criterion-run`` command."""
    from .criterion.decision import main

    main()
