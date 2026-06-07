"""Synthetic-data regression tests for the criterion's decision rule (no private data needed).

These lock the criterion's *discrimination*: it must stay dark on a non-contagious corpus and
light up when contagion is planted (the positive-control property the paper relies on).
"""

from toxicity_criterion.criterion.decision import make_demo_forest, run_criterion


def test_null_corpus_recommends_producer_count():
    posts = make_demo_forest(plant_rr=1.0, seed=0)
    v = run_criterion(posts, precision=0.92, recall=0.32, n_perm=80, seed=0)
    assert v["decision"]["use_network_methods"] is False
    assert v["decision"]["recommendation"] == "producer-count suffices"


def test_planted_contagion_recommends_network_methods():
    posts = make_demo_forest(plant_rr=16.0, seed=0)
    v = run_criterion(posts, precision=0.92, recall=0.32, n_perm=80, seed=0)
    assert v["decision"]["use_network_methods"] is True
    assert v["step1_local_effect"]["relative_risk"] > 2.0


def test_disattenuation_identity():
    # r_true = r_obs / kappa^2 with kappa from (precision, recall, q); monotone in recall.
    from toxicity_criterion.criterion.decision import disattenuate

    lo = disattenuate(0.046, precision=0.92, recall=0.32, q=0.0176)
    hi = disattenuate(0.046, precision=0.92, recall=0.77, q=0.0176)
    assert lo["r_true"] > hi["r_true"] > 0  # lower recall -> larger correction
    assert 0.10 < lo["r_true"] < 0.25
