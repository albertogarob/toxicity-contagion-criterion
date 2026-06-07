"""Regression tests for the criterion on the real corpus (skip if PII data absent).

These lock the paper's reported Table I values (Steps 1-3) and the engagement foil.
"""

from conftest import requires_corpus

from toxicity_criterion.criterion import cascade_structure, engagement_baseline, local_effect


@requires_corpus
def test_step1_local_effect():
    r = local_effect.run()
    assert round(r["assortativity"], 3) == 0.046
    assert round(r["relative_risk"], 2) == 3.35
    assert r["largest_component"] == 7
    assert r["assortativity_null"]["p"] < 0.01  # significant
    assert r["largest_component_null"]["p"] > 0.5  # component at chance
    assert round(r["base_rate"], 4) == 0.0176


@requires_corpus
def test_step3_cascade_structure():
    r = cascade_structure.run()
    assert r["deepest_chain"] == 3
    assert round(r["deepest_chain_null"]["mean"], 2) == 2.02
    assert r["chain_histogram"] == {1: 582, 2: 17, 3: 1}


@requires_corpus
def test_engagement_foil():
    r = engagement_baseline.run()
    st = r["structural_transmission"]
    assert round(st["frac_toxic_isolated"], 2) == 0.94
    assert round(st["P(reply_toxic|parent_toxic)"], 3) == 0.059
    et = r["engagement_proxy_reply_count"]["toxic_parents"]
    en = r["engagement_proxy_reply_count"]["nontoxic_parents"]
    assert et["mean"] > en["mean"]  # engagement reads it "contagious"...
    assert st["frac_toxic_isolated"] > 0.9  # ...yet transmission is absent
