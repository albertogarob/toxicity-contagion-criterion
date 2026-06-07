"""Regression tests for the reach-controlled validation tables, against committed result JSONs.

These assert the paper's reported Table II / Table III cells. They read the generated
``results/*.json`` (run ``make reproduce`` first; the files are PII-free and committed).
"""

from conftest import load_result


def test_table2_counts():
    res = load_result("table2_reach_controlled_counts.json")["results"]
    assert res["ToxicCount"]["direct_mean"] == 2.2
    assert res["ToxicCount"]["subtree_mean"] == 2.3
    assert res["IM"]["direct_mean"] == 1.2
    assert res["IM"]["subtree_mean"] == 1.7


def test_table2_rates():
    res = load_result("table2_reach_controlled_rates.json")["results"]
    assert res["ToxicCount"]["elicit_rate"] == 0.073
    assert res["ToxicCount"]["subtree_rate"] == 0.071
    assert res["Volume"]["subtree_rate"] == 0.059
    assert res["IM"]["elicit_rate"] == 0.029
    assert res["PageRank"]["elicit_rate"] == 0.043


def test_table3_toxicity_aware():
    d = load_result("table3_toxicity_aware.json")
    assert d["ToxicCount"]["subtree_pct"] == 7.1
    assert d["Closeness(blind)"]["subtree_pct"] == 3.9
    assert d["Betweenness(blind)"]["subtree_pct"] == 3.4
    assert d["ToxDegree"]["subtree_pct"] == 11.5  # circular (dagger) row
    assert d["ToxSub-PR"]["subtree_pct"] == 14.2  # circular (dagger) row


def test_table3_diffusion():
    d = load_result("table3_diffusion.json")
    assert d["PPR-clean"]["subtree_pct"] == 9.6
    assert d["NbrToxOut"]["subtree_pct"] == 8.2
    assert d["DiffuseTC"]["subtree_pct"] == 8.0
