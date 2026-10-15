"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig bis auf Gleitkommarand; Schwellen daher mit Abstand zur Messung."""

import numpy as np
import pytest

import rgi_constants as C
import rgi_evaluation as ev

BAL, UNEQ = C.DESIGNS
SQ, ML, RG, EB = C.ESTIMATORS
PO, BO, WA = ev.POSTERIOR, ev.BOOTSTRAP, ev.WALD


@pytest.fixture(scope="module")
def main():
    return ev.sweep_main()


@pytest.fixture(scope="module")
def normal():
    return ev.sweep_normal()


@pytest.fixture(scope="module")
def coupling():
    return ev.sweep_coupling()


@pytest.fixture(scope="module")
def stop():
    return ev.sweep_stop(C.EXP_DELTA), ev.sweep_stop(C.DELTA_NARROW)


def b(main, d, g, name, field="bias"):
    return main[d][g]["est"][name][field]


# --- Experiment 1: die Spitze ist überschätzt -------------------------------------------------------------------------------------------------

def test_the_top_is_overestimated_and_shrinkage_removes_it(main):
    assert [b(main, BAL, 66, k) for k in (ML, RG, EB)] == pytest.approx([1.85, 1.16, 0.18], abs=0.15)
    assert [b(main, BAL, 264, k) for k in (ML, RG, EB)] == pytest.approx([0.47, 0.42, 0.16], abs=0.1)
    assert [b(main, UNEQ, 66, k) for k in (ML, RG, EB)] == pytest.approx([2.73, 1.35, 0.18], abs=0.2)
    assert [b(main, UNEQ, 264, k) for k in (ML, RG, EB)] == pytest.approx([0.47, 0.37, 0.02], abs=0.1)
    for d in C.DESIGNS:
        for g in C.EXP_GAMES:
            assert b(main, d, g, ML) > b(main, d, g, RG) > b(main, d, g, EB) - 0.05 and abs(b(main, d, g, EB)) < 0.3, (d, g)
        ml = [b(main, d, g, ML) for g in C.EXP_GAMES]
        assert ml == sorted(ml, reverse=True)                                       # je mehr Partien, desto kleiner die Überschätzung


def test_spread_and_error_of_the_strengths(main):
    assert b(main, BAL, 66, ML, "spread") == pytest.approx(1.65, abs=0.1) and b(main, BAL, 66, EB, "spread") == pytest.approx(0.81, abs=0.07)
    assert b(main, BAL, 264, ML, "spread") == pytest.approx(1.14, abs=0.06) and b(main, UNEQ, 66, ML, "spread") == pytest.approx(2.04, abs=0.15)
    assert [b(main, BAL, 66, k, "mse") for k in (ML, RG, EB)] == pytest.approx([1.39, 0.62, 0.37], abs=0.1)
    assert [b(main, UNEQ, 66, k, "mse") for k in (ML, RG, EB)] == pytest.approx([2.71, 0.91, 0.48], abs=0.2)
    assert [b(main, BAL, 264, k, "mse") for k in (ML, RG, EB)] == pytest.approx([0.164, 0.148, 0.119], abs=0.03)
    for d in C.DESIGNS:
        for g in C.EXP_GAMES:
            assert b(main, d, g, EB, "mse") < b(main, d, g, ML, "mse")
    assert main[BAL][66]["lam_median"] == pytest.approx(0.88, abs=0.4)           # λ des Empirical Bayes: Gitterwert nahe 1/σ² ≈ 0,93


# --- Experiment 2: die Rangfolge --------------------------------------------------------------------------------------------------------------

def test_shrinkage_barely_changes_the_order(main):
    for g in C.EXP_GAMES:
        accs = [b(main, BAL, g, k, "acc") for k in C.ESTIMATORS]
        assert max(accs) - min(accs) < 0.01, g                                    # ausgewogen: gleiche Ordnung
    assert [b(main, BAL, 66, k, "acc") for k in C.ESTIMATORS] == pytest.approx([0.85] * 4, abs=0.02)
    assert [b(main, UNEQ, 66, k, "acc") for k in C.ESTIMATORS] == pytest.approx([0.811, 0.816, 0.819, 0.826], abs=0.015)
    assert b(main, UNEQ, 66, EB, "acc") > b(main, UNEQ, 66, SQ, "acc") + 0.008
    assert b(main, UNEQ, 66, EB, "acc") - b(main, UNEQ, 66, SQ, "acc") < 0.05        # klein
    assert b(main, UNEQ, 264, EB, "acc") - b(main, UNEQ, 264, SQ, "acc") < 0.03


# --- Experiment 3: Breite ---------------------------------------------------------------------------------------------------------------------

def test_interval_width_and_the_third_best(main):
    w = [main[BAL][g]["iv"][PO]["width"] for g in C.EXP_GAMES]
    assert w == pytest.approx([7.21, 5.57, 4.24, 3.34], abs=0.4) and w == sorted(w, reverse=True)
    wu = [main[UNEQ][g]["iv"][PO]["width"] for g in C.EXP_GAMES]
    assert wu == pytest.approx([7.76, 6.29, 4.97, 3.94], abs=0.4) and all(u > v for u, v in zip(wu, w))
    assert main[BAL][66]["third"] == pytest.approx((1.4, 7.4), abs=0.4) and main[BAL][264]["third"] == pytest.approx((1.5, 4.8), abs=0.4)
    assert main[UNEQ][66]["third"] == pytest.approx((1.3, 7.3), abs=0.5)
    assert main[BAL][66]["iv"][PO]["width"] > C.EXP_N / 2                            # „über die Hälfte der Liste“


# --- Experiment 4: Abdeckung ------------------------------------------------------------------------------------------------------------------

def test_coverage_of_the_intervals(main):
    for d in C.DESIGNS:
        for g in C.EXP_GAMES:
            assert main[d][g]["iv"][PO]["cover"] > 0.94, (d, g)                    # Posterior: konservativ (nominal 0,90)
    assert main[BAL][66]["iv"][BO]["cover"] == pytest.approx(0.929, abs=0.03)
    assert main[UNEQ][66]["iv"][BO]["cover"] == pytest.approx(0.892, abs=0.03) and main[UNEQ][66]["iv"][BO]["cover"] < 0.92
    assert main[UNEQ][66]["iv"][BO]["cover"] < main[UNEQ][66]["iv"][PO]["cover"] - 0.04
    assert main[UNEQ][66]["iv"][BO]["by_rank"][0] == pytest.approx(0.77, abs=0.08)       # am schwächsten beim Besten
    assert main[BAL][264]["iv"][BO]["cover"] == pytest.approx(0.975, abs=0.03)
    assert main[UNEQ][66]["iv"][PO]["by_rank"][0] == pytest.approx(0.87, abs=0.08)       # auch das Posterior-Intervall deckt den wahren Besten bei wenigen Partien nicht voll
    assert main[BAL][66]["iv"][WA]["cover"] == pytest.approx(0.958, abs=0.03)
    for g in (66, 264):
        assert main[BAL][g]["iv"][BO]["width"] < main[BAL][g]["iv"][PO]["width"]       # der Bootstrap ist schmaler


def test_with_a_matching_prior_the_posterior_is_still_conservative(normal):
    assert normal[BAL]["iv"][PO]["cover"] == pytest.approx(0.964, abs=0.03) and normal[UNEQ]["iv"][PO]["cover"] == pytest.approx(0.961, abs=0.03)
    assert normal[UNEQ]["iv"][BO]["cover"] == pytest.approx(0.879, abs=0.04) and normal[UNEQ]["iv"][BO]["cover"] < 0.91
    assert normal[UNEQ]["est"][ML]["bias"] == pytest.approx(2.39, abs=0.3) and abs(normal[UNEQ]["est"][EB]["bias"]) < 0.3


# --- Experiment 5: Kalibrierung und Abbruchregel ----------------------------------------------------------------------------------------------

def test_p_best_is_roughly_calibrated(main):
    cal = ev.calibration([main[d][g] for d in C.DESIGNS for g in (66, 264)])
    assert [r[2] for r in cal] == pytest.approx([2084, 377, 215, 141, 63], abs=1)        # 4 Zellen × 60 Läufe × 12 Spieler = 2880
    assert sum(r[2] for r in cal) == 4 * 60 * 12
    for lo, hi, n, said, true in cal:
        assert abs(said - true) < 0.12, (lo, hi)
    assert cal[-1][3] == pytest.approx(0.82, abs=0.05) and cal[-1][4] == pytest.approx(0.78, abs=0.1)


def test_the_stop_rule(stop):
    s, sn = stop
    assert s[0.8]["median_games"] == pytest.approx(396, abs=66) and s[0.9]["median_games"] >= 726
    assert [s[t]["stopped"] for t in C.STOP_THRESHOLDS] == pytest.approx([0.82, 0.47, 0.28], abs=0.1)
    assert [s[t]["hit_stopped"] for t in C.STOP_THRESHOLDS] == pytest.approx([0.86, 0.89, 1.0], abs=0.1)
    assert s[0.9]["hit_stopped"] > 0.8 and s[0.95]["hit_stopped"] > s[0.8]["hit_stopped"] - 0.05
    assert s[0.8]["stopped"] > s[0.9]["stopped"] > s[0.95]["stopped"]
    assert sn[0.9]["stopped"] == pytest.approx(0.10, abs=0.08) and sn[0.8]["hit_all"] == pytest.approx(0.57, abs=0.12)      # enges Feld: kaum ein Halt, viele falsch


# --- Experiment 6: Kopplung ---------------------------------------------------------------------------------------------------------------------

def test_when_the_strongest_play_rarely_shrinkage_costs_hits(coupling):
    ind, most, least = (coupling[c] for c in C.COUPLINGS)
    assert [c["est"][ML]["bias"] for c in (ind, most, least)] == pytest.approx([1.22, 0.58, 2.40], abs=0.3)
    assert least["est"][EB]["winner"] == pytest.approx(0.27, abs=0.1) and least["est"][SQ]["winner"] == pytest.approx(0.55, abs=0.1)
    assert least["est"][EB]["winner"] < least["est"][SQ]["winner"] - 0.15 and least["est"][EB]["top3"] < least["est"][SQ]["top3"]
    assert most["est"][EB]["winner"] >= most["est"][SQ]["winner"] - 0.1 and most["est"][EB]["bias"] < 0
    assert least["iv"][BO]["by_rank"][0] == pytest.approx(0.78, abs=0.1) and least["iv"][PO]["by_rank"][0] > 0.9
    assert ind["est"][EB]["winner"] == pytest.approx(0.55, abs=0.12)


# --- Verdikte nennen die Zahlen aus den Daten ---------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(main, normal, coupling, stop):
    t = ev.verdict_curse(main)[1]
    assert "**+1,85**" in t or "**+1,8" in t or "**+1,9" in t
    assert "66 Partien (jedes Paar einmal)" in t
    o = ev.verdict_order(main)[1]
    assert "Im ausgewogenen Plan ordnen alle vier Schätzer fast gleich" in o and "Shrinkage korrigiert die Werte" in o
    w = ev.verdict_width(main)[1]
    assert "von 12 Rängen" in w and "264 Partien" in w
    cv = ev.verdict_coverage(main, normal)[1]
    assert "Gewünscht 90 %" in cv and "konservativ" in cv
    cal = ev.verdict_calibration(ev.calibration([main[d][g] for d in C.DESIGNS for g in (66, 264)]), stop[0])[1]
    assert "Abbruchregel" in cal and "Halt bei P(Bester) ≥ 0,8" in cal
    cp = ev.verdict_coupling(coupling)[1]
    assert "Sieger richtig" in cp and "Bootstrap-Intervall" in cp
