"""Analyse des gezeigten Laufs, Verdikt-Zweige an handgesetzten Werten und die Hilfsfunktionen; die Experimente selbst prüft test_claims.py."""

import numpy as np
import pytest

import rgi_constants as C
import rgi_evaluation as ev
from rgi_scenario import Settings

BAL, UNEQ = C.DESIGNS


def test_german_number_formatting():
    assert ev.de(1.5, 1) == "1,5" and ev.de(0.666) == "0,67" and ev.pct(0.256) == "26" and ev.pct(0.256, 1) == "25,6"
    assert ev.signed(1.234) == "+1,23" and ev.signed(-0.5) == "−0,50" and ev.signed(0.0) == "+0,00"


def test_mean_skips_nan():
    assert ev._mean([1.0, float("nan"), 3.0]) == 2.0 and np.isnan(ev._mean([float("nan")]))


def test_analysis_is_consistent_and_deterministic():
    s = Settings(n=8, design=BAL, games=56, seed=2)
    a = ev.analyse(s)
    assert a.instance.games == 56 and len(a.lo) == 8 and (a.lo <= a.hi).all() and (a.lo >= 1).all() and (a.hi <= 8).all()
    assert sorted(a.true_rank.tolist()) == list(range(1, 9)) and sorted(a.est_rank.tolist()) == list(range(1, 9))
    assert a.width == pytest.approx(float((a.hi - a.lo + 1).mean())) and 0.0 <= a.cover <= 1.0 and abs(a.p_best.sum() - 1.0) < 1e-9
    b = ev.analyse(s)
    assert np.array_equal(a.lo, b.lo) and a.est[ev.EB]["acc"] == b.est[ev.EB]["acc"]


def test_interval_method_and_level_change_the_width():
    base = dict(n=10, design=UNEQ, games=100, seed=5)
    wide = ev.analyse(Settings(**base, level=95)).width
    narrow = ev.analyse(Settings(**base, level=50)).width
    assert wide > narrow
    boot = ev.analyse(Settings(**base, interval=C.INTERVALS[1]))
    assert boot.lo.shape == (10,) and boot.settings.interval == C.INTERVALS[1]


def test_estimator_choice_changes_the_shown_rank_only_in_the_estimator():
    s = Settings(n=8, design=UNEQ, games=40, seed=3)
    a = ev.analyse(s)
    b = ev.analyse(Settings(n=8, design=UNEQ, games=40, seed=3, estimator=ev.ML))
    assert np.array_equal(a.lo, b.lo) and a.est[ev.ML]["bias"] == b.est[ev.ML]["bias"]


def test_calibration_helper_by_hand():
    cell = {"pbest": (np.array([0.01, 0.1, 0.3, 0.5, 0.9, 0.8]), np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]))}
    rows = ev.calibration([cell])
    assert [r[2] for r in rows] == [1, 1, 1, 1, 2]
    assert rows[-1][3] == pytest.approx(0.85) and rows[-1][4] == pytest.approx(1.0) and rows[2][4] == pytest.approx(1.0)
    empty = ev.calibration([{"pbest": (np.array([0.5]), np.array([0.0]))}])
    assert np.isnan(empty[0][3]) and empty[3][2] == 1


# --- Verdikt-Zweige ---------------------------------------------------------------------------------------------------------------------------

def _est(**kw):
    out = {}
    for name in C.ESTIMATORS:
        out[name] = {"acc": 0.8, "winner": 0.5, "top3": 0.7, "bias": 0.1, "spread": 1.0, "mse": 0.2}
    for key, val in kw.items():
        name, field = key.split("__")
        out[{"siegquote": ev.SIEGQUOTE, "ml": ev.ML, "ridge": ev.RIDGE, "eb": ev.EB}[name]][field] = val
    return out


def _iv(cover=0.9, width=5.0, by_rank=(0.9, 0.9, 0.9, 0.9)):
    return {"cover": cover, "width": width, "by_rank": list(by_rank)}


def _cell(games=66, est=None, iv=None, third=(1.0, 6.0)):
    return {"games": games, "est": est or _est(), "iv": iv or {ev.POSTERIOR: _iv(), ev.WALD: _iv(), ev.BOOTSTRAP: _iv()}, "third": third,
            "pbest": (np.array([0.5]), np.array([1.0]))}


def _main(**kw):
    return {d: {g: _cell(g, **kw) for g in C.EXP_GAMES} for d in C.DESIGNS}


def test_curse_verdict_quotes_the_numbers():
    main = _main(est=_est(ml__bias=1.5, ridge__bias=0.9, eb__bias=0.03, ml__mse=1.5, eb__mse=0.4))
    level, t = ev.verdict_curse(main)
    assert level == "warning" and "+1,50" in t and "+0,90" in t and "+0,03" in t and "von 1,50 auf 0,40" in t and "66 Partien" in t


def test_order_verdict_quotes_the_numbers():
    est = _est(siegquote__acc=0.866, ml__acc=0.866, eb__acc=0.862)
    t = ev.verdict_order(_main(est=est))[1]
    assert "0,866" in t and "0,862" in t and "ändert sie wenig" in t


def test_width_verdict_quotes_interval_and_widths():
    main = _main(third=(1.3, 7.3), iv={ev.POSTERIOR: _iv(width=7.14), ev.WALD: _iv(), ev.BOOTSTRAP: _iv()})
    t = ev.verdict_width(main)[1]
    assert "Rang **1,3** bis **7,3**" in t and "7,1 von 12" in t and "264 Partien: 1,3 bis 7,3" in t


def test_coverage_verdict_quotes_methods():
    main = _main(iv={ev.POSTERIOR: _iv(0.968), ev.WALD: _iv(0.958), ev.BOOTSTRAP: _iv(0.929)})
    normal = {d: _cell(iv={ev.POSTERIOR: _iv(0.961), ev.WALD: _iv(), ev.BOOTSTRAP: _iv(0.879)}) for d in C.DESIGNS}
    t = ev.verdict_coverage(main, normal)[1]
    assert "Posterior **97 %**" in t and "Wald 96 %" in t and "Bootstrap der Partien 93 %" in t and "**93 %**" in t and "96 % gegen 88 %" in t


def test_calibration_and_stop_verdict():
    cal = [(0.0, 0.05, 10, 0.01, 0.0), (0.7, 1.0, 63, 0.82, 0.78)]
    stop = {0.8: {"median_games": 396.0, "stopped": 0.82, "hit_stopped": 0.86, "hit_all": 0.8}, 0.9: {"median_games": 792.0, "stopped": 0.47, "hit_stopped": 0.89, "hit_all": 0.78}}
    t = ev.verdict_calibration(cal, stop)[1]
    assert "sagt er im Mittel 0,82 und trifft 0,78 (n = 63)" in t and "82 % der Läufe (Median 396 Partien), Treffer 0,86" in t and "47 % der Läufe, Treffer 0,89" in t


def test_coupling_verdict_quotes_the_numbers():
    iv = {ev.POSTERIOR: _iv(by_rank=(0.97, 0.9, 0.9, 0.9)), ev.WALD: _iv(), ev.BOOTSTRAP: _iv(by_rank=(0.78, 0.9, 0.9, 0.9))}
    cp = {C.COUPLINGS[0]: _cell(est=_est(ml__bias=1.22)), C.COUPLINGS[1]: _cell(est=_est(ml__bias=0.58)),
          C.COUPLINGS[2]: _cell(est=_est(ml__bias=2.4, eb__winner=0.27, siegquote__winner=0.55), iv=iv)}
    t = ev.verdict_coupling(cp)[1]
    assert "+2,40" in t and "unabhängig +1,22" in t and "am meisten +0,58" in t and "**0,27** gegen 0,55" in t and "nur 78 %" in t and "Posterior 97 %" in t


def test_measure_marks_the_win_quote_without_spread_and_the_width_is_hi_minus_lo_plus_one():
    from rgi_scenario import simulate
    inst = simulate(8, 0.3, BAL, 56, 10, C.TRUTHS[0], C.COUPLINGS[0], 4)
    m = ev.measure(inst, 0.1, 4, boot=True)
    assert np.isnan(m["est"][ev.SIEGQUOTE]["spread"]) and np.isnan(m["est"][ev.SIEGQUOTE]["mse"])
    assert all(np.isfinite(m["est"][k]["spread"]) and np.isfinite(m["est"][k]["mse"]) for k in (ev.ML, ev.RIDGE, ev.EB))
    for method, v in m["iv"].items():
        assert (v["width"] == v["hi"] - v["lo"] + 1).all() and (v["width"] >= 1).all() and (v["lo"] >= 1).all() and (v["hi"] <= 8).all(), method
    assert set(m["iv"]) == {ev.POSTERIOR, ev.WALD, ev.BOOTSTRAP} and abs(m["pbest"].sum() - 1.0) < 1e-9
    assert set(ev.measure(inst, 0.1, 4, boot=False)["iv"]) == {ev.POSTERIOR, ev.WALD}


def test_calibration_bins_include_their_lower_bound_and_the_last_bin_ends_at_one():
    cell = {"pbest": (np.array([0.05, 0.2, 0.4, 0.7, 1.0, 0.0]), np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]))}
    rows = ev.calibration([cell])
    assert [r[2] for r in rows] == [1, 1, 1, 1, 2]
    assert rows[0][3] == pytest.approx(0.0) and rows[1][3] == pytest.approx(0.05) and rows[2][3] == pytest.approx(0.2)
    assert rows[3][3] == pytest.approx(0.4) and rows[4][3] == pytest.approx(0.85) and rows[-1][1] == 1.0 and rows[0][0] == 0.0


def test_cell_pools_one_best_player_per_run():
    c = ev.cell(BAL, 66)
    P, Tt = c["pbest"]
    assert P.shape == (len(C.SWEEP_SEEDS) * C.EXP_N,) and Tt.sum() == len(C.SWEEP_SEEDS)
    assert np.allclose(P.reshape(len(C.SWEEP_SEEDS), C.EXP_N).sum(axis=1), 1.0)
    assert set(c["iv"]) == {ev.POSTERIOR, ev.WALD} and len(c["iv"][ev.POSTERIOR]["by_rank"]) == 4 and c["est"][ev.EB]["spread"] > 0


def test_stop_runs_play_whole_rounds_up_to_the_budget():
    rows = ev._stop_run(C.DELTA_NARROW, C.SWEEP_SEEDS[0])
    assert [r[0] for r in rows] == [66 * k for k in range(1, len(rows) + 1)] and len(rows) <= C.STOP_MAX_ROUNDS
    strong = ev._stop_run(1.0, C.SWEEP_SEEDS[1])
    assert len(strong) < C.STOP_MAX_ROUNDS and strong[-1][1] >= max(C.STOP_THRESHOLDS) and strong[-1][2] is True     # starkes Signal: früh sicher und richtig
    never = [ev._stop_run(C.DELTA_NARROW, sd) for sd in C.SWEEP_SEEDS[:8]]
    assert any(len(r) == C.STOP_MAX_ROUNDS and r[-1][0] == 792 for r in never)


def test_stop_summary_uses_the_budget_when_the_threshold_is_never_met():
    s = ev.sweep_stop(C.DELTA_NARROW)
    assert s[0.95]["median_games"] == 792.0 and 0.0 <= s[0.95]["stopped"] < 0.2
    assert s[0.8]["stopped"] >= s[0.9]["stopped"] >= s[0.95]["stopped"]


def _fake_analysis(design, acc_eb, acc_sq, n=8, tr=None, lo=None, hi=None):
    from types import SimpleNamespace
    tr = np.array(tr if tr is not None else [3, 1, 2, 4, 5, 6, 7, 8])
    est = {k: {"acc": 0.8, "bias": 0.1, "winner": 0.5, "top3": 0.6} for k in C.ESTIMATORS}
    est[ev.EB]["acc"], est[ev.SIEGQUOTE]["acc"] = acc_eb, acc_sq
    s = SimpleNamespace(n=n, level=90, interval=C.INTERVALS[0], design=design)
    return SimpleNamespace(settings=s, true_rank=tr, lo=np.array(lo if lo is not None else [2, 1, 1, 3, 4, 5, 6, 7]), hi=np.array(hi if hi is not None else [6, 3, 4, 6, 7, 8, 8, 8]),
                           width=4.0, cover=0.9, est=est)


def test_analysis_verdict_names_the_third_best_and_the_balanced_remark():
    a = _fake_analysis(BAL, 0.80, 0.81)
    t = ev.verdict_analysis(a)[1]
    assert "von Rang **2** bis **6**" in t and "Im ausgewogenen Plan ändert die Schrumpfung die Ordnung kaum" in t
    assert "Im ausgewogenen Plan" not in ev.verdict_analysis(_fake_analysis(BAL, 0.80, 0.84))[1]
    assert "Im ausgewogenen Plan" not in ev.verdict_analysis(_fake_analysis(UNEQ, 0.80, 0.80))[1]
    two = ev.verdict_analysis(_fake_analysis(BAL, 0.8, 0.8, n=2, tr=[1, 2], lo=[1, 1], hi=[1, 2]))[1]
    assert "von Rang **1** bis **2**" in two                                    # min(3, n): bei zwei Spielern der Zweite


def test_stop_rule_halts_exactly_at_the_threshold(monkeypatch):
    import rgi_intervals as IV
    n = C.EXP_N

    def make(top):
        rest = (1.0 - top) / (n - 1)
        return lambda ranks: np.array([top] + [rest] * (n - 1))

    ev._stop_run.cache_clear()
    monkeypatch.setattr(IV, "p_best", make(0.95))
    assert len(ev._stop_run(0.3, 123456)) == 1                                   # P = Schwelle: Halt in der ersten Runde
    ev._stop_run.cache_clear()
    monkeypatch.setattr(IV, "p_best", make(0.9499))
    assert len(ev._stop_run(0.3, 123456)) == C.STOP_MAX_ROUNDS
    ev._stop_run.cache_clear()


def test_stop_summary_counts_a_run_as_stopped_when_p_equals_the_threshold(monkeypatch):
    rows = ((66, 0.8, True), (132, 0.9, False))
    monkeypatch.setattr(ev, "_stop_run", lambda delta, seed: rows)
    s = ev.sweep_stop(0.3)
    assert s[0.8]["median_games"] == 66.0 and s[0.8]["stopped"] == 1.0 and s[0.8]["hit_stopped"] == 1.0
    assert s[0.9]["median_games"] == 132.0 and s[0.9]["hit_stopped"] == 0.0
    assert s[0.95]["stopped"] == 0.0 and s[0.95]["median_games"] == 792.0 and s[0.95]["hit_all"] == 0.0
