"""Eigenschaftstests (metamorph) zu Rangintervallen und Shrinkage (ORACLE-PLAYBOOK §5.2). Aussagen, die gelten müssen, ohne dass das richtige Ergebnis bekannt ist.

Je Eigenschaft 12 bis 40 Zufallsinstanzen mit festem Seed plus Randfälle. Voraussetzungen stehen beim Test:
- Intervalle aus Ziehungen (Monte Carlo) sind unter Umnummerieren nur bis auf Ziehungsrauschen gleich (die Cholesky-Zerlegung ist nicht umnummerierungs-gleich); exakt gilt die
  Aussage für die Rangmatrix, die Quantilintervalle und alle Größen vor den Ziehungen.
- „Das Intervall enthält den Rang der Punktschätzung“ folgt nicht aus der Definition (bei sehr ungleichen Unsicherheiten kann der Rang der Schätzung außerhalb liegen); geprüft wird sie
  hier für das 90-%-Intervall bei den Plänen der Demo (ausgewogen und ungleich, 6 bis 12 Spieler), nicht als allgemeiner Satz.
- Streuung nach Shrinkage ≤ vor Shrinkage gilt exakt: Der Ridge-Schätzer hat Mittelwert 0 (die Summe der Gradienten der Likelihood ist 0), und die Norm des Ridge-Schätzers fällt
  mit λ; das Gitter des Empirical Bayes beginnt bei λ = 0,01 = ML_LAM."""

import numpy as np
import pytest

import rgi_constants as C
import rgi_evaluation as ev
import rgi_intervals as IV
import rgi_model as M
import rgi_scenario as SC
import turnier_generator as T

BAL, UNEQ = C.DESIGNS


def _inst(design, games, seed, n=12, truth=C.TRUTHS[0]):
    return SC.simulate(n, 0.3, design, games, 10, truth, C.COUPLINGS[0], seed)


def _ranks(inst, seed, draws=600):
    fit = ev.fit_all(inst, 0.1)
    return fit, ev.rank_draws(inst, fit, seed, boot=False, draws=draws)


# --- 1. Aufbau der Intervalle ------------------------------------------------------------------------------------------------------------------

def test_intervals_lie_in_one_to_n_contain_the_median_and_the_rank_of_the_point_estimate():
    rng = np.random.default_rng(20261017)
    for k in range(24):
        n = int(rng.integers(6, 13))
        design = (BAL, UNEQ)[k % 2]
        games = int(rng.choice((40, 66, 132, 264)))
        seed = int(rng.integers(0, 10 ** 6))
        inst = _inst(design, games, seed, n)
        fit, rk = _ranks(inst, seed)
        for name, v in ((ev.POSTERIOR, fit.values[ev.EB]), (ev.WALD, fit.values[ev.ML])):
            ranks = rk[name]
            d = ranks.shape[0]
            assert np.array_equal(np.sort(ranks, axis=1), np.tile(np.arange(1, n + 1), (d, 1)))        # jede Ziehung ordnet alle Spieler
            assert IV.p_best(ranks).sum() == pytest.approx(1.0)
            for level in (0.5, 0.9):
                lo, hi = IV.interval(ranks, level)
                med = np.sort(ranks, axis=0)[(d - 1) // 2]
                assert (lo >= 1).all() and (hi <= n).all() and (lo <= hi).all() and (lo <= med).all() and (med <= hi).all()
            lo, hi = IV.interval(ranks, 0.9)
            point = M.ranks_from(v)
            assert ((point >= lo) & (point <= hi)).all(), (name, design, games, seed, n)


def test_a_higher_level_gives_nested_intervals_and_the_full_level_gives_min_and_max():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = int(rng.integers(4, 13))
        values = rng.normal(size=(int(rng.integers(50, 400)), n)) * rng.uniform(0.2, 2.0, size=n)
        ranks = IV.rank_matrix(values)
        prev_lo, prev_hi = None, None
        for level in (0.1, 0.5, 0.8, 0.9, 0.95, 0.99):
            lo, hi = IV.interval(ranks, level)
            if prev_lo is not None:
                assert (lo <= prev_lo).all() and (hi >= prev_hi).all()
            prev_lo, prev_hi = lo, hi
        lo1, hi1 = IV.interval(ranks, 1.0)
        assert (lo1 == ranks.min(axis=0)).all() and (hi1 == ranks.max(axis=0)).all()
        assert (lo1 <= prev_lo).all() and (hi1 >= prev_hi).all()


# --- 2. Mehr Partien verengen die Intervalle (Mittel über Seeds) ---------------------------------------------------------------------------

@pytest.mark.parametrize("design", [BAL, UNEQ])
def test_more_games_narrow_the_mean_interval_width_and_lower_the_error(design):
    seeds = range(500, 512)
    widths, errors = [], []
    for games in (66, 132, 264, 528):
        w, e = [], []
        for seed in seeds:
            inst = _inst(design, games, seed)
            fit, rk = _ranks(inst, seed, draws=400)
            lo, hi = IV.interval(rk[ev.POSTERIOR], 0.9)
            w.append(float((hi - lo + 1).mean()))
            e.append(M.mse(fit.values[ev.EB], inst.strengths))
        widths.append(np.mean(w))
        errors.append(np.mean(e))
    assert all(a > b for a, b in zip(widths, widths[1:])), widths
    assert all(a > b for a, b in zip(errors, errors[1:])), errors


# --- 3. Umnummerieren permutiert die Intervalle ------------------------------------------------------------------------------------------------

def test_relabelling_permutes_ranks_intervals_p_best_and_coverage_exactly():
    rng = np.random.default_rng(3)
    for _ in range(40):
        n = int(rng.integers(3, 13))
        values = rng.normal(size=(int(rng.integers(30, 300)), n))
        p = rng.permutation(n)
        r, r2 = IV.rank_matrix(values), IV.rank_matrix(values[:, p])
        assert np.array_equal(r2, r[:, p])
        level = float(rng.choice((0.5, 0.8, 0.9)))
        lo, hi = IV.interval(r, level)
        lo2, hi2 = IV.interval(r2, level)
        assert np.array_equal(lo2, lo[p]) and np.array_equal(hi2, hi[p])
        assert np.array_equal(IV.p_best(r2), IV.p_best(r)[p])
        true = M.ranks_from(rng.normal(size=n))
        assert np.array_equal(IV.covered(lo2, hi2, true[p]), IV.covered(lo, hi, true)[p])


def test_relabelling_the_players_permutes_the_fit_the_empirical_bayes_lambda_and_the_covariance():
    rng = np.random.default_rng(4)
    for _ in range(20):
        n = int(rng.integers(5, 10))
        inst = _inst(BAL if rng.random() < 0.5 else UNEQ, int(rng.choice((40, 100, 200))), int(rng.integers(0, 10 ** 6)), n)
        W, G = inst.W, inst.G
        p = rng.permutation(n)
        W2, G2 = W[np.ix_(p, p)], G[np.ix_(p, p)]
        lam, s, Sigma = M.empirical_bayes(W, G)
        lam2, s2, Sigma2 = M.empirical_bayes(W2, G2)
        assert lam2 == lam
        assert s2 == pytest.approx(s[p], abs=1e-6) and Sigma2 == pytest.approx(Sigma[np.ix_(p, p)], abs=1e-6)
        assert M.win_quote(W2, G2) == pytest.approx(M.win_quote(W, G)[p])


def test_relabelling_the_players_changes_the_posterior_intervals_only_by_simulation_noise():
    rng = np.random.default_rng(5)
    for _ in range(10):
        n = 7
        inst = _inst(BAL, int(rng.choice((42, 84))), int(rng.integers(0, 10 ** 6)), n)
        p = rng.permutation(n)
        lam, s, Sigma = M.empirical_bayes(inst.W, inst.G)
        lo, hi = IV.interval(IV.posterior_ranks(s, Sigma, T.SplitMix64(1), 8000), 0.9)
        lo2, hi2 = IV.interval(IV.posterior_ranks(s[p], Sigma[np.ix_(p, p)], T.SplitMix64(2), 8000), 0.9)
        assert np.abs(lo2 - lo[p]).max() <= 1 and np.abs(hi2 - hi[p]).max() <= 1


# --- 4. Shrinkage ------------------------------------------------------------------------------------------------------------------------------

def test_shrinkage_lowers_the_spread_of_the_estimates_and_pulls_them_to_the_mean():
    rng = np.random.default_rng(6)
    for k in range(30):
        n = int(rng.integers(4, 13))
        inst = _inst((BAL, UNEQ)[k % 2], int(rng.choice((30, 66, 200))), int(rng.integers(0, 10 ** 6)), n)
        W, G = inst.W, inst.G
        stds = []
        for lam in (0.01, 0.1, 1.0, 10.0):
            s, _ = M.fit_bt(W, G, lam, iters=40)
            assert s.mean() == pytest.approx(0.0, abs=1e-8)                                           # Ridge zieht zum Mittel 0
            stds.append(float(np.std(s)))
        assert all(a >= b - 1e-9 for a, b in zip(stds, stds[1:])), stds                               # Streuung fällt mit λ
        lam_eb, s_eb, _ = M.empirical_bayes(W, G)
        s_ml, _ = M.fit_bt(W, G, C.ML_LAM, iters=40)
        assert np.std(s_eb) <= np.std(s_ml) + 1e-9
        assert M.spread_ratio(s_eb, inst.strengths) <= M.spread_ratio(s_ml, inst.strengths) + 1e-9
        assert lam_eb >= C.ML_LAM


def test_for_a_huge_prior_precision_all_estimates_are_equal_and_the_covariance_is_the_prior():
    rng = np.random.default_rng(7)
    for _ in range(15):
        n = int(rng.integers(4, 11))
        inst = _inst(BAL, int(rng.choice((66, 200))), int(rng.integers(0, 10 ** 6)), n)
        lam = 1e8
        s, Sigma = M.fit_bt(inst.W, inst.G, lam, iters=20)
        assert np.abs(s).max() < 1e-4 and np.ptp(s) < 1e-4
        assert Sigma == pytest.approx(np.eye(n) / lam, rel=1e-3)
    s0, S0 = M.fit_bt(np.zeros((5, 5)), np.zeros((5, 5)), 0.3)                                          # Randfall: ohne Daten ist die Schätzung der Prior
    assert (s0 == 0).all() and S0 == pytest.approx(np.eye(5) / 0.3)
    lam, s1, _ = M.empirical_bayes(np.zeros((5, 5)), np.zeros((5, 5)))
    assert (s1 == 0).all() and lam == C.EB_GRID[0]


# --- 5. Gleiche Eingabe, gleiche Ausgabe --------------------------------------------------------------------------------------------------------

def test_same_seed_gives_the_same_instance_the_same_fit_and_the_same_intervals():
    for seed in (0, 7, 12345):
        for design in (BAL, UNEQ):
            a, b = _inst(design, 100, seed), _inst(design, 100, seed)
            assert np.array_equal(a.W, b.W) and np.array_equal(a.a, b.a) and np.array_equal(a.strengths, b.strengths)
            ma, mb = ev.measure(a, 0.1, seed, boot=False), ev.measure(b, 0.1, seed, boot=False)
            for name in ma["iv"]:
                assert np.array_equal(ma["iv"][name]["lo"], mb["iv"][name]["lo"]) and np.array_equal(ma["iv"][name]["hi"], mb["iv"][name]["hi"])
            assert np.array_equal(ma["pbest"], mb["pbest"])
    assert not np.array_equal(_inst(BAL, 100, 1).W, _inst(BAL, 100, 2).W)
