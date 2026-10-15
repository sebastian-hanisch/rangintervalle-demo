"""Normalwerte, Ränge, Quantilintervalle, Posterior- und Bootstrap-Ränge und die Simulation der Daten an Handwerten."""

import numpy as np
import pytest

import rgi_constants as C
import rgi_intervals as IV
import rgi_scenario as SC
import turnier_generator as T


def test_normals_have_mean_zero_and_unit_variance_and_are_deterministic():
    z = IV.normals(T.SplitMix64(5), 20001)
    assert len(z) == 20001 and abs(z.mean()) < 0.03 and z.std() == pytest.approx(1.0, abs=0.03)
    assert np.array_equal(z, IV.normals(T.SplitMix64(5), 20001)) and not np.array_equal(z, IV.normals(T.SplitMix64(6), 20001))
    assert abs(np.mean(z ** 3)) < 0.1 and np.mean(z ** 4) == pytest.approx(3.0, abs=0.2)           # Normal, nicht nur Mittel und Varianz


def test_normals_are_box_muller_of_the_stream_by_hand():
    import math
    for count in (2, 3):
        u = T.SplitMix64(11)
        half = (count + 1) // 2
        u1, u2 = u.uniform(half), u.uniform(half)
        r = np.sqrt(-2.0 * np.log(u1))
        expected = np.concatenate([r * np.cos(2 * math.pi * u2), r * np.sin(2 * math.pi * u2)])[:count]
        assert IV.normals(T.SplitMix64(11), count) == pytest.approx(expected)


def test_rank_matrix_by_hand():
    v = np.array([[0.1, 0.9, -1.0], [2.0, 1.0, 3.0]])
    assert IV.rank_matrix(v).tolist() == [[2, 1, 3], [2, 3, 1]]


def test_interval_is_the_inverse_distribution_function_by_hand():
    ranks = np.array([[1, 3], [1, 3], [2, 3], [2, 4], [2, 4], [3, 4], [3, 4], [3, 5], [4, 5], [5, 5]])     # 10 Ziehungen, 2 Spieler
    lo, hi = IV.interval(ranks, 0.8)
    assert lo.tolist() == [1, 3] and hi.tolist() == [4, 5]                  # 5- bzw. 95-%-Quantil: kleinster Wert, 9. Wert
    lo2, hi2 = IV.interval(ranks, 0.5)
    assert lo2.tolist() == [2, 3] and hi2.tolist() == [3, 5]                # 25-%-Quantil: 3. Wert, 75-%-Quantil: 8. Wert
    lo3, hi3 = IV.interval(ranks, 1.0)
    assert lo3.tolist() == [1, 3] and hi3.tolist() == [5, 5]


def test_interval_contains_at_least_the_level_of_the_mass():
    rng = np.random.default_rng(1)
    ranks = np.argsort(np.argsort(rng.normal(size=(1000, 7)), axis=1), axis=1) + 1
    for level in (0.5, 0.8, 0.9, 0.95):
        lo, hi = IV.interval(ranks, level)
        inside = ((ranks >= lo) & (ranks <= hi)).mean(axis=0)
        assert (inside >= level - 1e-9).all()


def test_p_best_and_covered_by_hand():
    ranks = np.array([[1, 2], [1, 2], [2, 1], [1, 2]])
    assert IV.p_best(ranks).tolist() == [0.75, 0.25]
    assert IV.covered(np.array([1, 2]), np.array([3, 2]), np.array([3, 3])).tolist() == [1.0, 0.0]
    assert IV.covered(np.array([3, 1, 4]), np.array([3, 3, 5]), np.array([3, 1, 3])).tolist() == [1.0, 1.0, 0.0]       # Grenzen zählen mit


def test_posterior_ranks_have_the_right_shape_and_are_permutations():
    s = np.array([1.0, 0.0, -1.0])
    rk = IV.posterior_ranks(s, np.eye(3), T.SplitMix64(1), 500)
    assert rk.shape == (500, 3) and all(sorted(r) == [1, 2, 3] for r in rk)
    assert (rk[:, 0] < rk[:, 2]).mean() > 0.8                                # Spieler 0 meistens vor Spieler 2


def test_bootstrap_ranks_equal_an_independent_resampling():
    import rgi_model as M
    a = np.array([0, 0, 1, 1, 0, 2, 2]); b = np.array([1, 2, 2, 0, 1, 1, 0]); w = np.array([True, True, True, False, True, False, True])
    got = IV.bootstrap_ranks(a, b, w, 3, 0.5, np.zeros(3), T.SplitMix64(4), 25)
    games = len(a)
    u = T.SplitMix64(4).uniform(25 * games)
    for t in range(25):
        k = np.minimum((u[t * games:(t + 1) * games] * games).astype(int), games - 1)
        W = np.zeros((3, 3))
        for g in k:
            if w[g]:
                W[a[g], b[g]] += 1
            else:
                W[b[g], a[g]] += 1
        s, _ = M.fit_bt(W, W + W.T, 0.5, np.zeros(3), 6)
        assert list(got[t]) == list(np.argsort(np.argsort(-s)) + 1), t


def test_bootstrap_ranks_shape_and_determinism():
    a = np.array([0, 0, 1, 1, 0, 2]); b = np.array([1, 2, 2, 0, 1, 1]); w = np.array([True, True, True, False, True, False])
    r1 = IV.bootstrap_ranks(a, b, w, 3, 0.5, np.zeros(3), T.SplitMix64(2), 40)
    r2 = IV.bootstrap_ranks(a, b, w, 3, 0.5, np.zeros(3), T.SplitMix64(2), 40)
    assert r1.shape == (40, 3) and np.array_equal(r1, r2) and all(sorted(r) == [1, 2, 3] for r in r1)


# --- Simulation -------------------------------------------------------------------------------------------------------------------------------

def _sim(design=C.DESIGNS[0], games=66, act=10, truth=C.TRUTHS[0], coupling=C.COUPLINGS[0], seed=1, n=12, delta=0.3):
    return SC.simulate(n, delta, design, games, act, truth, coupling, seed)


def test_balanced_design_plays_every_pair_k_times():
    inst = _sim(games=132)
    assert inst.games == 132 and (inst.G[np.triu_indices(12, 1)] == 2).all() and (np.diag(inst.G) == 0).all()
    assert np.array_equal(inst.G, inst.W + inst.W.T)
    assert _sim(games=10).games == 66                                             # mindestens ein Durchgang


def test_unequal_design_plays_the_requested_games_and_is_deterministic():
    a, b = _sim(C.DESIGNS[1], 200, seed=4), _sim(C.DESIGNS[1], 200, seed=4)
    assert a.games == 200 and a.W.sum() == 200 and np.array_equal(a.a, b.a) and np.array_equal(a.a_wins, b.a_wins)
    assert not np.array_equal(a.a, _sim(C.DESIGNS[1], 200, seed=5).a)


def test_coupling_controls_who_plays_more():
    most = _sim(C.DESIGNS[1], 3000, coupling=C.COUPLINGS[1])
    least = _sim(C.DESIGNS[1], 3000, coupling=C.COUPLINGS[2])
    g_most, g_least = most.G.sum(axis=1), least.G.sum(axis=1)
    assert g_most[-1] > 3 * g_most[0]                                             # Index n − 1 ist der Beste
    assert g_least[0] > 3 * g_least[-1]
    indep = _sim(C.DESIGNS[1], 3000, coupling=C.COUPLINGS[0]).G.sum(axis=1)
    assert indep.max() > 3 * indep.min()


def test_truths_equal_spacing_and_normal_have_the_same_scale():
    s = _sim().strengths
    assert s == pytest.approx(0.3 * (np.arange(12) - 5.5))
    normal = np.concatenate([SC.true_strengths(12, 0.3, C.TRUTHS[1], sd) for sd in range(200)])
    assert normal.std() == pytest.approx(0.3 * np.sqrt(143 / 12.0), rel=0.12)
    one = SC.true_strengths(12, 0.3, C.TRUTHS[1], 7)
    assert (np.diff(one) >= 0).all()                                              # aufsteigend: der Beste hat den letzten Index


def test_unequal_design_reaches_every_pair_and_the_normal_truth_is_the_sorted_scaled_stream():
    inst = _sim(C.DESIGNS[1], 3000, act=1)
    assert (inst.G[np.triu_indices(12, 1)] > 0).all()                              # auch das letzte Paar kommt vor
    import math
    z = IV.normals(T.SplitMix64(7 ^ SC._SALT_TRUTH), 12)
    assert SC.true_strengths(12, 0.3, C.TRUTHS[1], 7) == pytest.approx(np.sort(0.3 * math.sqrt(143 / 12.0) * z))


def test_stronger_players_win_more_with_many_games():
    inst = _sim(games=66 * 30)
    wins = inst.W.sum(axis=1)
    assert np.corrcoef(wins, inst.strengths)[0, 1] > 0.95
