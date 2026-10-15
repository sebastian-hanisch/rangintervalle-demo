"""Schätzer, Laplace-Kovarianz, Evidenz, Empirical Bayes und die Maße an Handwerten."""

import math

import numpy as np
import pytest

import rgi_constants as C
import rgi_model as M
from conftest import counts


def test_sigmoid_is_stable():
    assert M.sigmoid(0.0) == 0.5 and M.sigmoid(-800.0) == 0.0 and M.sigmoid(800.0) == 1.0


def test_log_posterior_by_hand():
    W, G = counts(2, {(0, 1): 1})
    assert M.log_posterior(np.zeros(2), W, 0.5) == pytest.approx(-math.log(2.0))
    assert M.log_posterior(np.array([1.0, -1.0]), W, 0.5) == pytest.approx(-math.log1p(math.exp(-2.0)) - 0.25 * 2.0)


def test_two_players_one_game_with_ridge_one_by_hand():
    W, G = counts(2, {(0, 1): 1})
    s, Sigma = M.fit_bt(W, G, 1.0, iters=30)
    x = float(s[0])
    assert s[1] == pytest.approx(-x) and 1 - float(M.sigmoid(2 * x)) == pytest.approx(x, abs=1e-9) and x == pytest.approx(0.3365, abs=1e-3)
    p = float(M.sigmoid(2 * x))
    h = p * (1 - p)
    assert Sigma == pytest.approx(np.linalg.inv(np.array([[1 + h, -h], [-h, 1 + h]])), abs=1e-9)


def test_gradient_is_zero_at_the_solution():
    W, G = counts(4, {(0, 1): 6, (1, 0): 2, (1, 2): 5, (2, 1): 3, (2, 3): 4, (3, 2): 4, (0, 3): 7, (3, 0): 1})
    s, _ = M.fit_bt(W, G, lam=0.2, iters=40)
    P = M.sigmoid(s[:, None] - s[None, :])
    grad = (W - G * P).sum(axis=1) - 0.2 * s
    assert np.abs(grad).max() < 1e-8


def test_newton_does_not_diverge_for_separable_data_and_a_small_prior():
    W, G = counts(4, {(0, 1): 6, (0, 2): 6, (0, 3): 6, (1, 2): 4, (1, 3): 6, (2, 3): 6, (2, 1): 2})
    s = np.zeros(4)
    for _ in range(25):
        s, Sigma = M.fit_bt(W, G, lam=0.02, s0=s, iters=1)
    assert np.abs(s).max() < 12.0 and s[0] > s[1] > s[2] > s[3]
    assert M.log_posterior(s, W, 0.02) >= M.log_posterior(np.zeros(4), W, 0.02) and np.isfinite(Sigma).all()


def test_without_games_the_estimate_is_zero_and_the_covariance_is_the_prior():
    W, G = counts(4, {})
    s, Sigma = M.fit_bt(W, G, lam=0.5)
    assert np.allclose(s, 0.0) and np.allclose(Sigma, np.eye(4) / 0.5)


def test_more_games_shrink_the_covariance():
    W1, G1 = counts(3, {(0, 1): 2, (1, 0): 1, (1, 2): 2, (2, 1): 1})
    W2, G2 = counts(3, {(0, 1): 20, (1, 0): 10, (1, 2): 20, (2, 1): 10})
    _, S1 = M.fit_bt(W1, G1, 0.1, iters=20)
    _, S2 = M.fit_bt(W2, G2, 0.1, iters=20)
    assert np.diag(S2).sum() < np.diag(S1).sum() and np.allclose(S1, S1.T)


def test_evidence_prefers_the_prior_that_matches_the_spread_of_the_data():
    # Starke Streuung der Spieler (Siege 9:1 entlang einer Kette): kleines λ (breiter Prior) hat mehr Evidenz als ein großes
    W, G = counts(4, {(0, 1): 9, (1, 0): 1, (1, 2): 9, (2, 1): 1, (2, 3): 9, (3, 2): 1, (0, 2): 9, (2, 0): 1, (1, 3): 9, (3, 1): 1, (0, 3): 10})
    e_small = M.evidence(W, G, 0.1)[0]
    e_big = M.evidence(W, G, 20.0)[0]
    assert e_small > e_big
    # Ausgeglichene Daten (5:5): großes λ (enger Prior) gewinnt
    W2, G2 = counts(4, {(i, j): 5 for i in range(4) for j in range(4) if i != j})
    assert M.evidence(W2, G2, 20.0)[0] > M.evidence(W2, G2, 0.1)[0]


def test_empirical_bayes_picks_a_grid_value_and_matches_the_best_evidence():
    W, G = counts(4, {(0, 1): 7, (1, 0): 3, (1, 2): 6, (2, 1): 4, (2, 3): 6, (3, 2): 4, (0, 2): 8, (2, 0): 2})
    lam, s, Sigma = M.empirical_bayes(W, G)
    assert lam in C.EB_GRID
    evs = {g: M.evidence(W, G, g, iters=30)[0] for g in C.EB_GRID}
    assert evs[lam] == pytest.approx(max(evs.values()), abs=1e-6)
    assert s == pytest.approx(M.fit_bt(W, G, lam, iters=30)[0], abs=1e-6) and Sigma.shape == (4, 4)


def test_win_quote_and_ranks():
    W, G = counts(3, {(0, 1): 3, (1, 0): 1, (0, 2): 2})
    assert M.win_quote(W, G) == pytest.approx([5 / 6, 1 / 4, 0.0])
    W0, G0 = counts(3, {(0, 1): 1})
    assert M.win_quote(W0, G0).tolist() == [1.0, 0.0, 0.5]                      # eine einzige Partie zählt, ohne Partie 0,5
    assert list(M.ranks_from([0.1, 0.9, -1.0])) == [2, 1, 3]


def test_pair_accuracy_winner_and_top_k_by_hand():
    truth = [0.0, 1.0, 2.0, 3.0]
    assert M.pair_accuracy([0, 1, 2, 3], truth) == 1.0 and M.pair_accuracy([3, 2, 1, 0], truth) == 0.0 and M.pair_accuracy([5, 5, 5, 5], truth) == 0.5
    assert M.pair_accuracy([0, 0, 1, 1], truth) == pytest.approx((4 + 2 * 0.5) / 6)
    assert M.winner_correct([0, 1, 2, 3], truth) == 1.0 and M.winner_correct([3, 1, 2, 0], truth) == 0.0 and M.winner_correct([0, 1, 3, 3], truth) == 0.5
    assert M.winner_correct([1, 1, 1, 1], truth) == 0.25
    assert M.top_k_overlap([0, 1, 2, 3], truth, 2) == 1.0 and M.top_k_overlap([3, 2, 0, 1], truth, 2) == 0.0 and M.top_k_overlap([0, 3, 2, 1], truth, 2) == 0.5
    assert M.top_k_overlap([0, 1, 2], [0, 1, 2], 10) == 1.0


def test_top_bias_spread_and_mse_by_hand():
    truth = np.array([-1.0, 0.0, 1.0])
    est = np.array([-2.0, 0.0, 2.0])
    assert M.top_bias(est, truth) == pytest.approx(1.0)                      # (2 − 0) − (1 − 0)
    assert M.spread_ratio(est, truth) == pytest.approx(2.0)
    assert M.mse(est, truth) == pytest.approx((1 + 0 + 1) / 3)
    shifted = est + 5.0
    assert M.top_bias(shifted, truth) == pytest.approx(1.0) and M.mse(shifted, truth) == pytest.approx(2 / 3)     # um das Mittel zentriert
    t2 = truth + 7.0                                                              # wahre Stärken mit Mittel 7: nur die Abstände zählen
    assert M.top_bias(est, t2) == pytest.approx(1.0) and M.mse(est, t2) == pytest.approx(2 / 3)
    e3 = np.array([1.0, 2.0, 6.0])                                                # Schätzung mit Mittel 3: (6 − 3) − (1 − 0)
    assert M.top_bias(e3, truth) == pytest.approx(2.0) and M.mse(e3, truth) == pytest.approx(((-2 + 1) ** 2 + (-1 - 0) ** 2 + (3 - 1) ** 2) / 3)
    # der geschätzte Beste ist nicht der wahre Beste: Spieler 0 hat die größte Schätzung
    assert M.top_bias(np.array([3.0, 0.0, -3.0]), truth) == pytest.approx(3.0 - (-1.0))


def test_empirical_bayes_without_data_takes_the_first_grid_value():
    W, G = counts(4, {})
    lam, s, Sigma = M.empirical_bayes(W, G)
    assert lam == C.EB_GRID[0] and np.allclose(s, 0.0)
    assert M.evidence(W, G, 0.3)[0] == pytest.approx(0.0, abs=1e-9) and M.evidence(W, G, 5.0)[0] == pytest.approx(0.0, abs=1e-9)    # ohne Daten ist jede Evidenz gleich
