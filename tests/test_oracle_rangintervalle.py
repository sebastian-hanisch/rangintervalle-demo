"""Orakel: Schätzer gegen scipy, Laplace-Kovarianz gegen finite Differenzen, Evidenz gegen numerische Integration, Rangwahrscheinlichkeit gegen exakte Quadratur,
Eigenschaften von Shrinkage und Intervallen."""

import math

import numpy as np
import pytest

import rgi_constants as C
import rgi_evaluation as ev
import rgi_intervals as IV
import rgi_model as M
import rgi_scenario as SC
import turnier_generator as T


def _data(seed, n=5, games=60):
    rng = np.random.default_rng(seed)
    s = np.linspace(-1, 1, n)
    W = np.zeros((n, n))
    for _ in range(games):
        a, b = rng.choice(n, 2, replace=False)
        if rng.random() < 1 / (1 + math.exp(-(s[a] - s[b]))):
            W[a, b] += 1
        else:
            W[b, a] += 1
    return W, W + W.T


def _neg(W, lam):
    n = len(W)
    i, j = np.triu_indices(n, 1)

    def f(s):
        d = s[i] - s[j]
        return float(np.sum(W[i, j] * np.logaddexp(0, -d) + W[j, i] * np.logaddexp(0, d)) + 0.5 * lam * s @ s)

    return f


def test_bt_equals_the_scipy_maximiser():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(5):
        W, G = _data(seed)
        lam = 0.2
        res = optimize.minimize(_neg(W, lam), np.zeros(len(W)), method="BFGS", options={"gtol": 1e-10})
        ours, _ = M.fit_bt(W, G, lam, iters=40)
        assert ours == pytest.approx(res.x, abs=1e-5), seed


def test_laplace_covariance_is_the_inverse_hessian_of_the_negative_log_posterior():
    W, G = _data(1)
    n, lam = len(W), 0.2
    s, Sigma = M.fit_bt(W, G, lam, iters=40)
    neg = _neg(W, lam)
    H = np.zeros((n, n))
    eps = 1e-4
    for a in range(n):
        for b in range(n):
            ea, eb = np.eye(n)[a] * eps, np.eye(n)[b] * eps
            H[a, b] = (neg(s + ea + eb) - neg(s + ea - eb) - neg(s - ea + eb) + neg(s - ea - eb)) / (4 * eps * eps)
    assert Sigma == pytest.approx(np.linalg.inv(H), abs=2e-3)


def test_evidence_matches_numerical_integration_for_two_players():
    # p(D | λ) = ∫ σ(d)^a σ(−d)^b N(d; 0, 2/λ) dd (die Summe s_0 + s_1 fällt heraus)
    for a, b, lam in ((24, 8, 1.0), (30, 30, 2.0), (12, 4, 0.5)):
        W = np.array([[0.0, a], [b, 0.0]])
        G = W + W.T
        ours = M.evidence(W, G, lam, iters=40)[0]
        d = np.linspace(-12, 12, 40001)
        var = 2.0 / lam
        logf = a * (-np.logaddexp(0, -d)) + b * (-np.logaddexp(0, d)) - 0.5 * d * d / var - 0.5 * math.log(2 * math.pi * var)
        m = logf.max()
        exact = m + math.log(np.trapezoid(np.exp(logf - m), d))
        assert ours == pytest.approx(exact, abs=0.08), (a, b, lam)


def test_posterior_rank_probability_equals_exact_quadrature():
    mu = np.array([1.0, 0.0, -0.5])
    rk = IV.posterior_ranks(mu, np.eye(3), T.SplitMix64(11), 40000)
    z = np.linspace(-8, 8, 8001)
    phi = lambda x: np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)   # noqa: E731
    Phi = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))   # noqa: E731
    for i in range(3):
        others = [j for j in range(3) if j != i]
        exact = np.trapezoid(phi(z - mu[i]) * Phi(z - mu[others[0]]) * Phi(z - mu[others[1]]), z)
        assert (rk[:, i] == 1).mean() == pytest.approx(exact, abs=0.01), i


def test_posterior_rank_probabilities_by_enumeration_of_orderings():
    mu = np.array([0.5, 0.0, -0.5])
    rk = IV.posterior_ranks(mu, 0.5 * np.eye(3), T.SplitMix64(3), 20000)
    perms = {tuple(r) for r in rk.tolist()}
    assert perms <= {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}
    counts = {p: sum(1 for r in rk.tolist() if tuple(r) == p) for p in perms}
    assert max(counts, key=counts.get) == (1, 2, 3) and sum(counts.values()) == 20000


def test_shrinkage_does_not_change_the_order_in_a_balanced_plan():
    for seed in range(15):
        inst = SC.simulate(10, 0.3, C.DESIGNS[0], 45 * 2, 10, C.TRUTHS[0], C.COUPLINGS[0], seed)
        fit = ev.fit_all(inst, 0.1)
        orders = {k: tuple(M.ranks_from(fit.values[k])) for k in (ev.ML, ev.RIDGE, ev.EB)}
        quote = tuple(M.ranks_from(fit.values[ev.SIEGQUOTE]))
        wins = inst.W.sum(axis=1)
        if len(set(wins)) == len(wins):                       # ohne Gleichstände der Siegzahl
            assert orders[ev.ML] == orders[ev.RIDGE] == orders[ev.EB] == quote, seed


def test_shrinkage_pulls_the_spread_in_and_empirical_bayes_finds_the_prior():
    lams, spreads = [], []
    for seed in range(40):
        inst = SC.simulate(12, 0.3, C.DESIGNS[0], 66 * 2, 10, C.TRUTHS[1], C.COUPLINGS[0], seed)
        fit = ev.fit_all(inst, 0.1)
        lams.append(fit.lam_eb)
        spreads.append((np.std(fit.values[ev.ML]) / np.std(inst.strengths), np.std(fit.values[ev.EB]) / np.std(inst.strengths)))
    assert np.mean([s[0] for s in spreads]) > 1.2 > 1.0 > np.mean([s[1] for s in spreads]) - 0.05          # ML überspreizt, EB nicht
    true_lam = 1.0 / (0.3 ** 2 * 143 / 12.0)
    assert true_lam / 2.5 < float(np.median(lams)) < true_lam * 2.5


def test_intervals_get_narrower_with_more_games():
    widths = []
    for games in (66, 264, 1056):
        inst = SC.simulate(12, 0.3, C.DESIGNS[0], games, 10, C.TRUTHS[0], C.COUPLINGS[0], 2)
        m = ev.measure(inst, 0.1, 2, boot=False)
        widths.append(float(m["iv"][ev.POSTERIOR]["width"].mean()))
    assert widths[0] > widths[1] > widths[2] > 0


def test_bootstrap_sample_has_the_size_and_frequencies_of_the_data():
    a = np.array([0, 0, 0, 1]); b = np.array([1, 1, 2, 2]); w = np.array([True, True, False, True])
    n_games = len(a)
    idx = np.minimum((T.SplitMix64(9).uniform(4000 * n_games) * n_games).astype(int), n_games - 1).reshape(4000, n_games)
    assert idx.shape == (4000, n_games)
    freq = np.bincount(idx.ravel(), minlength=n_games) / idx.size
    assert freq == pytest.approx([0.25] * n_games, abs=0.02)
