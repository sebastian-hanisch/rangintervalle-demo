"""Rangverteilungen und Rangintervalle: Ziehungen aus der Laplace-Näherung (Posterior, Wald) und Bootstrap der Partien. Zufall nur über SplitMix64 (Generator der Linie).

Rang 1 = der Beste. Das Intervall zum Niveau L ist je Spieler das Quantilintervall der ganzzahligen Rangverteilung (Umkehrfunktion der Verteilungsfunktion): untere Grenze der kleinste
Rang r mit P(Rang ≤ r) ≥ (1 − L)/2, obere der kleinste Rang mit P(Rang ≤ r) ≥ 1 − (1 − L)/2. Es enthält daher mindestens den Anteil L der Rangverteilung."""

import math

import numpy as np

import rgi_model as M
import turnier_generator as T


def normals(rng: T.SplitMix64, count: int) -> np.ndarray:
    """Standardnormalverteilte Werte (Box-Muller) aus dem SplitMix64-Strom."""
    half = (count + 1) // 2
    u1 = np.maximum(rng.uniform(half), 1e-300)
    u2 = rng.uniform(half)
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2.0 * math.pi * u2), r * np.sin(2.0 * math.pi * u2)])
    return z[:count]


def rank_matrix(values: np.ndarray) -> np.ndarray:
    """Zeilenweise Ränge (1 = größter Wert) einer (Ziehungen × Spieler)-Matrix."""
    return np.argsort(np.argsort(-values, axis=1, kind="stable"), axis=1, kind="stable") + 1


def posterior_ranks(s_hat: np.ndarray, Sigma: np.ndarray, rng: T.SplitMix64, draws: int) -> np.ndarray:
    """Ränge von `draws` Ziehungen aus N(ŝ, Σ)."""
    n = len(s_hat)
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
    z = normals(rng, draws * n).reshape(draws, n)
    return rank_matrix(s_hat[None, :] + z @ L.T)


def bootstrap_ranks(a: np.ndarray, b: np.ndarray, a_wins: np.ndarray, n: int, lam: float, s0: np.ndarray, rng: T.SplitMix64, boots: int, iters: int = 6) -> np.ndarray:
    """Bootstrap der Partien: Partien mit Zurücklegen ziehen (Umfang wie die Daten), mit demselben λ neu anpassen, Ränge der Neuanpassungen."""
    games = len(a)
    idx = np.minimum((rng.uniform(boots * games) * games).astype(int), games - 1).reshape(boots, games)
    out = np.empty((boots, n))
    for t in range(boots):
        k = idx[t]
        win = np.where(a_wins[k], a[k], b[k])
        lose = np.where(a_wins[k], b[k], a[k])
        W = np.zeros((n, n))
        np.add.at(W, (win, lose), 1.0)
        s, _ = M.fit_bt(W, W + W.T, lam, s0, iters)
        out[t] = s
    return rank_matrix(out)


def interval(ranks: np.ndarray, level: float) -> tuple:
    """(untere, obere) Grenzen je Spieler aus den Rängen (Ziehungen × Spieler)."""
    d = ranks.shape[0]
    srt = np.sort(ranks, axis=0)
    a = (1.0 - level) / 2.0
    lo = srt[max(int(math.ceil(a * d - 1e-9)) - 1, 0)]
    hi = srt[min(int(math.ceil((1.0 - a) * d - 1e-9)) - 1, d - 1)]
    return lo.astype(int), hi.astype(int)


def p_best(ranks: np.ndarray) -> np.ndarray:
    """Anteil der Ziehungen, in denen der Spieler Rang 1 hat."""
    return (ranks == 1).mean(axis=0)


def covered(lo: np.ndarray, hi: np.ndarray, true_rank: np.ndarray) -> np.ndarray:
    return ((true_rank >= lo) & (true_rank <= hi)).astype(float)
