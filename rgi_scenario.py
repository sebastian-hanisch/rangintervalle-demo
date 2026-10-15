"""Szenario: Reglerwerte (Settings) und die simulierte Datenlage (wahre Stärken, Partien).

Zufall nur über SplitMix64 (Generator der Linie), feste Anordnung der Würfe je Instanz:
  Strom „Wahrheit“ (Seed ^ _SALT_TRUTH): nur bei normal gezogenen Stärken, n Normalwerte;
  Strom „Daten“ (Seed ^ _SALT_DATA): n Werte für die Zuordnung der Aktivität, dann je Partie ein Wert für die Paarwahl (ungleiches Design), dann je Partie ein Wert für den Ausgang.
Wahrheit: n Spieler, aufsteigend sortierte Stärken (Spieler n − 1 ist der Beste); P(i schlägt j) = σ(s_i − s_j)."""

import math
from dataclasses import dataclass

import numpy as np

import rgi_constants as C
import rgi_intervals as IV
import rgi_model as M
import turnier_generator as T

_SALT_TRUTH = 0x7A11
_SALT_DATA = 0xDA7A


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    design: str = C.DEFAULT_DESIGN
    games: int = C.DEFAULT_GAMES
    act: int = C.DEFAULT_ACT
    truth: str = C.DEFAULT_TRUTH
    coupling: str = C.DEFAULT_COUPLING
    estimator: str = C.DEFAULT_ESTIMATOR
    lam: float = C.DEFAULT_LAM
    interval: str = C.DEFAULT_INTERVAL
    level: int = C.DEFAULT_LEVEL
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Instance:
    strengths: np.ndarray
    a: np.ndarray               # Spieler a der Partie g
    b: np.ndarray
    a_wins: np.ndarray
    W: np.ndarray
    G: np.ndarray

    @property
    def games(self) -> int:
        return len(self.a)


def true_strengths(n: int, delta: float, truth: str, seed: int) -> np.ndarray:
    """Gleichabständig (Generator) oder normal gezogen mit derselben Standardabweichung wie die gleichabständigen Stärken; aufsteigend."""
    if truth == C.TRUTHS[0]:
        return T.make_strengths(n, delta)
    sd = delta * math.sqrt((n * n - 1) / 12.0)
    z = IV.normals(T.SplitMix64(int(seed) ^ _SALT_TRUTH), n)
    return np.sort(sd * z)


def simulate(n: int, delta: float, design: str, games: int, act: int, truth: str, coupling: str, seed: int) -> Instance:
    s = true_strengths(n, delta, truth, seed)
    rng = T.SplitMix64(int(seed) ^ _SALT_DATA)
    iu = np.triu_indices(n, 1)
    pairs = len(iu[0])
    if design == C.DESIGNS[0]:
        k = max(1, int(round(games / pairs)))
        total = pairs * k
        u_perm = rng.uniform(n)                       # Anordnung bleibt gleich; im ausgewogenen Design ungenutzt
        rng.uniform(total)
        u_out = rng.uniform(total)
        a, b = np.tile(iu[0], k), np.tile(iu[1], k)
    else:
        total = int(games)
        u_perm = rng.uniform(n)
        u_pair = rng.uniform(total)
        u_out = rng.uniform(total)
        base = np.geomspace(1.0, float(max(act, 1)), n)
        if coupling == C.COUPLINGS[1]:
            weight = base                              # Index n − 1 (der Beste) spielt am meisten
        elif coupling == C.COUPLINGS[2]:
            weight = base[::-1].copy()
        else:
            weight = base[np.argsort(u_perm, kind="stable")]
        w = weight[iu[0]] * weight[iu[1]]
        cum = np.cumsum(w) / w.sum()
        idx = np.minimum(np.searchsorted(cum, u_pair, side="right"), pairs - 1)
        a, b = iu[0][idx], iu[1][idx]
    a_wins = u_out < M.sigmoid(s[a] - s[b])
    win = np.where(a_wins, a, b)
    lose = np.where(a_wins, b, a)
    W = np.zeros((n, n))
    np.add.at(W, (win, lose), 1.0)
    return Instance(s, a, b, a_wins, W, W + W.T)
