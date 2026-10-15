"""Schätzer und Maße: Bradley-Terry mit Ridge (gedämpftes Newton-Verfahren), Laplace-Kovarianz, Evidenz und Empirical Bayes, Trefferquoten der Ordnung. Nur numpy; Stärken in θ-Einheiten.

Daten: Siegmatrix W[i, j] (Siege von i gegen j) und Partienmatrix G = W + Wᵀ. MAP: maximiert Σ_{i<j}[W_ij ln σ(s_i − s_j) + W_ji ln σ(s_j − s_i)] − λ/2 |s|²; Hesse-Matrix des
negativen Zielwerts H = Σ G_ij p_ij (1 − p_ij)(e_i − e_j)(e_i − e_j)ᵀ + λ I, Kovarianz Σ = H⁻¹ (Laplace). Die Evidenz (Laplace-Näherung der Randverteilung der Daten) für das λ des
Empirical Bayes: ln p(D | λ) ≈ ln post(ŝ) + (n/2) ln λ − ½ ln det H (Konstanten weggelassen)."""

import numpy as np

import rgi_constants as C


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def log_posterior(s: np.ndarray, W: np.ndarray, lam: float) -> float:
    """Σ_{i≠j} W_ij ln σ(s_i − s_j) − λ/2 |s|² (bis auf eine Konstante)."""
    d = s[:, None] - s[None, :]
    return float(-(W * np.logaddexp(0.0, -d)).sum() - 0.5 * lam * s @ s)


def fit_bt(W: np.ndarray, G: np.ndarray, lam: float, s0=None, iters: int = 12) -> tuple:
    """Rückgabe: (s, Σ). Newton-Schritte mit Rückwärtsschritt (Halbieren der Schrittweite, bis der Zielwert nicht sinkt): das volle Newton-Verfahren schwingt bei wenigen Partien und
    kleinem λ (Spieler ohne Sieg oder ohne Niederlage). Vollständige n × n-Rechnung (n ≤ 16)."""
    n = len(W)
    s = np.zeros(n) if s0 is None else np.asarray(s0, dtype=float).copy()
    eye = np.eye(n)
    f = log_posterior(s, W, lam)
    for _ in range(max(iters, 1)):
        P = sigmoid(s[:, None] - s[None, :])
        np.fill_diagonal(P, 0.5)
        grad = (W - G * P).sum(axis=1) - lam * s
        h = G * P * (1.0 - P)
        np.fill_diagonal(h, 0.0)
        H = lam * eye + np.diag(h.sum(axis=1)) - h
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-6:
            if log_posterior(s + t * step, W, lam) >= f - 1e-12:
                break
            t *= 0.5
        s = s + t * step
        f = log_posterior(s, W, lam)
    return s, covariance(s, G, lam)


def covariance(s: np.ndarray, G: np.ndarray, lam: float) -> np.ndarray:
    n = len(s)
    P = sigmoid(s[:, None] - s[None, :])
    np.fill_diagonal(P, 0.5)
    h = G * P * (1.0 - P)
    np.fill_diagonal(h, 0.0)
    H = lam * np.eye(n) + np.diag(h.sum(axis=1)) - h
    return np.linalg.inv(H)


def evidence(W: np.ndarray, G: np.ndarray, lam: float, s0=None, iters: int = 12) -> tuple:
    """(ln Evidenz, ŝ, Σ) bei festem λ."""
    s, Sigma = fit_bt(W, G, lam, s0, iters)
    logdet_H = -np.linalg.slogdet(Sigma)[1]
    return log_posterior(s, W, lam) + 0.5 * len(s) * np.log(lam) - 0.5 * logdet_H, s, Sigma


def empirical_bayes(W: np.ndarray, G: np.ndarray, grid=C.EB_GRID) -> tuple:
    """λ mit der größten Evidenz auf dem Gitter: (λ̂, ŝ, Σ). Das Gitter wird aufsteigend mit warmem Start durchlaufen."""
    best = None
    s = None
    for lam in grid:
        ev, s, Sigma = evidence(W, G, lam, s, iters=8)
        if best is None or ev > best[0]:
            best = (ev, lam, s.copy(), Sigma)
    return best[1], best[2], best[3]


def win_quote(W: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Siege je Partie; ohne Partie 0,5."""
    g = G.sum(axis=1)
    return np.where(g > 0, W.sum(axis=1) / np.maximum(g, 1.0), 0.5)


def ranks_from(values: np.ndarray) -> np.ndarray:
    """Rang 1 = größter Wert; Gleichstände nach Index (nur numerisch relevant)."""
    return np.argsort(np.argsort(-np.asarray(values, dtype=float), kind="stable"), kind="stable") + 1


def pair_accuracy(est, truth) -> float:
    """Anteil richtig geordneter Spielerpaare; ein Gleichstand des Schätzers zählt halb (0,5 = Zufall)."""
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    iu = np.triu_indices(len(est), 1)
    se, st = np.sign(est[iu[0]] - est[iu[1]]), np.sign(truth[iu[0]] - truth[iu[1]])
    return float(np.where(se == 0, 0.5, (se == st).astype(float)).mean())


def winner_correct(est, truth) -> float:
    """1, wenn der Schätzer den wahren Besten allein an die Spitze setzt; bei Gleichstand an der Spitze 1/Zahl der Gleichstände, sonst 0."""
    est = np.asarray(est, dtype=float)
    best = int(np.argmax(truth))
    top = est.max()
    tied = int((est == top).sum())
    return (1.0 / tied) if est[best] == top else 0.0


def top_k_overlap(est, truth, k: int) -> float:
    """Anteil der wahren Top-k, die der Schätzer unter seinen Top-k hat."""
    k = min(k, len(est))
    return len(set(np.argsort(-np.asarray(est), kind="stable")[:k].tolist()) & set(np.argsort(-np.asarray(truth), kind="stable")[:k].tolist())) / k


def top_bias(est, truth) -> float:
    """Gewinnerfluch: (Schätzung des geschätzten Besten − Mittel der Schätzung) − (wahre Stärke dieses Spielers − Mittel der wahren Stärken)."""
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    top = int(np.argmax(est))
    return float((est[top] - est.mean()) - (truth[top] - truth.mean()))


def spread_ratio(est, truth) -> float:
    """Standardabweichung der Schätzung durch Standardabweichung der wahren Stärken (1 = richtig; über 1 heißt: überspreizt)."""
    return float(np.std(est) / np.std(truth))


def mse(est, truth) -> float:
    """Mittlerer quadratischer Fehler der um ihr Mittel zentrierten Stärken."""
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    return float(np.mean(((est - est.mean()) - (truth - truth.mean())) ** 2))
