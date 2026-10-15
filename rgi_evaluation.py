"""Analyse des gezeigten Laufs, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Vier Schätzer (Siegquote, Bradley-Terry ML, Ridge, Empirical Bayes) und drei Rangintervalle (Posterior aus der Laplace-Näherung des Empirical Bayes, Bootstrap der Partien, Wald mit
ML) auf derselben Wahrheit und denselben Würfen. Maße: Verzerrung des geschätzten Besten (Gewinnerfluch), Spreizung, MSE der Stärken, Trefferquoten der Ordnung, Abdeckung und
Breite der Rangintervalle, Kalibrierung von P(Bester), Abbruchregel."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import rgi_constants as C
import rgi_intervals as IV
import rgi_model as M
import rgi_scenario as SC
import turnier_generator as T
from rgi_scenario import Settings

NAN = float("nan")
_SALT_DRAW, _SALT_BOOT = 0xD2A7, 0xB007
POSTERIOR, BOOTSTRAP, WALD = C.INTERVALS[0], C.INTERVALS[1], C.INTERVALS[2]
SIEGQUOTE, ML, RIDGE, EB = C.ESTIMATORS


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def signed(x: float, digits: int = 2) -> str:
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


# --- Eine Instanz ------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fit:
    values: dict            # Schätzer -> Werte je Spieler (Siegquote: Quote, sonst θ)
    lam_eb: float
    Sigma_eb: np.ndarray
    Sigma_ml: np.ndarray


def fit_all(inst: SC.Instance, ridge_lam: float) -> Fit:
    W, G = inst.W, inst.G
    s_ml, S_ml = M.fit_bt(W, G, C.ML_LAM, iters=30)
    s_r, _ = M.fit_bt(W, G, ridge_lam, iters=30)
    lam, s_eb, S_eb = M.empirical_bayes(W, G)
    return Fit({SIEGQUOTE: M.win_quote(W, G), ML: s_ml, RIDGE: s_r, EB: s_eb}, float(lam), S_eb, S_ml)


def rank_draws(inst: SC.Instance, fit: Fit, seed: int, boot: bool, draws: int = C.DRAWS, boots: int = C.BOOT) -> dict:
    n = len(inst.strengths)
    out = {POSTERIOR: IV.posterior_ranks(fit.values[EB], fit.Sigma_eb, T.SplitMix64(int(seed) ^ _SALT_DRAW), draws),
           WALD: IV.posterior_ranks(fit.values[ML], fit.Sigma_ml, T.SplitMix64((int(seed) ^ _SALT_DRAW) + 1), draws)}
    if boot:
        out[BOOTSTRAP] = IV.bootstrap_ranks(inst.a, inst.b, inst.a_wins, n, fit.lam_eb, fit.values[EB], T.SplitMix64(int(seed) ^ _SALT_BOOT), boots)
    return out


def measure(inst: SC.Instance, ridge_lam: float, seed: int, boot: bool, level: float = 0.9) -> dict:
    s = inst.strengths
    tr = M.ranks_from(s)
    fit = fit_all(inst, ridge_lam)
    est = {}
    for name, v in fit.values.items():
        est[name] = {"acc": M.pair_accuracy(v, s), "winner": M.winner_correct(v, s), "top3": M.top_k_overlap(v, s, 3), "bias": M.top_bias(v, s),
                     "spread": NAN if name == SIEGQUOTE else M.spread_ratio(v, s), "mse": NAN if name == SIEGQUOTE else M.mse(v, s)}
    ranks = rank_draws(inst, fit, seed, boot)
    iv = {}
    for name, rk in ranks.items():
        lo, hi = IV.interval(rk, level)
        iv[name] = {"lo": lo, "hi": hi, "cover": IV.covered(lo, hi, tr), "width": (hi - lo + 1).astype(float)}
    return {"fit": fit, "true_rank": tr, "est": est, "iv": iv, "pbest": IV.p_best(ranks[POSTERIOR]), "ranks": ranks}


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    instance: SC.Instance
    fit: Fit
    true_rank: np.ndarray
    est_rank: np.ndarray        # Rang nach dem gewählten Schätzer
    lo: np.ndarray              # Rangintervall nach dem gewählten Verfahren und Niveau
    hi: np.ndarray
    p_best: np.ndarray
    est: dict                   # Schätzer -> Kennzahlen (acc, winner, top3, bias, spread, mse)
    width: float                # mittlere Intervallbreite
    cover: float                # Anteil der Spieler, deren wahrer Rang im Intervall liegt


def analyse(s: Settings) -> Analysis:
    inst = SC.simulate(s.n, s.delta, s.design, s.games, s.act, s.truth, s.coupling, s.seed)
    m = measure(inst, s.lam, s.seed, boot=(s.interval == BOOTSTRAP), level=s.level / 100.0)
    lo, hi = IV.interval(m["ranks"][s.interval], s.level / 100.0)
    tr = m["true_rank"]
    return Analysis(s, inst, m["fit"], tr, M.ranks_from(m["fit"].values[s.estimator]), lo, hi, m["pbest"], m["est"], float((hi - lo + 1).mean()), float(IV.covered(lo, hi, tr).mean()))


def verdict_analysis(a: Analysis):
    s = a.settings
    tr = a.true_rank
    i3 = int(np.argmax(tr == min(3, s.n)))
    text = (f"Der wahre Drittbeste liegt mit {s.level} % Sicherheit zwischen Rang **{int(a.lo[i3])}** und **{int(a.hi[i3])}** ({s.interval}); mittlere Breite der Intervalle {de(a.width, 1)} von "
            f"{s.n} Rängen, der wahre Rang liegt bei {pct(a.cover)} % der Spieler im Intervall. ")
    b, e = a.est[ML]["bias"], a.est[EB]["bias"]
    text += (f"Der geschätzte Beste liegt bei der Bradley-Terry-Schätzung {signed(b)} über seiner wahren Stärke, mit Empirical Bayes {signed(e)}. "
             f"Paartreffer: Siegquote {de(a.est[SIEGQUOTE]['acc'], 3)}, Bradley-Terry {de(a.est[ML]['acc'], 3)}, Empirical Bayes {de(a.est[EB]['acc'], 3)}.")
    if s.design == C.DESIGNS[0] and abs(a.est[EB]["acc"] - a.est[SIEGQUOTE]["acc"]) < 0.02:
        text += " Im ausgewogenen Plan ändert die Schrumpfung die Ordnung kaum."
    return "info", text


# --- Feste Läufe der Experimente --------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _measure(n, delta, design, games, act, truth, coupling, seed, boot):
    inst = SC.simulate(n, delta, design, games, act, truth, coupling, seed)
    m = measure(inst, 0.1, seed, boot)
    return {"est": m["est"], "iv": {k: {"cover": v["cover"], "width": v["width"], "lo": v["lo"], "hi": v["hi"]} for k, v in m["iv"].items()}, "true_rank": m["true_rank"],
            "pbest": m["pbest"], "lam": m["fit"].lam_eb}


def cell(design: str, games: int, truth: str = C.TRUTHS[0], coupling: str = C.COUPLINGS[0], boot: bool = False, delta: float = C.EXP_DELTA) -> dict:
    """Mittel über die festen Läufe für ein Design mit `games` Partien: Verzerrung, Spreizung, MSE, Trefferquoten, Abdeckung und Breite je Verfahren, nach wahrem Rang, P(Bester)."""
    res = [_measure(C.EXP_N, delta, design, games, C.EXP_ACT, truth, coupling, sd, boot) for sd in C.SWEEP_SEEDS]
    out = {"games": games, "lam_median": float(np.median([r["lam"] for r in res])), "est": {}, "iv": {}}
    for name in C.ESTIMATORS:
        out["est"][name] = {k: _mean([r["est"][name][k] for r in res]) for k in ("acc", "winner", "top3", "bias", "spread", "mse")}
    methods = [POSTERIOR, WALD] + ([BOOTSTRAP] if boot else [])
    for m in methods:
        groups = []
        for lo_r, hi_r in C.RANK_GROUPS:
            groups.append(_mean([float(r["iv"][m]["cover"][(r["true_rank"] >= lo_r) & (r["true_rank"] <= hi_r)].mean()) for r in res]))
        out["iv"][m] = {"cover": _mean([float(r["iv"][m]["cover"].mean()) for r in res]), "width": _mean([float(r["iv"][m]["width"].mean()) for r in res]), "by_rank": groups}
    out["third"] = (_mean([float(r["iv"][POSTERIOR]["lo"][r["true_rank"] == 3][0]) for r in res]), _mean([float(r["iv"][POSTERIOR]["hi"][r["true_rank"] == 3][0]) for r in res]))
    out["pbest"] = (np.concatenate([r["pbest"] for r in res]), np.concatenate([(r["true_rank"] == 1).astype(float) for r in res]))
    return out


BOOT_CELLS = {C.DESIGNS[0]: (66, 264), C.DESIGNS[1]: (66, 132, 264)}


def sweep_main() -> dict:
    """Experimente 1 bis 4: je Design die vier Partienzahlen EXP_GAMES (gleichabständige Stärken); Bootstrap an den Partienzahlen BOOT_CELLS."""
    return {d: {g: cell(d, g, boot=g in BOOT_CELLS[d]) for g in C.EXP_GAMES} for d in C.DESIGNS}


def sweep_normal() -> dict:
    """Wahrheit aus der Normalverteilung (der Prior stimmt): 66 Partien, beide Designs, mit Bootstrap."""
    return {d: cell(d, 66, truth=C.TRUTHS[1], boot=True) for d in C.DESIGNS}


def sweep_coupling() -> dict:
    """Experiment 6: ungleiche Aktivität (132 Partien), unabhängig / die Besten spielen am meisten / am wenigsten."""
    return {c: cell(C.DESIGNS[1], 132, coupling=c, boot=True) for c in C.COUPLINGS}


@functools.lru_cache(maxsize=256)
def _stop_run(delta: float, seed: int) -> tuple:
    """Vollständige Runden (jedes Paar einmal); nach jeder Runde P(Bester) = größter Wert der Posterior-Ziehungen. Rückgabe: (Partien, P_max, Bester richtig) je Runde, bis die höchste
    Schwelle erreicht ist (die niedrigeren liegen früher)."""
    n = C.EXP_N
    pairs = n * (n - 1) // 2
    inst = SC.simulate(n, delta, C.DESIGNS[0], pairs * C.STOP_MAX_ROUNDS, C.EXP_ACT, C.TRUTHS[0], C.COUPLINGS[0], seed)
    rows = []
    rng = T.SplitMix64(int(seed) ^ _SALT_DRAW)
    for r in range(1, C.STOP_MAX_ROUNDS + 1):
        k = r * pairs
        win = np.where(inst.a_wins[:k], inst.a[:k], inst.b[:k])
        lose = np.where(inst.a_wins[:k], inst.b[:k], inst.a[:k])
        W = np.zeros((n, n))
        np.add.at(W, (win, lose), 1.0)
        _lam, s, Sigma = M.empirical_bayes(W, W + W.T)
        pb = IV.p_best(IV.posterior_ranks(s, Sigma, rng, 400))
        rows.append((k, float(pb.max()), int(np.argmax(pb)) == n - 1))
        if pb.max() >= max(C.STOP_THRESHOLDS):
            break
    return tuple(rows)


def sweep_stop(delta: float = C.EXP_DELTA) -> dict:
    """Abbruchregel je Schwelle: Median der Partien (Budget STOP_MAX_ROUNDS Runden, wenn nie erreicht), Anteil der durch die Schwelle beendeten Läufe, Trefferquote (der als Bester
    Gemeldete ist der wahre Beste) dieser Läufe und aller Läufe (bei Budgetende zählt der zuletzt Gemeldete)."""
    runs = [_stop_run(delta, sd) for sd in C.SWEEP_SEEDS]
    budget = C.STOP_MAX_ROUNDS * (C.EXP_N * (C.EXP_N - 1) // 2)
    out = {}
    for thr in C.STOP_THRESHOLDS:
        games, stopped, hit, hit_stopped = [], [], [], []
        for rows in runs:
            stop = next((r for r in rows if r[1] >= thr), None)
            final = stop if stop is not None else rows[-1]
            games.append(final[0] if stop is not None else budget)
            stopped.append(stop is not None)
            hit.append(bool(final[2]))
            if stop is not None:
                hit_stopped.append(bool(stop[2]))
        out[thr] = {"median_games": float(np.median(games)), "stopped": float(np.mean(stopped)), "hit_stopped": float(np.mean(hit_stopped)) if hit_stopped else NAN, "hit_all": float(np.mean(hit))}
    return out


def calibration(cells) -> list:
    """P(Bester) in Klassen: (untere Grenze, obere Grenze, Zahl, gesagt, tatsächlich) über die gegebenen Zellen gepoolt."""
    P = np.concatenate([c["pbest"][0] for c in cells])
    Tt = np.concatenate([c["pbest"][1] for c in cells])
    out = []
    for lo, hi in C.CALIB_BINS:
        sel = (P >= lo) & (P < hi)
        out.append((lo, min(hi, 1.0), int(sel.sum()), float(P[sel].mean()) if sel.any() else NAN, float(Tt[sel].mean()) if sel.any() else NAN))
    return out


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

BAL, UNEQ = C.DESIGNS


def verdict_curse(main):
    b = main[BAL][EXP_FIRST]["est"]
    u = main[UNEQ][EXP_FIRST]["est"]
    last = C.EXP_GAMES[-1]
    bl = main[BAL][last]["est"]
    return "warning", (f"Bei {EXP_FIRST} Partien (jedes Paar einmal) liegt der geschätzte Beste mit Bradley-Terry **{signed(b[ML]['bias'])}** über seiner wahren Stärke (Einheiten θ), mit "
                       f"Ridge {signed(b[RIDGE]['bias'])}, mit Empirical Bayes **{signed(b[EB]['bias'])}**; der Fehler der Stärken (MSE) fällt von {de(b[ML]['mse'])} auf {de(b[EB]['mse'])}. "
                       f"Bei ungleicher Aktivität ist es {signed(u[ML]['bias'])} gegen {signed(u[EB]['bias'])}. Mit {last} Partien bleibt bei Bradley-Terry {signed(bl[ML]['bias'])}: "
                       "Die Spitze einer Rangliste ist umso zu hoch geschätzt, je weniger Partien hinter ihr stehen.")


def verdict_order(main):
    b = main[BAL][EXP_FIRST]["est"]
    u = main[UNEQ][EXP_FIRST]["est"]
    return "info", (f"Im ausgewogenen Plan ordnen alle vier Schätzer fast gleich: Paartreffer Siegquote {de(b[SIEGQUOTE]['acc'], 3)}, Bradley-Terry {de(b[ML]['acc'], 3)}, "
                    f"Empirical Bayes {de(b[EB]['acc'], 3)}. Bei ungleicher Aktivität ({EXP_FIRST} Partien) {de(u[SIEGQUOTE]['acc'], 3)}, {de(u[ML]['acc'], 3)}, {de(u[EB]['acc'], 3)}: "
                    "Shrinkage korrigiert die Werte, die Rangfolge ändert sie wenig.")


def verdict_width(main):
    c = main[BAL][EXP_FIRST]
    d = main[BAL][264]
    return "info", (f"Bei {EXP_FIRST} Partien (jedes Paar einmal) reicht das 90-%-Intervall des wahren Drittbesten im Mittel von Rang **{de(c['third'][0], 1)}** bis **{de(c['third'][1], 1)}**, "
                    f"die mittlere Breite aller Intervalle ist {de(c['iv'][POSTERIOR]['width'], 1)} von {C.EXP_N} Rängen. Mit 264 Partien: {de(d['third'][0], 1)} bis {de(d['third'][1], 1)} "
                    f"(Breite {de(d['iv'][POSTERIOR]['width'], 1)}), mit {C.EXP_GAMES[-1]}: Breite {de(main[BAL][C.EXP_GAMES[-1]]['iv'][POSTERIOR]['width'], 1)}.")


def verdict_coverage(main, normal):
    u = main[UNEQ][EXP_FIRST]["iv"]
    b = main[BAL][EXP_FIRST]["iv"]
    n = normal[UNEQ]["iv"]
    return "warning", (f"Gewünscht 90 %. Gemessen bei {EXP_FIRST} Partien, ausgewogen: Posterior **{pct(b[POSTERIOR]['cover'])} %**, Wald {pct(b[WALD]['cover'])} %, "
                       f"Bootstrap der Partien {pct(b[BOOTSTRAP]['cover'])} %; bei ungleicher Aktivität Posterior {pct(u[POSTERIOR]['cover'])} %, Bootstrap **{pct(u[BOOTSTRAP]['cover'])} %**. "
                       f"Sind die Stärken wirklich normalverteilt (der Prior stimmt), ist es {pct(n[POSTERIOR]['cover'])} % gegen {pct(n[BOOTSTRAP]['cover'])} %. "
                       "Das Posterior-Intervall ist konservativ, der Bootstrap der Partien liegt bei wenigen Partien und ungleicher Aktivität darunter, am stärksten beim Besten.")


def verdict_calibration(cal, stop):
    top = cal[-1]
    s9 = stop[0.9]
    s8 = stop[0.8]
    return "info", (f"Wo der Posterior dem Spieler Platz 1 mit mindestens {de(top[0], 1)} zuspricht, sagt er im Mittel {de(top[3])} und trifft {de(top[4])} (n = {top[2]}). Abbruchregel: "
                    f"Halt bei P(Bester) ≥ 0,8 in {pct(s8['stopped'])} % der Läufe (Median {int(s8['median_games'])} Partien), Treffer {de(s8['hit_stopped'])}; bei ≥ 0,9 in "
                    f"{pct(s9['stopped'])} % der Läufe, Treffer {de(s9['hit_stopped'])}. Wer nie die Schwelle erreicht, hat das Budget von {C.STOP_MAX_ROUNDS * C.STOP_ROUND} Partien ausgeschöpft.")


def verdict_coupling(cp):
    n, hi, lo = (cp[c] for c in C.COUPLINGS)
    return "warning", (f"Spielen die Stärksten selten (132 Partien, ungleiche Aktivität), steigt die Überschätzung der Spitze bei Bradley-Terry auf {signed(lo['est'][ML]['bias'])} "
                       f"(unabhängig {signed(n['est'][ML]['bias'])}, die Besten spielen am meisten {signed(hi['est'][ML]['bias'])}), und Empirical Bayes kostet Treffer: Sieger richtig "
                       f"**{de(lo['est'][EB]['winner'])}** gegen {de(lo['est'][SIEGQUOTE]['winner'])} bei der Siegquote. Das Bootstrap-Intervall des wahren Besten deckt "
                       f"nur {pct(lo['iv'][BOOTSTRAP]['by_rank'][0])} % (Posterior {pct(lo['iv'][POSTERIOR]['by_rank'][0])} %).")


EXP_FIRST = C.EXP_GAMES[0]
