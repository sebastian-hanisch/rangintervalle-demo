"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

import numpy as np

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ
BASE = 1500.0

N_MIN, N_MAX, DEFAULT_N = 6, 16, 12
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.01, 0.3
DESIGNS = ("Ausgewogen (jedes Paar gleich oft)", "Ungleich (zufällige Paare)")
DEFAULT_DESIGN = DESIGNS[0]
GAMES_MIN, GAMES_MAX, GAMES_STEP, DEFAULT_GAMES = 30, 800, 10, 66
ACT_MIN, ACT_MAX, DEFAULT_ACT = 1, 20, 10      # Aktivität der aktivsten gegen die seltenste Spielerin (ungleiches Design)
TRUTHS = ("Gleichabständig", "Normal gezogen")
DEFAULT_TRUTH = TRUTHS[0]
COUPLINGS = ("Unabhängig von der Stärke", "Die Besten spielen am meisten", "Die Besten spielen am wenigsten")
DEFAULT_COUPLING = COUPLINGS[0]
ESTIMATORS = ("Siegquote", "Bradley-Terry (ML)", "Ridge", "Empirical Bayes")
DEFAULT_ESTIMATOR = ESTIMATORS[3]
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 0.01, 2.0, 0.01, 0.1
INTERVALS = ("Posterior (Laplace)", "Bootstrap der Partien", "Wald (Bradley-Terry ML)")
DEFAULT_INTERVAL = INTERVALS[0]
LEVEL_MIN, LEVEL_MAX, LEVEL_STEP, DEFAULT_LEVEL = 50, 95, 5, 90
SEED_MAX, DEFAULT_SEED = 999999, 3

ML_LAM = 0.01                                              # „kaum Shrinkage“: Ridge nur, damit der Schätzer bei Lücken endlich bleibt
EB_GRID = tuple(float(x) for x in np.logspace(-2.0, 1.5, 10))   # Gitter für λ des Empirical Bayes
DRAWS = 1000                                               # Ziehungen aus dem Posterior
BOOT = 100                                                 # Wiederholungen des Bootstraps der Partien

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_DELTA, EXP_ACT = 12, 0.3, 10
EXP_GAMES = (66, 132, 264, 528)                            # 66 = jedes Paar einmal (n = 12)
STOP_THRESHOLDS = (0.8, 0.9, 0.95)
STOP_ROUND = 66                                            # Partien je Runde der Abbruchregel (jedes Paar einmal)
STOP_MAX_ROUNDS = 12
DELTA_NARROW = 0.12
CALIB_BINS = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.01))
RANK_GROUPS = ((1, 1), (2, 3), (4, 9), (10, 12))

COLORS = {"Siegquote": "#9aa0a6", "Bradley-Terry (ML)": "#c0392b", "Ridge": "#8e6bb0", "Empirical Bayes": "#2F6B65"}
COLOR_TRUTH = "#e67e22"
COLOR_BAND = "#2F6B65"

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "design": DEFAULT_DESIGN, "games": DEFAULT_GAMES, "act": DEFAULT_ACT, "truth": DEFAULT_TRUTH, "coupling": DEFAULT_COUPLING,
         "estimator": DEFAULT_ESTIMATOR, "lam": DEFAULT_LAM, "interval": DEFAULT_INTERVAL, "level": DEFAULT_LEVEL, "seed": DEFAULT_SEED}
PRESETS = {
    "Platz 3 bei 66 Partien": dict(_BASE),
    "Der Gewinnerfluch": {**_BASE, "design": DESIGNS[1], "estimator": ESTIMATORS[1]},
    "Mehr Partien": {**_BASE, "games": 528},
    "Die Besten spielen selten": {**_BASE, "design": DESIGNS[1], "games": 132, "coupling": COUPLINGS[2]},
}
PRESET_HELP = {
    "Platz 3 bei 66 Partien": "12 Spieler, jedes Paar einmal: Der wahre Drittbeste liegt mit 90 % Sicherheit irgendwo zwischen Rang 1 und etwa 7.",
    "Der Gewinnerfluch": "66 Partien bei ungleicher Aktivität, reine Bradley-Terry-Schätzung: Der geschätzte Beste liegt weit über seiner wahren Stärke. Wechseln Sie auf Empirical Bayes.",
    "Mehr Partien": "528 Partien (jedes Paar achtmal): Die Intervalle werden schmal, die Überschätzung der Spitze klein.",
    "Die Besten spielen selten": "Wer stark ist, spielt wenig: Shrinkage zieht ausgerechnet den wahren Besten nach unten, Sieger richtig sinkt.",
}
