"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Es gibt keinen ausblendbaren Regler: alle Regler sind in jedem Szenario sichtbar."""

import random
from dataclasses import dataclass

import streamlit as st

import rgi_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "delta_slider": SettingSpec("d", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX),
    "design_select": SettingSpec("ds", str, C.DEFAULT_DESIGN, choices=C.DESIGNS),
    "games_slider": SettingSpec("g", int, C.DEFAULT_GAMES, C.GAMES_MIN, C.GAMES_MAX),
    "act_slider": SettingSpec("a", int, C.DEFAULT_ACT, C.ACT_MIN, C.ACT_MAX),
    "truth_select": SettingSpec("t", str, C.DEFAULT_TRUTH, choices=C.TRUTHS),
    "coupling_select": SettingSpec("cp", str, C.DEFAULT_COUPLING, choices=C.COUPLINGS),
    "estimator_select": SettingSpec("e", str, C.DEFAULT_ESTIMATOR, choices=C.ESTIMATORS),
    "lam_slider": SettingSpec("l", float, C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
    "interval_select": SettingSpec("iv", str, C.DEFAULT_INTERVAL, choices=C.INTERVALS),
    "level_slider": SettingSpec("lv", int, C.DEFAULT_LEVEL, C.LEVEL_MIN, C.LEVEL_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "delta": "delta_slider", "design": "design_select", "games": "games_slider", "act": "act_slider", "truth": "truth_select", "coupling": "coupling_select",
               "estimator": "estimator_select", "lam": "lam_slider", "interval": "interval_select", "level": "level_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
