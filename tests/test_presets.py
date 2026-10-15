"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen; Adressparameter, Szenario und PDF."""

import rgi_constants as C
import rgi_evaluation as ev
import rgi_pdf_export as X
import rgi_presets as P
from rgi_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_first_preset_equals_the_default_settings():
    p = C.PRESETS["Platz 3 bei 66 Partien"]
    assert Settings(**p) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(P.PRESET_KEYS) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key_and_have_unique_parameters():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_presets_show_their_mechanism():
    assert C.PRESETS["Der Gewinnerfluch"]["estimator"] == C.ESTIMATORS[1] and C.PRESETS["Der Gewinnerfluch"]["design"] == C.DESIGNS[1]
    assert C.PRESETS["Mehr Partien"]["games"] > C.DEFAULT_GAMES and C.PRESETS["Die Besten spielen selten"]["coupling"] == C.COUPLINGS[2]
    assert C.PRESETS["Platz 3 bei 66 Partien"]["design"] == C.DESIGNS[0] and C.PRESETS["Platz 3 bei 66 Partien"]["games"] == 66


def test_sweep_instances_are_fixed_and_distinct_and_the_grids_are_sound():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60
    assert set(C.COLORS) == set(C.ESTIMATORS) and C.EXP_GAMES[0] == C.EXP_N * (C.EXP_N - 1) // 2
    assert list(C.EB_GRID) == sorted(C.EB_GRID) and C.EB_GRID[0] < C.DEFAULT_LAM < C.EB_GRID[-1]
    assert all(0 < t < 1 for t in C.STOP_THRESHOLDS) and C.STOP_ROUND == C.EXP_N * (C.EXP_N - 1) // 2


def test_pdf_is_built_for_every_preset():
    for name, p in C.PRESETS.items():
        s = Settings(**{**p, "n": 8, "games": 60})
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("λ ≈ 0,5; σ → τ µ θ − **fett** „Zitat“")
    assert out == 'lambda ca. 0,5; sigma -> Tau mu theta - fett "Zitat"'
    assert out.encode("latin-1")
