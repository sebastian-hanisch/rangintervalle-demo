"""Rangintervalle und Shrinkage – wie sicher ist Platz 3? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dreizehntes Stück der Ranking-Linie: Eine Rangliste nennt einen Rang, aber wie sicher ist er? Sechs Experimente messen, wie breit ein Rangintervall ist, dass die Spitze der Liste
zu hoch geschätzt wird (Gewinnerfluch), was Shrinkage (Empirical Bayes) daran ändert und was nicht (die Rangfolge), ob die Intervalle ihre Abdeckung halten und was passiert,
wenn die Stärksten selten spielen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import rgi_constants as C
import rgi_evaluation as ev
from rgi_pdf_export import build_pdf
from rgi_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rgi_scenario import Settings
from rgi_visualization import (
    build_bias,
    build_calibration,
    build_coupling,
    build_coverage,
    build_order,
    build_pbest,
    build_rank_intervals,
    build_shrinkage,
    build_stop,
    build_width,
)

st.set_page_config(page_title="Rangintervalle und Shrinkage – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}
BAL, UNEQ = C.DESIGNS


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _main():
    return ev.sweep_main()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _normal():
    return ev.sweep_normal()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _coupling():
    return ev.sweep_coupling()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _stop(delta):
    return ev.sweep_stop(delta)


st.title("📏 Rangintervalle und Shrinkage: wie sicher ist Platz 3?")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n = st.slider("Spieler n", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler gerankt werden.")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider", format="%.2f",
                      help="Stärkedifferenz zweier benachbarter Spieler (logistische Einheiten). Klein heißt: kaum etwas zu trennen.")
    design = st.selectbox("Spielplan", C.DESIGNS, key="design_select",
                          help="Ausgewogen: jedes Paar gleich oft. Ungleich: zufällige Paare, einige Spieler spielen viel, andere selten (Aktivität siehe unten).")
    games = st.slider("Partien insgesamt", *bounds("games_slider"), step=C.GAMES_STEP, key="games_slider",
                      help="Im ausgewogenen Plan wird auf ganze Runden gerundet (jedes Paar k-mal, mindestens einmal).")
    act = st.slider("Aktivität: aktivster gegen seltensten Spieler", *bounds("act_slider"), key="act_slider",
                    help="Nur im ungleichen Spielplan: Verhältnis der Spielhäufigkeit; 1 = alle gleich.")
    coupling = st.selectbox("Aktivität und Stärke", C.COUPLINGS, key="coupling_select", help="Nur im ungleichen Spielplan: Hängt die Spielhäufigkeit an der Stärke?")
    truth = st.selectbox("Wahre Stärken", C.TRUTHS, key="truth_select",
                         help="Gleichabständig (wie in der Linie) oder aus einer Normalverteilung gezogen (dann passt die Vorannahme des Empirical Bayes genau).")
    estimator = st.selectbox("Schätzer", C.ESTIMATORS, key="estimator_select",
                             help="Für den geschätzten Rang und die Stärken im Detail. Die Kennzahlen zeigen alle vier.")
    lam = st.slider("Ridge λ (nur Schätzer Ridge)", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider", format="%.2f",
                    help="Prior N(0, 1/λ) auf jeder Stärke. Beim Empirical Bayes wird λ aus den Daten gewählt.")
    interval = st.selectbox("Rangintervall", C.INTERVALS, key="interval_select",
                            help="Posterior: Ziehungen aus der Laplace-Näherung (Empirical Bayes). Bootstrap: Partien mit Zurücklegen, neu angepasst. Wald: dasselbe mit Bradley-Terry ML.")
    level = st.slider("Sicherheit des Intervalls (%)", *bounds("level_slider"), step=C.LEVEL_STEP, key="level_slider")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "design_select": design, "games_slider": int(games), "act_slider": int(act), "truth_select": truth,
                   "coupling_select": coupling, "estimator_select": estimator, "lam_slider": round(float(lam), 2), "interval_select": interval, "level_slider": int(level),
                   "seed_input": int(seed)})

st.markdown(
    """
Eine Rangliste sagt „Platz 3“, aber nicht, wie sicher das ist. Mit wenigen Partien reicht der Platz eines Spielers über die Hälfte der Liste, und der **geschätzte Beste** liegt im
Mittel über seiner wahren Stärke: Wer aus vielen den Besten wählt, wählt auch das Glück. Diese Demo ist das **dreizehnte Stück der Ranking-Linie** und misst, **wie breit ein
Rangintervall ist**, **wie sehr die Spitze überschätzt** wird, was **Shrinkage** (Empirical Bayes) daran ändert, und was es nicht ändert: **die Rangfolge**.

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie sicher ist ein Rang?", expanded=True):
    st.markdown(
        """
1. **Schätzen:** Aus den Siegen und Niederlagen wird je Spieler eine Stärke geschätzt (Bradley-Terry) und eine **Unsicherheit** dazu (Laplace-Näherung): Ein Spieler mit wenigen
   Partien hat eine große.
2. **Ziehen:** Aus dieser Unsicherheit werden viele mögliche „wahre Stärken“ gezogen; jede Ziehung ergibt eine Rangliste. Das Intervall ist die Spanne, in der der Rang eines
   Spielers in 90 % der Ziehungen liegt.
3. **Shrinkage:** Die Stärken der Spieler streuen weniger als die Schätzungen (Zufall vergrößert jede Streuung). Empirical Bayes schätzt aus den Daten, wie stark die Stärken
   wirklich streuen, und zieht jede Schätzung entsprechend zur Mitte, die unsicheren am stärksten.
4. **Gegenprobe:** Der Bootstrap der Partien zieht Partien mit Zurücklegen und schätzt neu; die Abdeckung sagt, wie oft der wahre Rang im Intervall liegt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

settings = Settings(int(n), round(float(delta), 2), design, int(games), int(act), truth, coupling, estimator, round(float(lam), 2), interval, int(level), int(seed))
a = _analysis(settings)

st.markdown("---")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📏 Die Rangliste mit ihren Intervallen")
cols = st.columns(len(C.ESTIMATORS))
for col, name in zip(cols, C.ESTIMATORS):
    e = a.est[name]
    col.metric(name, ev.de(e["acc"], 3), delta=f"Sieger {ev.de(e['winner'], 0)} · Top-3 {ev.de(e['top3'])}", delta_color="off",
               help="Anteil richtig geordneter Spielerpaare (0,5 = Zufall); darunter: Sieger richtig (1 = der wahre Beste steht oben) und Top-3-Treffer.")
verdict(*ev.verdict_analysis(a))
st.caption(f"{a.instance.games} Partien gespielt; Verzerrung des geschätzten Besten (θ): Bradley-Terry {ev.signed(a.est[ev.ML]['bias'])}, Ridge {ev.signed(a.est[ev.RIDGE]['bias'])}, "
           f"Empirical Bayes {ev.signed(a.est[ev.EB]['bias'])}; λ des Empirical Bayes {ev.de(a.fit.lam_eb, 2)}.")

st.plotly_chart(build_rank_intervals(a), width="stretch", key="rank_chart")
st.caption(f"Geschätzter Rang ({settings.estimator}) mit {settings.level}-%-Intervall ({settings.interval}) für jeden Spieler und sein wahrer Rang (Raute); der Balken enthält den wahren Rang, "
           "wenn er die Raute einschließt.")
cl, cr = st.columns(2)
with cl:
    st.plotly_chart(build_shrinkage(a), width="stretch", key="shrink_chart")
    st.caption("Stärken in Elo-Punkten: Bradley-Terry spreizt die Spitze zu weit, Empirical Bayes zieht sie zur Mitte; die Raute ist die Wahrheit.")
with cr:
    st.plotly_chart(build_pbest(a), width="stretch", key="pbest_chart")
    st.caption("Wahrscheinlichkeit für Rang 1 je Spieler (Posterior). Orange: der wahre Beste.")
st.download_button("📄 PDF des aktuellen Laufs", data=build_pdf(a), file_name="rangintervalle.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

main = _main()
FIRST, LAST = C.EXP_GAMES[0], C.EXP_GAMES[-1]
st.subheader("🔬 Experiment 1: Die Spitze ist überschätzt")
st.caption("12 Spieler (feste Instanz, unabhängig vom Regler), Stärkeabstand 0,3, 60 feste Läufe; Verzerrung = geschätzte Stärke des geschätzten Besten minus seine wahre Stärke, "
           "beide um ihr Mittel zentriert (θ-Einheiten, 0,3 = ein Stärkeabstand). Aktivität bei ungleichem Plan 1 : 10, unabhängig von der Stärke.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Ausgewogen** (jedes Paar gleich oft)")
    st.plotly_chart(build_bias(main, BAL), width="stretch", key="exp1a_chart")
with c2:
    st.markdown("**Ungleich** (zufällige Paare)")
    st.plotly_chart(build_bias(main, UNEQ), width="stretch", key="exp1b_chart")
verdict(*ev.verdict_curse(main))
rows = []
for d in C.DESIGNS:
    for g in C.EXP_GAMES:
        e = main[d][g]["est"]
        rows.append({"Spielplan": d.split(" (")[0], "Partien": g, "Spreizung ML": round(e[ev.ML]["spread"], 2), "Spreizung EB": round(e[ev.EB]["spread"], 2),
                     "MSE ML": round(e[ev.ML]["mse"], 3), "MSE Ridge": round(e[ev.RIDGE]["mse"], 3), "MSE EB": round(e[ev.EB]["mse"], 3)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Ändert Shrinkage die Rangfolge?")
st.caption("Paartreffer (Anteil richtig geordneter Spielerpaare) der vier Schätzer, 60 feste Läufe.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Ausgewogen**")
    st.plotly_chart(build_order(main, BAL), width="stretch", key="exp2a_chart")
with c2:
    st.markdown("**Ungleich**")
    st.plotly_chart(build_order(main, UNEQ), width="stretch", key="exp2b_chart")
verdict(*ev.verdict_order(main))

st.subheader("🔬 Experiment 3: Wie breit ist ein Rangintervall?")
st.caption("Mittlere Breite der 90-%-Posterior-Intervalle über alle 12 Spieler (in Rängen) und das Intervall des wahren Drittbesten.")
st.plotly_chart(build_width(main), width="stretch", key="exp3_chart")
verdict(*ev.verdict_width(main))
st.dataframe(pd.DataFrame([{"Spielplan": d.split(" (")[0], "Partien": g, "Breite (Posterior)": round(main[d][g]["iv"][ev.POSTERIOR]["width"], 2),
                            "Drittbester: Rang von": round(main[d][g]["third"][0], 1), "bis": round(main[d][g]["third"][1], 1)} for d in C.DESIGNS for g in C.EXP_GAMES]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Halten die Intervalle ihre Abdeckung?")
normal = _normal()
st.caption("Anteil der Spieler (über 60 Läufe), deren wahrer Rang im 90-%-Intervall liegt. Normal: Die wahren Stärken sind aus einer Normalverteilung gezogen, der Prior stimmt.")
cells = {}
for d in C.DESIGNS:
    for g in C.EXP_GAMES:
        if ev.BOOTSTRAP in main[d][g]["iv"]:
            cells[f"{d.split(' (')[0]} {g}"] = main[d][g]
for d in C.DESIGNS:
    cells[f"{d.split(' (')[0]} {FIRST}, normal"] = normal[d]
st.plotly_chart(build_coverage(cells), width="stretch", key="exp4_chart")
verdict(*ev.verdict_coverage(main, normal))
rows = []
for label, c in cells.items():
    for m in (ev.POSTERIOR, ev.BOOTSTRAP):
        g = c["iv"][m]["by_rank"]
        rows.append({"Fall": label, "Verfahren": m, "Rang 1": round(g[0], 2), "Rang 2–3": round(g[1], 2), "Rang 4–9": round(g[2], 2), "Rang 10–12": round(g[3], 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: P(Bester) und die Abbruchregel")
st.caption("Kalibrierung: Die gesagte Wahrscheinlichkeit für Rang 1 (Posterior) gegen den tatsächlichen Anteil des wahren Besten, gepoolt über 60 Läufe und 12 Spieler (Fälle: 66 und 264 "
           "Partien, beide Spielpläne). Abbruchregel: Nach jeder vollständigen Runde (jedes Paar einmal) Halt, sobald P(Bester) die Schwelle erreicht.")
cal = ev.calibration([main[d][g] for d in C.DESIGNS for g in (66, 264)])
stop, stop_n = _stop(C.EXP_DELTA), _stop(C.DELTA_NARROW)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_calibration(cal), width="stretch", key="exp5a_chart")
with c2:
    st.plotly_chart(build_stop(stop, stop_n), width="stretch", key="exp5b_chart")
verdict(*ev.verdict_calibration(cal, stop))
st.dataframe(pd.DataFrame([{"Schwelle": thr, "Median der Partien (Abstand 0,3)": int(v["median_games"]), "Läufe mit Halt": f"{ev.pct(v['stopped'])} %", "Treffer bei Halt": round(v["hit_stopped"], 2),
                            "Treffer aller Läufe": round(v["hit_all"], 2), "Läufe mit Halt (Abstand 0,12)": f"{ev.pct(stop_n[thr]['stopped'])} %"} for thr, v in stop.items()]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 6: Wenn die Stärksten selten spielen")
cp = _coupling()
st.caption("Ungleicher Spielplan, 132 Partien, Aktivität 1 : 10: Hängt die Spielhäufigkeit an der Stärke? 60 feste Läufe.")
st.plotly_chart(build_coupling(cp), width="stretch", key="exp6_chart")
verdict(*ev.verdict_coupling(cp))
st.dataframe(pd.DataFrame([{"Aktivität": c, "Verzerrung ML": round(cp[c]["est"][ev.ML]["bias"], 2), "Verzerrung EB": round(cp[c]["est"][ev.EB]["bias"], 2),
                            "Sieger Siegquote": round(cp[c]["est"][ev.SIEGQUOTE]["winner"], 2), "Sieger EB": round(cp[c]["est"][ev.EB]["winner"], 2),
                            "Abdeckung Rang 1 Posterior": round(cp[c]["iv"][ev.POSTERIOR]["by_rank"][0], 2), "Abdeckung Rang 1 Bootstrap": round(cp[c]["iv"][ev.BOOTSTRAP]["by_rank"][0], 2)}
                           for c in C.COUPLINGS]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell stimmt** | Wahrheit und Schätzer sind beide Bradley-Terry. Mit Zyklen (nicht-transitive Stärken) sind die Intervalle zu eng; nicht gemessen. | Nicht-Transitivität |
| **Die Stärken sind normalverteilt um 0** | Das ist der Prior des Empirical Bayes. Gleichabständige Stärken (Standard) sind keine Normalverteilung: Empirical Bayes unterschätzt die Spreizung (Experiment 1). | – |
| **Wer spielt, hängt nicht an der Stärke** | Spielen die Stärksten selten, zieht Shrinkage ausgerechnet sie nach unten (Experiment 6). | – |
| **Alle Partien sind unabhängig** | Form, Heimvorteil und Zeitverlauf sind nicht modelliert. | Elo, WHR, Tore und Heimvorteil |
| **Das Intervall hat 90 %** | Das Posterior-Intervall über ganzzahlige Ränge ist konservativ; der Bootstrap der Partien liegt bei wenigen Partien darunter (Experiment 4). | – |
| **Die Partien sind gegeben** | Hier wird nur ausgewertet, nicht gewählt; welche Partie als Nächstes sinnvoll ist, zeigt das aktive Ranking. | Aktives Ranking |
| **Alles ist simuliert** | Die Zahlen gelten für 12 Spieler, diese Abstände und dieses Aktivitätsverhältnis. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Spieler $i = 1, \dots, n$ mit Stärken $s_i$; $P(i \text{ schlägt } j) = \sigma(s_i - s_j)$. Daten: Siege $W_{ij}$, Partien $G_{ij}$.

**Schätzer.** *Siegquote:* $W_{i\cdot}/G_{i\cdot}$. *Bradley-Terry (ML):* $\hat s = \arg\max_s \sum_{i<j}\bigl[W_{ij}\ln\sigma(s_i - s_j) + W_{ji}\ln\sigma(s_j - s_i)\bigr] - \tfrac{\lambda}{2}\lVert s\rVert^2$
mit kleinem $\lambda = 0{,}01$ (damit der Schätzer bei Spielern ohne Sieg endlich bleibt); *Ridge:* derselbe Ausdruck mit gewähltem $\lambda$ (Prior $s_i \sim N(0, 1/\lambda)$);
gedämpftes Newton-Verfahren.

**Unsicherheit (Laplace).** $\Sigma = H^{-1}$ mit $H = \sum_{i<j} G_{ij}\, p_{ij}(1 - p_{ij})\,(e_i - e_j)(e_i - e_j)^\top + \lambda I$.

**Empirical Bayes.** $\hat\lambda = \arg\max_\lambda\; \ln p(D \mid \lambda)$ mit der Laplace-Näherung $\ln p(D \mid \lambda) \approx \ln \mathrm{post}(\hat s_\lambda) + \tfrac{n}{2}\ln\lambda - \tfrac12 \ln\det H_\lambda$
auf einem Gitter von 0,01 bis etwa 32. Die geschätzten Stärken sind die Ridge-Schätzung zu $\hat\lambda$; die Schrumpfung wirkt auf Spieler mit wenigen Partien stärker.

**Rangintervall.** Ziehungen $s^{(b)} \sim N(\hat s, \Sigma)$, $b = 1, \dots, 1000$; $r^{(b)}_i$ der Rang von $i$ in Ziehung $b$ (1 = Bester). Untere Grenze: kleinster Rang $r$ mit
$\hat P(r_i \le r) \ge (1 - L)/2$, obere: kleinster Rang mit $\hat P(r_i \le r) \ge 1 - (1 - L)/2$. *Bootstrap der Partien:* $N$ Partien mit Zurücklegen, Neuanpassung mit $\hat\lambda$, 100 Wiederholungen,
dieselbe Quantilregel. *Wald (ML):* wie Posterior mit dem Bradley-Terry-ML und seiner Kovarianz.

**Maße.** *Verzerrung des geschätzten Besten:* $\hat s_{i^*} - \bar{\hat s} - (s_{i^*} - \bar s)$ mit $i^* = \arg\max_i \hat s_i$. *Spreizung:* $\mathrm{sd}(\hat s)/\mathrm{sd}(s)$. *MSE:*
$\tfrac1n\sum_i \bigl[(\hat s_i - \bar{\hat s}) - (s_i - \bar s)\bigr]^2$. *Paartreffer:* Anteil der Paare, die $\hat s$ wie die Wahrheit ordnet (Gleichstand ½). *Abdeckung:* Anteil der Spieler, deren wahrer
Rang im Intervall liegt. *P(Bester):* $\hat P(r_i = 1)$.

**Zufall.** SplitMix64-Strom des Generators (Box-Muller für die Normalwerte); Ausgänge, Paarwahl, Ziehungen und Bootstrap in fester Anordnung je Instanz.

**Literatur.**
- Efron, B., Morris, C. (1975): Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association* 70, 311–319 (nicht eingesehen).
- Laird, N. M., Louis, T. A. (1989): Empirical Bayes ranking methods. *Journal of Educational Statistics* 14, 29–46 (nicht eingesehen).
- Goldstein, H., Spiegelhalter, D. J. (1996): League tables and their limitations. *Journal of the Royal Statistical Society A* 159, 385–443 (nicht eingesehen).
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs. *Biometrika* 39, 324–345 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
