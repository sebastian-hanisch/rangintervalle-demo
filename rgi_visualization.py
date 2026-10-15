"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import rgi_constants as C
import rgi_evaluation as ev

SHORT = {C.DESIGNS[0]: "ausgewogen", C.DESIGNS[1]: "ungleich"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _order(a: ev.Analysis):
    """Spieler nach wahrer Stärke, der Beste zuerst; Beschriftung = wahrer Rang."""
    idx = np.argsort(-a.instance.strengths, kind="stable")
    return idx, [str(i + 1) for i in range(len(idx))]


def build_rank_intervals(a: ev.Analysis):
    """Geschätzter Rang mit Rangintervall je Spieler (Balken) und wahrer Rang (Raute); Spieler nach wahrer Stärke geordnet."""
    idx, labels = _order(a)
    est = a.est_rank[idx].astype(float)
    lo, hi = a.lo[idx].astype(float), a.hi[idx].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=est, mode="markers", name=f"geschätzter Rang ({a.settings.estimator})", marker=dict(color=C.COLORS[a.settings.estimator], size=9),
                             error_y=dict(type="data", symmetric=False, array=np.maximum(hi - est, 0), arrayminus=np.maximum(est - lo, 0), visible=True, color=C.COLOR_BAND),
                             hovertemplate="wahrer Rang %{x}: geschätzt %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[int(r) for r in a.true_rank[idx]], mode="markers", name="wahrer Rang", marker=dict(color=C.COLOR_TRUTH, size=8, symbol="diamond"),
                             hovertemplate="wahrer Rang %{y}<extra></extra>"))
    fig.update_xaxes(title_text="Spieler, nach wahrer Stärke geordnet (1 = Bester)", type="category")
    fig.update_yaxes(title_text="Rang (1 = Bester)", autorange="reversed")
    return _base(fig, height=400)


def build_shrinkage(a: ev.Analysis):
    """Stärken in Elo-Punkten: Bradley-Terry (ML), gewählter Schätzer und die wahre Stärke; Spieler nach wahrer Stärke geordnet."""
    idx, labels = _order(a)
    s = a.instance.strengths
    fig = go.Figure()
    for name, v in ((ev.ML, a.fit.values[ev.ML]), (a.settings.estimator, a.fit.values[a.settings.estimator])):
        if name == ev.SIEGQUOTE:
            continue
        if any(t.name == name for t in fig.data):
            continue
        fig.add_trace(go.Scatter(x=labels, y=C.BASE + C.SCALE * (v - v.mean())[idx], mode="markers", name=name, marker=dict(color=C.COLORS[name], size=9),
                                 hovertemplate="wahrer Rang %{x}: " + name + " %{y:.0f} Punkte<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=C.BASE + C.SCALE * (s - s.mean())[idx], mode="markers", name="wahre Stärke", marker=dict(color=C.COLOR_TRUTH, size=8, symbol="diamond"),
                             hovertemplate="wahrer Rang %{x}: wahr %{y:.0f} Punkte<extra></extra>"))
    fig.update_xaxes(title_text="Spieler, nach wahrer Stärke geordnet (1 = Bester)", type="category")
    fig.update_yaxes(title_text="Elo-Punkte (um 1500)")
    return _base(fig, height=380)


def build_pbest(a: ev.Analysis):
    idx, labels = _order(a)
    colors = [C.COLOR_TRUTH if int(a.true_rank[i]) == 1 else C.COLOR_BAND for i in idx]
    fig = go.Figure(go.Bar(x=labels, y=a.p_best[idx], marker_color=colors, hovertemplate="wahrer Rang %{x}: P(Bester) %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Spieler, nach wahrer Stärke geordnet (1 = Bester)", type="category")
    fig.update_yaxes(title_text="Posterior-Wahrscheinlichkeit für Rang 1", range=[0, 1])
    return _base(fig, height=320)


def _lines(rows: dict, xs, title: str, yrange=None, names=None, dash=None):
    fig = go.Figure()
    for name, ys in rows.items():
        fig.add_trace(go.Scatter(x=list(xs), y=ys, mode="lines+markers", name=name, line=dict(color=C.COLORS.get(name, "#4c78a8"), width=3, dash=(dash or {}).get(name)),
                                 hovertemplate=name + ", %{x} Partien: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Partien", type="category")
    fig.update_yaxes(title_text=title, **({"range": yrange} if yrange else {}))
    return _base(fig, height=360)


def build_bias(main, design: str):
    rows = {name: [main[design][g]["est"][name]["bias"] for g in C.EXP_GAMES] for name in (ev.ML, ev.RIDGE, ev.EB)}
    fig = _lines(rows, C.EXP_GAMES, "Verzerrung des geschätzten Besten (θ)")
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    return fig


def build_order(main, design: str):
    fig = go.Figure()
    for name in C.ESTIMATORS:
        fig.add_trace(go.Bar(x=[f"{g} Partien" for g in C.EXP_GAMES], y=[main[design][g]["est"][name]["acc"] for g in C.EXP_GAMES], name=name, marker_color=C.COLORS[name],
                             hovertemplate=name + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Paartreffer", range=[0.7, 1.0])
    return _base(fig, height=340)


def build_width(main):
    fig = go.Figure()
    for design, color in zip(C.DESIGNS, ("#2F6B65", "#c0392b")):
        fig.add_trace(go.Scatter(x=[f"{g}" for g in C.EXP_GAMES], y=[main[design][g]["iv"][ev.POSTERIOR]["width"] for g in C.EXP_GAMES], mode="lines+markers", name=SHORT[design],
                                 line=dict(color=color, width=3), hovertemplate=SHORT[design] + ", %{x} Partien: Breite %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text="Partien", type="category")
    fig.update_yaxes(title_text="mittlere Breite des 90-%-Rangintervalls (Ränge)", range=[0, C.EXP_N])
    return _base(fig, height=340)


def build_coverage(cells: dict):
    """cells: Bezeichnung -> Zelle; je Verfahren ein Balken, die Linie ist die gewünschte Abdeckung."""
    fig = go.Figure()
    colors = {ev.POSTERIOR: "#2F6B65", ev.WALD: "#9aa0a6", ev.BOOTSTRAP: "#8e6bb0"}
    for method in (ev.POSTERIOR, ev.WALD, ev.BOOTSTRAP):
        labels = [k for k, c in cells.items() if method in c["iv"]]
        fig.add_trace(go.Bar(x=labels, y=[cells[k]["iv"][method]["cover"] for k in labels], name=method, marker_color=colors[method],
                             hovertemplate="%{x}, " + method + ": %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.9, line_dash="dash", line_color="rgba(150,150,150,0.8)", annotation_text="gewünscht 0,90", annotation_position="bottom right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil der Spieler, deren wahrer Rang im Intervall liegt", range=[0.7, 1.0])
    return _base(fig, height=380)


def build_calibration(cal):
    xs = [r[3] for r in cal if r[2] > 0]
    ys = [r[4] for r in cal if r[2] > 0]
    ns = [r[2] for r in cal if r[2] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfekt kalibriert", line=dict(color="rgba(150,150,150,0.8)", dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="gemessen", marker=dict(color=C.COLOR_BAND, size=10), line=dict(color=C.COLOR_BAND, width=2),
                             text=[f"n = {n}" for n in ns], hovertemplate="gesagt %{x:.2f}, tatsächlich %{y:.2f} (%{text})<extra></extra>"))
    fig.update_xaxes(title_text="gesagte Wahrscheinlichkeit für Rang 1", range=[0, 1])
    fig.update_yaxes(title_text="tatsächlicher Anteil des wahren Besten", range=[0, 1])
    return _base(fig, height=340)


def build_stop(stop: dict, stop_narrow: dict):
    thr = [f"≥ {t:.2f}".replace(".", ",") for t in stop]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=thr, y=[stop[t]["hit_stopped"] for t in stop], name="Treffer bei Halt (Abstand 0,3)", marker_color="#2F6B65",
                         hovertemplate="Schwelle %{x}: Treffer %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=thr, y=[stop_narrow[t]["hit_all"] for t in stop_narrow], name="Treffer aller Läufe (Abstand 0,12)", marker_color="#c0392b",
                         hovertemplate="Schwelle %{x}: Treffer %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=thr, y=list(stop), mode="markers", name="versprochen", marker=dict(color="#e67e22", size=11, symbol="line-ew-open", line=dict(width=3))))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil: der gemeldete Beste ist der wahre Beste", range=[0, 1.05])
    return _base(fig, height=340)


def build_coupling(cp: dict):
    fig = go.Figure()
    names = ("Siegquote", "Bradley-Terry (ML)", "Empirical Bayes")
    labels = {C.COUPLINGS[0]: "unabhängig", C.COUPLINGS[1]: "Beste spielen viel", C.COUPLINGS[2]: "Beste spielen wenig"}
    for name in names:
        fig.add_trace(go.Bar(x=[labels[c] for c in C.COUPLINGS], y=[cp[c]["est"][name]["winner"] for c in C.COUPLINGS], name=name, marker_color=C.COLORS[name],
                             hovertemplate=name + ", %{x}: %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Sieger richtig (Anteil der Läufe)", range=[0, 1])
    return _base(fig, height=340)
