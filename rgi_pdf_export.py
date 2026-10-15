"""PDF-Export der aktuellen Konfiguration und der Kennzahlen (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import rgi_constants as C
import rgi_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("λ", "lambda"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-"), ("„", '"'), ("“", '"')):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Rangintervalle und Shrinkage: wie sicher ist Platz 3?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Stärkeabstand {ev.de(s.delta)}, Design: {s.design}, Partien: {s.games} (gespielt {a.instance.games}), Wahrheit: {s.truth}, "
                                  f"Schätzer: {s.estimator}, Intervall: {s.interval} ({s.level} %), Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Schätzer", "Paartreffer", "Sieger", "Top-3", "Verzerrung Bester", "MSE")
    widths = (56, 24, 18, 18, 38, 24)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for name in C.ESTIMATORS:
        e = a.est[name]
        row = (name, f"{e['acc']:.3f}", f"{e['winner']:.2f}", f"{e['top3']:.2f}", "-" if name == C.ESTIMATORS[0] else f"{e['bias']:+.2f}", "-" if name == C.ESTIMATORS[0] else f"{e['mse']:.3f}")
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Stärken aus dem Generator der Ranking-Linie, Ausgänge aus der logistischen Gewinnwahrscheinlichkeit); der wahre Rang ist nur in der Simulation bekannt. "
        "Shrinkage korrigiert die Werte, nicht die Ordnung. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
