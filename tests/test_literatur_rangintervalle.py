"""Literatur-Referenzwerte zu Rangintervallen und Shrinkage (ORACLE-PLAYBOOK §5.1). Nur Werte, die in der genannten Quelle eingesehen wurden; keiner stammt aus dem Code der Demo.

Wert 1: Bradley-Terry-Maximum-Likelihood-Schätzung der Zeitschriften-Zitationsdaten (Stigler 1994, übernommen aus Agresti 2002, S. 448).
  Quelle: Turner, H., Firth, D. (2012): Bradley-Terry models in R: The BradleyTerry2 package. Journal of Statistical Software 48(9); Vignette des Pakets BradleyTerry2,
  Abschnitt 2.1 „Example: Analysis of journal citations“, https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html (Referenzsoftware BTm).
  Fundstelle: Zählmatrix „cited/citing“ (4 × 4: Biometrika, Comm Statist, JASA, JRSS-B) und die gedruckten Koeffizienten von `citeModel` (Biometrika = 0 als Identifikation):
  Comm Statist −2.9491, JASA −0.4796, JRSS-B 0.2690; mit Bezugsjournal JASA: Biometrika 0.4796, Comm Statist −2.4695, JRSS-B 0.7485. Gewinner = zitiertes Journal, Verlierer =
  zitierendes Journal; die Diagonale (Selbstzitate) geht nicht ein. Eingesehen am 2026-10-17. Die ML-Schätzung ist hier der Grenzwert des Ridge-Schätzers `fit_bt` für λ → 0
  (die Verschiebung ist unbestimmt; verglichen werden Differenzen). Standardfehler dieses Modells stehen in der eingesehenen Fassung nicht.

Wert 2: Anteil einer Normalverteilung innerhalb von μ ± 1σ, ± 2σ, ± 3σ: 0.682689492137086, 0.954499736103642, 0.997300203936740.
  Quelle: Wikipedia, „68–95–99.7 rule“, Tabelle der Anteile innerhalb k Standardabweichungen, https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule,
  eingesehen am 2026-10-17. Geprüft wird der Box-Muller-Strom `IV.normals`, aus dem alle Ziehungen der Rangverteilung stammen (der vorhandene Test prüft nur Momente)."""

import numpy as np
import pytest

import rgi_intervals as IV
import rgi_model as M
import turnier_generator as T

# Zeilen: zitiert (Gewinner), Spalten: zitierend (Verlierer); Reihenfolge Biometrika, Comm Statist, JASA, JRSS-B (Quelle, Abschnitt 2.1)
CITATIONS = np.array([[714, 730, 498, 221],
                      [33, 425, 68, 17],
                      [320, 813, 1072, 142],
                      [284, 276, 325, 188]], dtype=float)


def test_bradley_terry_maximum_likelihood_matches_the_published_journal_citation_fit():
    W = CITATIONS.copy()
    np.fill_diagonal(W, 0.0)
    s, _ = M.fit_bt(W, W + W.T, 1e-8, iters=100)
    assert (s - s[0])[1:] == pytest.approx([-2.9491, -0.4796, 0.2690], abs=6e-5)           # Bezug Biometrika
    assert (s - s[2]) == pytest.approx([0.4796, -2.4695, 0.0, 0.7485], abs=1.2e-4)         # Bezug JASA (gedruckt auf vier Stellen)


def test_box_muller_stream_has_the_normal_mass_within_one_two_and_three_standard_deviations():
    z = IV.normals(T.SplitMix64(2026), 400000)
    for k, mass in ((1, 0.682689492137086), (2, 0.954499736103642), (3, 0.997300203936740)):
        assert float(np.mean(np.abs(z) <= k)) == pytest.approx(mass, abs=3e-3)
