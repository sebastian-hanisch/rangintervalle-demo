# Rangintervalle und Shrinkage: wie sicher ist Platz 3? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-rangintervalle-demo.streamlit.app/)**

Dreizehntes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `aktives-ranking-demo` und `bradley-terry-demo`: Eine Rangliste nennt einen Rang, aber nicht, wie sicher er ist.
Gemessen wird, **wie breit ein Rangintervall ist**, **wie sehr die Spitze der Liste überschätzt** wird (Gewinnerfluch), was **Shrinkage** (Empirical Bayes) daran ändert und was nicht
(**die Rangfolge**), ob die Intervalle **ihre Abdeckung halten** und was passiert, wenn **die Stärksten selten spielen**. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel
(Stärken aus dem Generator Version 1, wortgleich); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Wie sicher ist Platz 3? **Bei 66 Partien (12 Spieler, jedes Paar einmal) liegt der wahre Drittbeste mit 90 % Sicherheit irgendwo zwischen Rang 1,4 und 7,4**; die mittlere Breite aller
Intervalle ist 7,2 von 12 Rängen. Mit 264 Partien (vier je Paar) sind es 1,5 bis 4,8, mit 528 Partien Breite 3,3. **Die Spitze ist zu hoch geschätzt:** der geschätzte Beste liegt mit reiner
Bradley-Terry-Schätzung im Mittel **1,85 Stärkeeinheiten** (θ; ein Stärkeabstand zwischen Nachbarn ist 0,3) über seiner wahren Stärke, mit Ridge (λ = 0,1) 1,16, mit
**Empirical-Bayes-Shrinkage 0,18**; der Fehler der Stärken (MSE) fällt von 1,39 auf 0,37. **Die Rangfolge ändert Shrinkage kaum:** im ausgewogenen Plan gar nicht (Paartreffer
0,850 gegen 0,849), bei ungleicher Aktivität um 0,015 (0,811 gegen 0,826 bei 66 Partien). Shrinkage korrigiert die **Werte**, nicht die **Ordnung**. **Die Intervalle halten ihre
Abdeckung, sind aber konservativ:** das Posterior-Intervall deckt den wahren Rang in 0,96 bis 0,99 der Fälle (gewünscht 0,90), der Bootstrap der Partien bei ungleicher Aktivität und 66
Partien nur 0,89; den wahren Besten deckt auch das Posterior-Intervall dort nur zu 0,87. Und **wenn die Stärksten selten spielen**, kostet Shrinkage Treffer: Sieger richtig 0,27 gegen
0,55 bei der Siegquote.

## Modell

- **Daten:** Siegmatrix W und Partienmatrix G = W + Wᵀ; wahre Siegwahrscheinlichkeit σ(s_i − s_j); wahre Stärken gleichabständig (Generator, Standard) oder aus einer Normalverteilung mit
  derselben Standardabweichung gezogen. Zwei Spielpläne: **ausgewogen** (jedes Paar k-mal) und **ungleich** (zufällige Paare mit Auswahlgewicht a_i · a_j, Aktivität a geometrisch von 1
  bis 10); die Aktivität ist unabhängig von der Stärke oder an sie gekoppelt (Experiment 6).
- **Schätzer:** *Siegquote*; *Bradley-Terry (ML)* (Ridge λ = 0,01, damit der Schätzer bei Spielern ohne Sieg endlich bleibt); *Ridge* (λ als Regler); *Empirical Bayes* (λ aus der
  Laplace-Näherung der Evidenz ln p(D | λ) ≈ ln post(ŝ) + (n/2) ln λ − ½ ln det H, Gitter 0,01 bis 31). Gedämpftes Newton-Verfahren (Rückwärtsschritt), Laplace-Kovarianz Σ = H⁻¹.
- **Rangintervall:** je Spieler das Quantilintervall der ganzzahligen Rangverteilung aus 1000 Ziehungen (*Posterior*: aus N(ŝ_EB, Σ); *Wald*: aus N(ŝ_ML, Σ_ML)) oder aus 100
  Neuanpassungen auf Partien mit Zurücklegen (*Bootstrap der Partien*, λ des Empirical Bayes).
- **Maße:** Verzerrung des geschätzten Besten (geschätzt minus wahr, beide zentriert), Spreizung (Streuung der Schätzung durch Streuung der Wahrheit), MSE der Stärken, Paartreffer, Sieger
  richtig, Top-3, Abdeckung und Breite der Intervalle (auch nach wahrem Rang), Kalibrierung von P(Bester).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 12 Spielern, Stärkeabstand 0,3, Aktivität 1 : 10,
  Partienzahlen 66 / 132 / 264 / 528, unabhängig vom Regler.
- **Gleiche Daten für alle Schätzer und Verfahren:** Alle vier Schätzer und die drei Intervalle sehen dieselben Partien einer Instanz.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); Normalwerte per Box-Muller aus demselben Strom; Ausgänge, Paarwahl, Ziehungen und
  Bootstrap in fester Anordnung je Instanz.
- **Rechenzeit:** Die Experimente rechnen beim ersten Aufruf rund eine Minute und werden danach zwischengespeichert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 12 Spieler, δ = 0,3, Aktivität 1 : 10 (ungleich), wo nichts anderes steht.

| Frage | Befund | Test |
|---|---|---|
| Verzerrung des geschätzten Besten (ML / Ridge / EB), 66 / 132 / 264 / 528 Partien | ausgewogen **+1,85 / +1,16 / +0,18**, +0,84 / +0,68 / +0,18, +0,47 / +0,42 / +0,16, +0,12 / +0,10 / −0,03; ungleich **+2,73 / +1,35 / +0,18**, +1,22 / +0,68 / +0,02, +0,47 / +0,37 / +0,02, +0,25 / +0,20 / +0,02. | `test_the_top_is_overestimated_and_shrinkage_removes_it` |
| Spreizung und MSE der Stärken | Spreizung ML 1,65 (66 Partien) → 1,14 (264), EB 0,81 → 0,96; MSE (ML / Ridge / EB) 1,39 / 0,62 / 0,37 (66, ausgewogen), 0,16 / 0,15 / 0,12 (264); ungleich, 66 Partien: 2,71 / 0,91 / 0,48. | `test_spread_and_error_of_the_strengths` |
| Paartreffer (Siegquote / ML / Ridge / EB) | ausgewogen: alle vier innerhalb 0,01 (66 Partien 0,850 / 0,851 / 0,848 / 0,849); ungleich, 66 Partien **0,811 / 0,816 / 0,819 / 0,826**, 264: 0,909 / 0,916 / 0,916 / 0,920; Sieger richtig ungleich 264: 0,68 (Siegquote) gegen 0,62 (EB). | `test_shrinkage_barely_changes_the_order` |
| Breite des 90-%-Posterior-Intervalls (66 / 132 / 264 / 528 Partien) | ausgewogen **7,2** / 5,6 / 4,2 / 3,3 von 12 Rängen, ungleich 7,8 / 6,3 / 5,0 / 3,9; wahrer Drittbester bei 66 Partien Rang **1,4 bis 7,4**, bei 264 Partien 1,5 bis 4,8. | `test_interval_width_and_the_third_best` |
| Abdeckung (gewünscht 0,90): Posterior / Bootstrap | Posterior 0,96 bis 0,99 in allen acht Fällen; Wald (ML) 0,96 (66 Partien, ausgewogen); Bootstrap 0,93 (ausgewogen, 66), **0,89** (ungleich, 66), 0,93 (ungleich, 132), 0,98 (ausgewogen, 264); den wahren Besten decken bei ungleich und 66 Partien Posterior 0,87 und Bootstrap **0,77**. | `test_coverage_of_the_intervals` |
| Abdeckung, wenn der Prior stimmt (Stärken normal gezogen, 66 Partien) | Posterior 0,96 (beide Spielpläne), Bootstrap 0,92 (ausgewogen) und **0,88** (ungleich). | `test_with_a_matching_prior_the_posterior_is_still_conservative` |
| Kalibrierung von P(Bester) (gepoolt: 66 und 264 Partien, beide Spielpläne, 2880 Spieler-Läufe) | gesagt / tatsächlich in fünf Klassen: 0,01 / 0,01; 0,10 / 0,10; 0,29 / 0,35; 0,54 / 0,45; **0,82 / 0,78** (n = 63): alle Abweichungen unter 0,12. | `test_p_best_is_roughly_calibrated` |
| Abbruchregel (Halt bei P(Bester) ≥ Schwelle, nach jeder vollständigen Runde, höchstens 12 Runden) | Schwelle 0,8 / 0,9 / 0,95: Läufe mit Halt 82 % / 47 % / 28 %; Treffer bei Halt **0,86 / 0,89 / 1,00**; Median der Partien 396 / 792 / 792 (Budget 792). Enges Feld (δ = 0,12): Halt bei 0,9 in 10 % der Läufe, Treffer aller Läufe 0,57. | `test_the_stop_rule` |
| Die Stärksten spielen selten (132 Partien, ungleich) | Verzerrung ML **+2,40** (unabhängig +1,22, die Besten spielen am meisten +0,58); Sieger richtig **0,27 (EB) gegen 0,55 (Siegquote)**, Top-3 0,60 gegen 0,66; Bootstrap deckt den wahren Besten zu 0,78 (Posterior 0,97). | `test_when_the_strongest_play_rarely_shrinkage_costs_hits` |
| Verdikte nennen dieselben Zahlen | Die live berechneten Texte der App geben die obigen Werte wieder. | `test_verdicts_quote_the_measured_numbers` |

Die Kalibrierung und die Treffer der Abbruchregel beruhen auf 63 bzw. 17 bis 49 Läufen je Zeile und sind **grob**: Standardfehler 0,05 bis 0,10.

## Befunde und Korrekturen gegenüber dem Plan

- **Zwei Aussagen der Vorab-Messung (100 Läufe, anderer Zufallsstrom) tragen nicht:** (1) „die oberste P(Bester)-Klasse ist zu zuversichtlich“ (0,80 gesagt gegen 0,54 tatsächlich, n = 13):
  gepoolt über vier Fälle und 60 Läufe sind es **0,82 gegen 0,78**, alle Klassen weichen weniger als 0,12 ab; (2) „die Abbruchregel verfehlt ihr Versprechen“ (Schwelle 0,8 trifft 0,76):
  über 60 feste Läufe trifft sie **0,86**. Die Abweichungen der Vorab-Messung lagen im Standardfehler (0,05).
- **Die Zahlen verschieben sich** gegen die Vorab-Messung (Verzerrung ML 1,57 → 1,85, Posterior-Breite 7,1 → 7,2, Sieger richtig bei selten spielenden Besten 0,20 → 0,27, Siegquote 0,41 → 0,55);
  die Aussagen bleiben (Spitze überschätzt, Shrinkage ändert die Rangfolge kaum, Intervalle konservativ, Bootstrap bei wenigen Partien darunter).
- **Neu gegen den Plan:** das Posterior-Intervall deckt den wahren **Besten** bei ungleicher Aktivität und 66 Partien nur zu 0,87 (unter den gewünschten 0,90), obwohl die mittlere Abdeckung
  0,97 ist; der Durchschnitt über alle Spieler verdeckt es (Experiment 4, Tabelle nach wahrem Rang).
- **Empirical Bayes ist nicht erwartungstreu:** Bei gleichabständigen Stärken (keine Normalverteilung) unterschätzt es die Spreizung (0,81 bei 66 Partien); die Verzerrung der Spitze bleibt klein, ist aber
  nicht null (+0,18).

## Abgrenzung

- **Bradley-Terry** (Stück 5): Schätzer und Standardfehler mit λ als gesetzter Vorannahme; hier λ aus den Daten (Empirical Bayes) und Intervalle für **Ränge**, nicht für θ.
- **Aktives Ranking** (Stück 12): wählt die nächste Partie nach der Unsicherheit; hier die Abbruchregel als Zusatzzeile (Halt bei P(Bester) ≥ Schwelle) und die Bewertung der Intervalle.
- **Mehrfachtests und Gewinnerfluch** (`multiple-testing-demo`): korrigiert die **Teststatistik**; hier wird die **Schätzung** korrigiert (Shrinkage), und die Rangfolge bleibt fast gleich.
- **Glicko und TrueSkill** (Stück 8): Unsicherheit je Spieler mitführen; hier die Unsicherheit des **Rangs** und was Shrinkage am Wert ändert.
- **Nachfolger (nur beim Namen):** Learning to Rank.

## Ehrliche Grenzen

- **Das Modell stimmt:** Wahrheit und Schätzer sind beide Bradley-Terry; mit Zyklen (Stück 11) wären die Intervalle zu eng. Nicht gemessen.
- **Der Prior ist normal um 0:** Gleichabständige Stärken (Standard) sind keine Normalverteilung; Empirical Bayes unterschätzt dann die Spreizung (Experiment 1). Die Normal-Wahrheit zeigt, was
  bei passendem Prior passiert.
- **Die Abdeckung ist über feste Wahrheiten gemessen:** Wer den Prior anders wählt, erhält andere Zahlen; die Aussage ist „so verhalten sich diese Verfahren hier“.
- **12 Spieler, ein Aktivitätsverhältnis (1 : 10):** Andere Größen nur über die Regler, nicht in den festen Läufen.
- **Ganzzahlige Ränge machen das Posterior-Intervall konservativ** (gemessen 0,96 bis 0,99 gegen 0,90); das ist kein Qualitätsmerkmal.
- **Der Bootstrap der Partien ist ein Gegenstück, nicht die einzige richtige Methode:** der parametrische Bootstrap und Bayes-Bootstraps sind nicht umgesetzt.
- **Alles simuliert:** keine Aussage über eine echte Liga.

## Quellen

- Efron, B., Morris, C. (1975): Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association* 70, 311–319 (eingesehen 2026-10-17, Abschnitt 1, Gl. (1.7)–(1.10), und Abschnitt 2, Gl. (2.3): Stein-Schätzer als Empirical-Bayes-Schätzer mit aus den Daten geschätztem Schrumpfungsfaktor, Schrumpfung zum Gesamtmittel; die Demo schätzt λ stattdessen über die Laplace-Evidenz).
- Laird, N. M., Louis, T. A. (1989): Empirical Bayes ranking methods. *Journal of Educational Statistics* 14, 29–46 (eingesehen 2026-10-17, S. 31–33, Gl. (2)–(3) und Anhang A: Rang als Summe von Indikatoren, erwartete Ränge aus der Posterior-Verteilung; bei gleicher Posterior-Streuung stimmen sie mit der Ordnung der Posterior-Mittel überein, bei ungleicher Streuung wandern Einheiten mit großer Streuung zum mittleren Rang).
- Goldstein, H., Spiegelhalter, D. J. (1996): League tables and their limitations. *Journal of the Royal Statistical Society A* 159, 385–443 (eingesehen 2026-10-17, Abschnitt 3.2, S. 391–392, und Abschnitt 5.2, S. 400–402: Intervalle für Ränge aus simulierten Rangverteilungen (Bootstrap oder MCMC); die Rangintervalle des Zufallseffekte-Modells mit Schrumpfung sind noch breiter bzw. konservativer als die der unabhängigen Schätzung).
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs. *Biometrika* 39, 324–345 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** Arbeiten zu Bayes-Bootstrap und zu simultanen Rangintervallen; die Zahlen dieser Demo sind Simulation.

## Tests

99 Tests, gut eine Minute (die Experimente rechnen über 60 feste Läufe): Einheiten gegen Handwerte (`test_model.py`: Schätzer, Evidenz, Empirical Bayes, Maße; `test_intervals.py`: Ränge, Quantilintervalle,
Ziehungen, Simulation der Daten), Generator (`test_generator.py`, wortgleich), **Orakel** (`test_oracle_rangintervalle.py`: Schätzer gegen den BFGS von scipy, Laplace-Kovarianz gegen finite
Differenzen, Evidenz gegen numerische Integration bei zwei Spielern, Rangwahrscheinlichkeiten gegen exakte Quadratur und Aufzählen der Anordnungen, Eigenschaften von Shrinkage: gleiche
Ordnung im ausgewogenen Plan, Spreizung, λ des Empirical Bayes), Maße und Verdikte (`test_evaluation.py`), alle Zahlen dieses README (`test_claims.py`), Presets, Szenario und PDF
(`test_presets.py`), Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die vier `rgi_`-Module; schnelle Testdateien zuerst, Überlebende gegen die Claims-Tests): **148 Stellen, 139 getötet, 9 äquivalent.** Im ersten, schnellen Lauf überlebten 60 Stellen (88 getötet); 49 davon sind gegen die Claims-Tests oder mit neuen Einheitentests getötet (die Vorzeichen in Verzerrung und MSE bei Mittel ungleich null, die Box-Muller-Zuordnung, der Bootstrap gegen eine unabhängige Neuziehung, die Wahl des Drittbesten bei zwei Spielern, die Klassengrenzen der Kalibrierung, die Rundenzahl und das Budget der Abbruchregel). **Echte Lücken** waren: die Siegquote eines Spielers mit einer einzigen Partie, die Vorzeichen von Verzerrung und MSE bei Mitteln ungleich null (die Handwerte hatten Mittel 0), die letzte Zeile des Bootstraps (nicht initialisiert), die Randbedingung `>=` der Abdeckung, das letzte Paar im ungleichen Spielplan, die Skala der normal gezogenen Stärken, die Klassengrenzen der Kalibrierung, die Runden und das Budget der Abbruchregel und deren Halt genau an der Schwelle; sie sind mit Tests geschlossen und auf einer Kopie einzeln getötet. **9 sind äquivalent:** `>` gegen `>=` bei Gleitkommagleichstand der Evidenz, der Aufschlag 10⁻¹⁰ in der Cholesky-Zerlegung, die Toleranz 10⁻¹² im Rückwärtsschritt, die Obergrenze `games − 1` des Ziehungsindex und `pairs − 1` der Paarwahl (die Gleichverteilung bleibt unter 1), `<` gegen `<=` bei stetigen Würfen und am Rand 0,02 des Hinweises im Verdikt.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `rgi_model.py` | Bradley-Terry (gedämpftes Newton), Laplace-Kovarianz, Evidenz, Empirical Bayes, Maße |
| `rgi_intervals.py` | Normalwerte, Ränge, Quantilintervalle, Posterior- und Bootstrap-Ränge |
| `rgi_scenario.py` | Reglerwerte (`Settings`) und die Simulation der Daten |
| `rgi_evaluation.py` | Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `rgi_presets.py`, `rgi_constants.py` | Permalink, Presets, Grenzen |
| `rgi_visualization.py`, `rgi_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **James-Stein-Formel, vollständiger Bayes (MCMC)** und der parametrische Bootstrap.
- **Simultane Rangintervalle** (alle Ränge zugleich zum Niveau L).
- **Zyklen und Zeitachse** (nur eine transitive, feste Stärke je Spieler).

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
