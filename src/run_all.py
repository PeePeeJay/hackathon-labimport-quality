"""Kompletter Durchlauf: Daten erzeugen, Fehler injizieren, pruefen, Grafiken schreiben.

Aufruf:  python run_all.py
"""
from pathlib import Path

import pandas as pd

from hackathon_labimport_quality.hcc_synth import make_dataset, plot_overview, LABS
from hackathon_labimport_quality.ood_univariat import (
    check_univariate, plot_univariate, evaluate, plot_evaluation)
from hackathon_labimport_quality.ood_isoforest import check_isolation_forest
from hackathon_labimport_quality.plot_interaktiv import plot_overview_html

SEED = 42          # ein Seed fuer alle Daten (Kohorte, Fehler, Reihenfolge)
ISO_ANTEIL = 0.15  # Isolation Forest: erwarteter Anteil auffaelliger Zeilen (Annahme)

BASE = Path(__file__).resolve().parent
daten, folien = BASE / "daten", BASE / "folien"
daten.mkdir(exist_ok=True)
folien.mkdir(exist_ok=True)

# 1. Daten erzeugen und Fehler injizieren
clean, df, truth = make_dataset(n=1000, seed=SEED)
clean.to_csv(daten / "hcc_clean.csv", index=False)
df.to_csv(daten / "hcc_mit_fehlern.csv", index=False)
truth.to_csv(daten / "hcc_ground_truth.csv", index=False)
plot_overview(df, truth, folien / "folie_datensatz.png")

# 2. Univariate Pruefung
flags = check_univariate(df, columns=LABS)
flags.to_csv(daten / "univariat_flags.csv", index=False)
plot_univariate(df, flags, folien / "folie_univariat_dichten.png", columns=LABS)
res_uni = evaluate(df, flags, truth)
plot_evaluation(res_uni, folien / "folie_univariat_erkennung.png")

# 3. Isolation Forest (multivariat)
iso = check_isolation_forest(df, columns=LABS, contamination=ISO_ANTEIL, seed=SEED)
iso.to_csv(daten / "isoforest_scores.csv", index=False)
iso_flags = iso[iso["markiert"]]
res_iso = evaluate(df, iso_flags, truth)
plot_evaluation(res_iso, folien / "folie_isoforest_erkennung.png", titel="Isolation Forest")

# 4. Interaktiver Bericht: Panel 1 univariat, Panel 2 bivariat mit Isolation Forest
plot_overview_html(df, truth, folien / "laborwerte_interaktiv.html", clean=clean, iso=iso)

# 5. Zusammenfassung
beide = {"patient_id": list(set(flags["patient_id"]) | set(iso_flags["patient_id"]))}
res_beide = evaluate(df, pd.DataFrame(beide), truth)
for name, res in [("Univariat", res_uni), ("Isolation Forest", res_iso),
                  ("Beide kombiniert", res_beide)]:
    print(f"\n{name}: {res['erkannt'].sum()} von {res['injiziert'].sum()} erkannt, "
          f"{res.attrs['fehlalarme']} Fehlalarme auf {res.attrs['zeilen_ohne_fehler']}")
    print(res.to_string())