"""Kompletter Durchlauf: Daten erzeugen, Fehler injizieren, univariat pruefen, Folien-Grafiken schreiben.

Aufruf:  python run_all.py
"""
from pathlib import Path

from hackathon_labimport_quality.hcc_synth import (
    generate_clean, inject_errors, plot_overview, LABS)
from hackathon_labimport_quality.ood_univariat import (
    check_univariate, plot_univariate, evaluate, plot_evaluation)

SEED = 42   # ein Seed fuer alle Daten

BASE = Path(__file__).resolve().parent
daten, folien = BASE / "daten", BASE / "folien"
daten.mkdir(exist_ok=True)
folien.mkdir(exist_ok=True)

clean = generate_clean(n=1000, seed=SEED)
df, truth = inject_errors(clean, seed=SEED)
clean.to_csv(daten / "hcc_clean.csv", index=False)
df.to_csv(daten / "hcc_mit_fehlern.csv", index=False)
truth.to_csv(daten / "hcc_ground_truth.csv", index=False)
plot_overview(df, truth, folien / "folie_datensatz.png")

flags = check_univariate(df, columns=LABS)
flags.to_csv(daten / "univariat_flags.csv", index=False)
plot_univariate(df, flags, folien / "folie_univariat_dichten.png", columns=LABS)

res = evaluate(df, flags, truth)
plot_evaluation(res, folien / "folie_univariat_erkennung.png")
print(res.to_string())
print("Fehlalarme:", res.attrs["fehlalarme"], "von", res.attrs["zeilen_ohne_fehler"])