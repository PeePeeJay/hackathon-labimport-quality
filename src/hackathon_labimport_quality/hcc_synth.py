"""Synthetischer HCC-eCRF-Datensatz mit injizierten Fehlern (PoC OOD-Monitoring).

Zwei Schritte:
  1. generate_clean()  -> plausible Kohorte, korreliert ueber latente Variablen
  2. inject_errors()   -> bekannte Anzahl Fehler je Typ + Ground Truth

Alle Verteilungsparameter sind grobe Naeherungen fuer einen PoC,
KEINE validierten Literaturwerte.

Aufruf:  python hcc_synth.py --n 1000 --seed 42 --out .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABS_NORMAL = ["albumin_g_dl", "thrombozyten_g_l"]           # normalverteilt
LABS_HEAVY = ["bilirubin_mg_dl", "afp_ng_ml", "inr"]         # heavy-tailed (lognormal)
LABS = LABS_NORMAL + LABS_HEAVY

# Fehler je Typ (Default). Jeder Fehler trifft eine eigene Zeile.
DEFAULT_ERRORS = {
    "einheitenfehler": 10,      # Bilirubin in umol/l statt mg/dl (x17.1)
    "kommaverschiebung": 10,    # Laborwert x10 oder /10
    "feldtausch": 10,           # Albumin <-> INR vertauscht
    "unmoegliche_kombination": 10,  # Child-Pugh A trotz schwerer Dekompensation
    "platzhalterwert": 10,      # 999 / 0 / -1 als Laborwert
    "duplikat": 10,             # Zeile mit neuer Patienten-ID kopiert
}


def _pick(rng, p, labels):
    """Zeilenweise kategoriale Ziehung; p hat Form (n, k)."""
    p = p / p.sum(axis=1, keepdims=True)
    u = rng.random((len(p), 1))
    return np.asarray(labels)[(u > p.cumsum(axis=1)).sum(axis=1).clip(max=len(labels) - 1)]


def generate_clean(n: int = 1000, seed: int = 42, center: str = "Z1",
                   shift: float = 0.0) -> pd.DataFrame:
    """Plausible Kohorte. `shift` verschiebt die Leberfunktion (fuer 2. Zentrum)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(shift, 1, n)      # latent: Leberdysfunktion (hoch = schlechter)
    t = rng.normal(0, 1, n)          # latent: Tumorlast

    df = pd.DataFrame({"patient_id": [f"{center}-{i:05d}" for i in range(n)],
                       "zentrum": center})
    df["alter"] = rng.normal(65, 10, n).clip(18, 95).round().astype(int)
    df["geschlecht"] = rng.choice(["m", "w"], n, p=[0.77, 0.23])
    df["aetiologie"] = rng.choice(["Alkohol", "HCV", "HBV", "MASLD", "Sonstige"], n,
                                  p=[0.35, 0.22, 0.10, 0.23, 0.10])
    alk = df["aetiologie"].eq("Alkohol").to_numpy()
    p_alk = np.where(alk[:, None], [0.0, 0.55, 0.45], [0.50, 0.30, 0.20])
    df["alkoholkonsum"] = _pick(rng, p_alk, ["nie", "frueher", "aktuell"])

    # --- Labor: normalverteilt ---
    df["albumin_g_dl"] = (3.8 - 0.45 * z + rng.normal(0, 0.35, n)).clip(1.5, 5.5).round(1)
    df["thrombozyten_g_l"] = (175 - 45 * z + rng.normal(0, 50, n)).clip(20, 500).round()
    # --- Labor: heavy-tailed (lognormal) ---
    df["bilirubin_mg_dl"] = np.exp(np.log(1.0) + 0.55 * z + rng.normal(0, 0.45, n)).round(2)
    df["inr"] = (1.0 + np.exp(-2.0 + 0.6 * z + rng.normal(0, 0.5, n))).round(2)
    df["afp_ng_ml"] = (1.5 + np.exp(np.log(20) + 1.2 * t + rng.normal(0, 1.8, n))).round(1)

    # --- Tumor ---
    df["tumorgroesse_cm"] = np.exp(np.log(3.5) + 0.5 * t + rng.normal(0, 0.35, n)).clip(0.5, 25).round(1)
    df["anzahl_laesionen"] = 1 + rng.poisson(np.exp(-0.4 + 0.6 * t))
    metastase = rng.random(n) < 1 / (1 + np.exp(-(t - 2.0)))
    gefaess = rng.random(n) < 1 / (1 + np.exp(-(t - 1.0)))

    # --- Child-Pugh (Aszites/Enzephalopathie nur intern) ---
    asz = np.digitize(z + rng.normal(0, 0.7, n), [0.8, 1.8])
    enz = np.digitize(z + rng.normal(0, 0.7, n), [1.5, 2.4])
    pts = (5 + np.digitize(df["bilirubin_mg_dl"], [2, 3])
           + (2 - np.digitize(df["albumin_g_dl"], [2.8, 3.5]))
           + np.digitize(df["inr"], [1.7, 2.3]) + asz + enz)
    df["child_pugh"] = np.select([pts <= 6, pts <= 9], ["A", "B"], "C")

    # --- ECOG ---
    ecog = np.digitize(0.6 * z + 0.6 * t + rng.normal(0, 0.8, n), [0.3, 1.3, 2.2, 3.2])
    df["ecog"] = ecog

    # --- UICC (stark vereinfacht) ---
    size, nles = df["tumorgroesse_cm"], df["anzahl_laesionen"]
    df["uicc"] = np.select(
        [metastase, gefaess | ((nles > 1) & (size > 5)), (nles > 1) | (size > 5)],
        ["IV", "III", "II"], "I")

    # --- BCLC (vereinfacht) ---
    cp = df["child_pugh"]
    df["bclc"] = np.select(
        [(cp == "C") | (ecog > 2),
         (ecog >= 1) | metastase | gefaess,
         (nles > 3) | ((nles > 1) & (size > 3)),
         (nles == 1) & (size <= 2) & (cp == "A")],
        ["D", "C", "B", "0"], "A")
    return df


def inject_errors(df: pd.DataFrame, errors: dict[str, int] | None = None,
                  seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injiziert bekannte Fehler. Gibt (Daten, Ground Truth) zurueck."""
    errors = {**DEFAULT_ERRORS, **(errors or {})}
    rng = np.random.default_rng(seed)
    df = df.copy()
    idx = rng.permutation(df.index)
    truth, pos = [], 0

    def log(i, typ, feld, alt, neu):
        truth.append({"patient_id": df.at[i, "patient_id"], "fehlertyp": typ,
                      "feld": feld, "wert_original": alt, "wert_fehler": neu})

    for typ, k in errors.items():
        rows, pos = idx[pos:pos + k], pos + k
        for i in rows:
            if typ == "einheitenfehler":
                alt = df.at[i, "bilirubin_mg_dl"]
                df.at[i, "bilirubin_mg_dl"] = round(alt * 17.1, 2)
                log(i, typ, "bilirubin_mg_dl", alt, df.at[i, "bilirubin_mg_dl"])
            elif typ == "kommaverschiebung":
                feld = rng.choice(LABS)
                alt = df.at[i, feld]
                df.at[i, feld] = round(alt * rng.choice([10, 0.1]), 2)
                log(i, typ, feld, alt, df.at[i, feld])
            elif typ == "feldtausch":
                a, b = df.at[i, "albumin_g_dl"], df.at[i, "inr"]
                df.at[i, "albumin_g_dl"], df.at[i, "inr"] = b, a
                log(i, typ, "albumin_g_dl<->inr", a, b)
            elif typ == "unmoegliche_kombination":
                alt = df.at[i, "child_pugh"]
                df.at[i, "child_pugh"] = "A"
                df.at[i, "bilirubin_mg_dl"] = round(rng.uniform(5, 12), 2)
                df.at[i, "albumin_g_dl"] = round(rng.uniform(1.8, 2.5), 1)
                log(i, typ, "child_pugh|bilirubin_mg_dl|albumin_g_dl", alt, "A")
            elif typ == "platzhalterwert":
                feld = rng.choice(LABS)
                alt = df.at[i, feld]
                df.at[i, feld] = rng.choice([999, 0, -1])
                log(i, typ, feld, alt, df.at[i, feld])
            elif typ == "duplikat":
                new = df.loc[i].copy()
                new["patient_id"] = f"{df.at[i, 'zentrum']}-D{len(truth):04d}"
                df.loc[len(df)] = new
                truth.append({"patient_id": new["patient_id"], "fehlertyp": typ,
                              "feld": "zeile", "wert_original": df.at[i, "patient_id"],
                              "wert_fehler": new["patient_id"]})
            else:
                raise ValueError(f"Unbekannter Fehlertyp: {typ}")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, pd.DataFrame(truth)


def plot_overview(df: pd.DataFrame, truth: pd.DataFrame, path: Path) -> None:
    """Folien-Grafik: Verteilung der Laborwerte, injizierte Fehler markiert."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    titles = {"albumin_g_dl": "Albumin (g/dl) – normal",
              "thrombozyten_g_l": "Thrombozyten (G/l) – normal",
              "bilirubin_mg_dl": "Bilirubin (mg/dl) – heavy-tailed",
              "afp_ng_ml": "AFP (ng/ml) – heavy-tailed",
              "inr": "INR – heavy-tailed"}
    PLATZHALTER = [999, 0, -1]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), facecolor="#fcfcfb")
    for ax, lab in zip(axes, LABS):
        # nur Zeilen markieren, in denen genau dieses Feld manipuliert wurde
        ids = truth.loc[truth["feld"].str.contains(lab, regex=False), "patient_id"]
        bad = df["patient_id"].isin(ids)
        ph = bad & df[lab].isin(PLATZHALTER)
        x_ok, x_bad = df.loc[~bad, lab], df.loc[bad & ~ph, lab]
        log = lab in LABS_HEAVY and lab != "inr"
        allx = pd.concat([x_ok, x_bad])
        bins = (np.geomspace(allx.min(), allx.max(), 45) if log
                else np.linspace(allx.min(), allx.max(), 45))
        ax.hist(x_ok, bins=bins, color=BLUE, alpha=0.85, label="plausibel")
        ymax = ax.get_ylim()[1]
        ax.plot(x_bad, np.full(len(x_bad), -0.05 * ymax), "|", color=ORANGE, ms=12,
                mew=2, label="injizierter Fehler in diesem Feld", clip_on=False)
        if ph.sum():
            ax.text(0.98, 0.95, f"+ {int(ph.sum())} Platzhalter\n(999 / 0 / −1)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    color=MUTED)
        if log:
            ax.set_xscale("log")
        ax.set_title(titles[lab], fontsize=10.5, color=INK, loc="left")
        ax.set_facecolor("#fcfcfb")
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.grid(axis="y", color="#e8e7e2", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Anzahl Patienten", color=MUTED, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK)
    fig.suptitle(f"Synthetische HCC-Kohorte: n = {len(df)}, davon {len(truth)} "
                 "Zeilen mit injiziertem Fehler", x=0.01, ha="left", fontsize=13,
                 color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("../../../../ood-poc"))
    a = ap.parse_args()

    clean = generate_clean(a.n, a.seed)
    data, truth = inject_errors(clean, seed=a.seed + 1)
    a.out.mkdir(parents=True, exist_ok=True)
    clean.to_csv(a.out / "hcc_clean.csv", index=False)
    data.to_csv(a.out / "hcc_mit_fehlern.csv", index=False)
    truth.to_csv(a.out / "hcc_ground_truth.csv", index=False)
    plot_overview(data, truth, a.out / "folie_datensatz.png")
    print(f"{len(data)} Zeilen, {len(truth)} Fehler")
    print(truth["fehlertyp"].value_counts().to_string())
    for c in ["child_pugh", "bclc", "uicc", "ecog"]:
        print(clean[c].value_counts(normalize=True).sort_index().round(2).to_dict())
    print(clean[LABS].describe().round(2).loc[["mean", "50%", "min", "max"]].to_string())