"""Univariate Ausreisserpruefung fuer eCRF-Daten (PoC).

Zwei Aufrufe reichen:

    from ood_univariat import check_univariate, plot_univariate
    flags = check_univariate(df)                 # Tabelle der auffaelligen Werte
    plot_univariate(df, flags, "univariat.png")  # Dichten + markierte Werte

Methode: robuster z-Score (Median/MAD, Iglewicz & Hoaglin), mit getrennter
Streuung unterhalb und oberhalb des Medians ("Double MAD"), damit schiefe
Verteilungen nicht einseitig Fehlalarme erzeugen. Schiefe Variablen werden
automatisch auf der Log-Skala geprueft. Keine Annahmen aus der Literatur.

Demo:  python ood_univariat.py --data hcc_mit_fehlern.csv --truth hcc_ground_truth.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

BLUE, ORANGE, INK, MUTED, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def _columns(df, columns):
    if columns is not None:
        return list(columns)
    num = df.select_dtypes("number")
    return [c for c in num if num[c].nunique() > 10]   # stetige Variablen


def _fit(x: pd.Series, scale: str | None):
    """Skala waehlen; Median und MAD (unten/oben getrennt) auf dieser Skala."""
    pos = x[x > 0]
    if scale is None:
        # Schiefe aus Quantilen: unempfindlich gegen einzelne Extremwerte
        q1, q2, q3 = np.quantile(x.dropna(), [0.125, 0.5, 0.875])
        schief = (q3 + q1 - 2 * q2) / (q3 - q1) if q3 > q1 else 0
        scale = "log" if schief > 0.2 and len(pos) > 10 else "linear"
    v = np.log(pos) if scale == "log" else x.dropna()
    med = float(np.median(v))
    mad_lo = float(np.median(med - v[v <= med]))
    mad_hi = float(np.median(v[v >= med] - med))
    return scale, med, mad_lo, mad_hi


def check_univariate(df: pd.DataFrame, columns=None, threshold: float = 3.5,
                     scales: dict[str, str] | None = None,
                     id_col: str = "patient_id",
                     nonnegative: bool = True) -> pd.DataFrame:
    """Markiert Werte mit |robuster z-Score| > threshold.

    nonnegative: negative Werte gelten immer als ungueltig (Laborwerte).

    scales: optional {"spalte": "log" | "linear"}; sonst automatisch nach Schiefe.
    Rueckgabe: eine Zeile je auffaelligem Wert (id, feld, wert, score, grund).
    Die verwendeten Parameter stehen in flags.attrs["fit"].
    """
    scales, out, fit = scales or {}, [], {}
    for col in _columns(df, columns):
        x = df[col]
        scale, med, mad_lo, mad_hi = _fit(x, scales.get(col))
        fit[col] = {"scale": scale, "median": med, "mad_lo": mad_lo,
                    "mad_hi": mad_hi, "threshold": threshold}
        if min(mad_lo, mad_hi) == 0:
            continue
        if scale == "log":
            invalid = x <= 0
            v = np.log(x.where(~invalid))
        else:
            invalid, v = (x < 0) & nonnegative, x
        z = 0.6745 * (v - med) / np.where(v < med, mad_lo, mad_hi)
        z = pd.Series(z, index=x.index)
        hit = (z.abs() > threshold) | invalid
        for i in df.index[hit]:
            out.append({id_col: df.at[i, id_col], "feld": col, "wert": x[i],
                        "score": np.inf if invalid[i] else round(float(z[i]), 1),
                        "grund": "Wert nicht positiv" if invalid[i]
                        else ("zu hoch" if z[i] > 0 else "zu niedrig")})
    flags = pd.DataFrame(out, columns=[id_col, "feld", "wert", "score", "grund"])
    flags.attrs["fit"] = fit
    return flags


def plot_univariate(df: pd.DataFrame, flags: pd.DataFrame, path=None,
                    columns=None, id_col: str = "patient_id", ncols: int = 3):
    """Dichte je Variable, Akzeptanzbereich und markierte Werte."""
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = flags.attrs["fit"]
    cols = [c for c in _columns(df, columns) if c in fit and min(fit[c]["mad_lo"], fit[c]["mad_hi"]) > 0]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows),
                             facecolor=BG, squeeze=False)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    for ax, col in zip(axes.flat, cols):
        f = fit[col]
        log = f["scale"] == "log"
        h_lo = f["threshold"] * f["mad_lo"] / 0.6745
        h_hi = f["threshold"] * f["mad_hi"] / 0.6745
        lo, hi = f["median"] - h_lo, f["median"] + h_hi        # Akzeptanzbereich
        v_lo, v_hi = lo - 1.3 * h_lo, hi + 1.3 * h_hi           # Sichtfenster
        tr = (lambda s: np.log(s.where(s > 0))) if log else (lambda s: s)
        back = np.exp if log else (lambda a: a)

        bad_ids = flags.loc[flags["feld"] == col, id_col]
        bad = df[id_col].isin(bad_ids)
        v_ok = tr(df.loc[~bad, col]).dropna()
        v_bad = tr(df.loc[bad, col])
        inside = v_bad.between(v_lo, v_hi)

        grid = np.linspace(v_lo, v_hi, 400)
        dens = gaussian_kde(v_ok)(grid)
        ax.fill_between(back(grid), dens, color=BLUE, alpha=0.25, lw=0)
        ax.plot(back(grid), dens, color=BLUE, lw=2, label="Dichte (unauffällig)")
        ax.axvspan(back(v_lo), back(lo), color=ORANGE, alpha=0.07, lw=0)
        ax.axvspan(back(hi), back(v_hi), color=ORANGE, alpha=0.07, lw=0)
        for b in (lo, hi):
            ax.axvline(back(b), color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.plot(back(v_bad[inside]), np.zeros(inside.sum()), "o", color=ORANGE, ms=8,
                mec=BG, mew=2, clip_on=False, zorder=5, label="markierter Wert")
        n_out = int((~inside).sum())
        txt = f"{int(bad.sum())} markiert"
        if n_out:
            txt += f"\ndavon {n_out} außerhalb\ndes Bildausschnitts"
        ax.text(0.98, 0.96, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=MUTED)
        if log:
            ax.set_xscale("log")
            if v_hi - v_lo < np.log(100):      # schmaler Bereich: normale Zahlen
                from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
                ax.xaxis.set_major_locator(LogLocator(subs=[1, 1.5, 2, 3, 5, 7]))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(back(v_lo), back(v_hi))
        ax.set_ylim(bottom=0)
        ax.set_title(f"{col}  ({'Log-Skala' if log else 'linear'})", fontsize=11,
                     color=INK, loc="left")
        ax.set_yticks([])
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    h, l = axes.flat[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3))))
    l.append(f"Grenze: |robuster z-Score| = {fit[cols[0]]['threshold']}")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10,
               labelcolor=INK)
    fig.suptitle("Univariate Prüfung: Dichte je Variable und markierte Werte",
                 x=0.01, ha="left", fontsize=14, color=INK, weight="bold")
    fig.subplots_adjust(left=0.03, right=0.98, top=1 - 0.95 / (3.4 * nrows),
                        bottom=0.75 / (3.4 * nrows), wspace=0.12, hspace=0.45)
    if path is not None:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig


def evaluate(df, flags, truth, id_col: str = "patient_id") -> pd.DataFrame:
    """Erkennungsrate je Fehlertyp (zeilenweise) + Fehlalarme."""
    flagged = set(flags[id_col])
    t = truth.assign(erkannt=truth[id_col].isin(flagged))
    res = t.groupby("fehlertyp")["erkannt"].agg(injiziert="size", erkannt="sum")
    res["rate"] = (res["erkannt"] / res["injiziert"]).round(2)
    res.attrs["fehlalarme"] = len(flagged - set(truth[id_col]))
    res.attrs["zeilen_ohne_fehler"] = len(df) - len(truth)
    return res


def plot_evaluation(res: pd.DataFrame, path, titel: str = "Univariate Prüfung") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = res.sort_values("rate")
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    ax.barh(r.index.str.replace("_", " "), r["rate"] * 100, color=BLUE, height=0.55)
    for y, (e, n, rate) in enumerate(zip(r["erkannt"], r["injiziert"], r["rate"])):
        ax.text(rate * 100 + 1.5, y, f"{e} von {n}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("erkannte Fehler (%)", color=MUTED, fontsize=10)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.grid(axis="x", color="#e8e7e2", lw=0.6)
    ax.set_axisbelow(True)
    fa, n0 = res.attrs["fehlalarme"], res.attrs["zeilen_ohne_fehler"]
    ax.set_title(f"{titel}: Erkennungsrate je Fehlertyp", loc="left", fontsize=13,
                 color=INK, weight="bold", pad=22)
    ax.text(0, 1.03, f"Fehlalarme: {fa} von {n0} fehlerfreien Zeilen "
            f"({fa / n0:.1%})", transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("hcc_mit_fehlern.csv"))
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("../../../../ood-poc"))
    a = ap.parse_args()

    labs = ["albumin_g_dl", "thrombozyten_g_l", "bilirubin_mg_dl", "afp_ng_ml", "inr"]
    df = pd.read_csv(a.data)
    flags = check_univariate(df, columns=labs)
    flags.to_csv(a.out / "univariat_flags.csv", index=False)
    plot_univariate(df, flags, a.out / "folie_univariat_dichten.png", columns=labs)
    print(pd.DataFrame(flags.attrs["fit"]).T.round(3).to_string())
    print(f"\n{len(flags)} markierte Werte in {flags['patient_id'].nunique()} Zeilen")
    if a.truth:
        res = evaluate(df, flags, pd.read_csv(a.truth))
        plot_evaluation(res, a.out / "folie_univariat_erkennung.png")
        print(res.to_string())
        print("Fehlalarme:", res.attrs["fehlalarme"], "von", res.attrs["zeilen_ohne_fehler"])
