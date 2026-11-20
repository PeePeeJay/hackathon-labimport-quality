"""Interaktiver Bericht der Laborwerte als HTML (Plotly), zwei Panels.

    from plot_interaktiv import plot_overview_html
    plot_overview_html(df, truth, "laborwerte_interaktiv.html", clean=clean, iso=iso)

Panel 1 (univariat), je Laborwert mit gemeinsamer x-Achse:
  1. Original:            Verteilung vor der Manipulation
  2. Nach Manipulation:   Gesamtverteilung aller Werte (unveraendert + manipuliert)
  3. Einzelne Patienten:  eine Zeile je Fehlertyp; Mouseover zeigt Patienten-ID und Wert
  Auswahl "Fehlertyp" blendet einzelne Manipulationen ein und aus.

Panel 2 (bivariat):
  ein Streudiagramm zweier Laborwerte, Paar per Auswahl umschaltbar.
  Blau = unveraendert, Orange = manipuliert (Symbol je Fehlertyp).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, AQUA, INK, MUTED, BG, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b",
                                            "#52514e", "#fcfcfb", "#e8e7e2")
FONT = "Inter, Segoe UI, Arial, sans-serif"

# Feld -> (Anzeigename, Einheit, Achse)
LABS = {
    "albumin_g_dl": ("Albumin", "g/dl", "linear"),
    "thrombozyten_g_l": ("Thrombozyten", "G/l", "linear"),
    "bilirubin_mg_dl": ("Bilirubin", "mg/dl", "log"),
    "afp_ng_ml": ("AFP", "ng/ml", "log"),
    "inr": ("INR", "", "linear"),
}
FEHLERTYPEN = {
    "einheitenfehler": "Einheitenfehler",
    "kommaverschiebung": "Kommaverschiebung",
    "feldtausch": "Feldtausch",
    "unmoegliche_kombination": "Unmögliche Kombination",
    "platzhalterwert": "Platzhalterwert",
    "duplikat": "Duplikat",
}
PLATZHALTER = [999, 0, -1]


# ---------------------------------------------------------------- Hilfsfunktionen
def _fenster(vorher: pd.Series, sicht: pd.Series, log: bool):
    """Achsenbereich: Originalwerte plus Fehler, aber hoechstens eine Spannweite
    (linear) bzw. Faktor 30 (log) darueber hinaus. Der Rest kommt an den Rand."""
    sicht = sicht[sicht > 0] if log else sicht
    v_lo = float(vorher[vorher > 0].min() if log else vorher.min())
    v_hi = float(vorher.max())
    lo, hi = float(sicht.min()), float(sicht.max())
    if log:
        return max(lo, v_lo / 30) / 1.5, min(hi, v_hi * 30) * 1.5
    lo, hi = max(lo, v_lo - (v_hi - v_lo)), min(hi, v_hi + (v_hi - v_lo))
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _bins(werte: pd.Series, lo: float, hi: float, log: bool, n: int = 45):
    """Gemeinsame Klassen fuer beide Histogramme ueber den ganzen Achsenbereich."""
    if log:
        return np.geomspace(lo, hi, n + 1)
    u = np.unique(werte)
    step = np.diff(u).min() if len(u) > 1 else 1.0       # Messaufloesung
    w = step * max(1, int(np.ceil((hi - lo) / step / n)))
    start = np.floor(lo / step) * step - step / 2
    return np.arange(start, hi + w, w)


def _bar(x, bins, color, name, einheit, label, legend=None):
    n, _ = np.histogram(x, bins=bins)
    return go.Bar(
        x=(bins[:-1] + bins[1:]) / 2, y=n, width=np.diff(bins),
        marker=dict(color=color, line=dict(color=BG, width=0.5)),
        customdata=np.c_[bins[:-1], bins[1:]],
        hovertemplate=(f"{name}: %{{customdata[0]:.3g}} bis %{{customdata[1]:.3g}} "
                       f"{einheit}<br>%{{y}} Patienten<extra>{label}</extra>"),
        name=legend or label, showlegend=legend is not None)


def _manipuliert(df, truth, col, id_col):
    """Zeilen, in denen `col` manipuliert wurde (Duplikate: ganze Zeile)."""
    t = truth[truth["feld"].str.contains(col, regex=False)
              | truth["fehlertyp"].eq("duplikat")]
    typ_von = t.set_index(id_col)["fehlertyp"]
    bad = df[id_col].isin(typ_von.index)
    man = df.loc[bad, [id_col, col]].assign(typ=lambda d: d[id_col].map(typ_von))
    return df.loc[~bad, [id_col, col]], man


def _achsen_stil(fig):
    fig.update_xaxes(showgrid=False, linecolor="#c9c8c2", ticks="outside",
                     tickcolor="#c9c8c2", tickfont=dict(color=MUTED, size=11),
                     zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED, size=11))


def _titel(haupt, unter):
    return dict(text=(f"<b>{haupt}</b><br>"
                      f"<span style='font-size:13px;color:{MUTED}'>{unter}</span>"),
                x=0.01, xanchor="left", font=dict(color=INK, size=18))


# ---------------------------------------------------------------- Panel 1
def fig_univariat(df: pd.DataFrame, truth: pd.DataFrame,
                  clean: pd.DataFrame | None = None, id_col: str = "patient_id",
                  seed: int = 0) -> go.Figure:
    """Verteilung vor/nach Manipulation und einzelne Patienten je Fehlertyp."""
    rng = np.random.default_rng(seed)
    labs = [c for c in LABS if c in df.columns]
    typen = [t for t in FEHLERTYPEN if t in set(truth["fehlertyp"])]
    zeilen = ["Unverändert"] + [FEHLERTYPEN[t] for t in typen]
    orig = clean.set_index(id_col) if clean is not None else None
    dup_von = truth.loc[truth["fehlertyp"] == "duplikat"].set_index(id_col)["wert_original"]

    y_lin, y_log = {}, {}       # Umschalter fuer die y-Achse der Histogramme
    spur_typ = []               # Fehlertyp je Spur (None = immer sichtbar)
    fig = make_subplots(
        rows=3, cols=len(labs), shared_xaxes=True,
        row_heights=[0.29, 0.29, 0.42], vertical_spacing=0.04, horizontal_spacing=0.035,
        subplot_titles=[LABS[c][0] + (f" ({LABS[c][1]})" if LABS[c][1] else "")
                        for c in labs])

    def add(trace, typ, row, col):
        fig.add_trace(trace, row=row, col=col)
        spur_typ.append(typ)

    for k, col in enumerate(labs, start=1):
        name, einheit, achse = LABS[col]
        log = achse == "log"
        ok, man = _manipuliert(df, truth, col, id_col)
        vorher = clean[col] if clean is not None else ok[col]

        ist_ph = man["typ"].eq("platzhalterwert") & man[col].isin(PLATZHALTER)
        lo, hi = _fenster(vorher, pd.concat([vorher, ok[col], man.loc[~ist_ph, col]]), log)
        bins = _bins(vorher, lo, hi, log)
        lo, hi = float(bins[0]), float(bins[-1])

        # Werte ausserhalb der Achse in die Randklasse legen
        wert = man[col].to_numpy(dtype=float)
        x_man = wert.copy()
        links, rechts = wert <= lo, wert >= hi
        x_man[links], x_man[rechts] = (bins[0] + bins[1]) / 2, (bins[-2] + bins[-1]) / 2

        # --- Zeile 1: Original ---
        add(_bar(vorher, bins, BLUE, name, einheit, "unverändert",
                 "unverändert" if k == 1 else None), None, 1, k)
        # --- Zeile 2: Gesamtverteilung nach Manipulation (gestapelt) ---
        add(_bar(ok[col], bins, BLUE, name, einheit, "unverändert"), None, 2, k)
        # --- Zeile 3: unveraenderte Patienten ---
        add(go.Scattergl(
            x=ok[col], y=rng.uniform(-0.38, 0.38, len(ok)), mode="markers",
            marker=dict(color=BLUE, size=4, opacity=0.3),
            customdata=ok[[id_col]].to_numpy(),
            hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{name}: %{{x}} {einheit}"
                           "<extra>unverändert</extra>"),
            showlegend=False), None, 3, k)

        # --- manipulierte Werte, eine Spur je Fehlertyp ---
        for i, typ in enumerate(typen, start=1):
            m = (man["typ"] == typ).to_numpy()
            if not m.any():
                continue
            add(_bar(x_man[m], bins, ORANGE, name, einheit, FEHLERTYPEN[typ]), typ, 2, k)
            info = []
            for pid, w, aussen in zip(man.loc[m, id_col], wert[m], (links | rechts)[m]):
                z = f"{name}: <b>{w:g} {einheit}</b>"
                if typ == "duplikat":
                    z += f"<br>Kopie von {dup_von.get(pid, '?')}"
                elif orig is not None and pid in orig.index:
                    z += f"<br>Original: {orig.at[pid, col]:g} {einheit}"
                if aussen:
                    z += "<br>(außerhalb der Achse, am Rand gezeichnet)"
                info.append(z)
            add(go.Scatter(
                x=x_man[m], y=i + rng.uniform(-0.3, 0.3, m.sum()), mode="markers",
                marker=dict(color=ORANGE, size=8,
                            line=dict(color=BG, width=1.2)),
                customdata=np.c_[man.loc[m, id_col], info],
                hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}"
                               f"<extra>{FEHLERTYPEN[typ]}</extra>"),
                showlegend=False), typ, 3, k)

        rng_x = [np.log10(lo), np.log10(hi)] if log else [lo, hi]
        for r in (1, 2, 3):
            fig.update_xaxes(type=achse, range=rng_x, dtick=1 if log else None,
                             row=r, col=k)
        # Zeile 1 und 2 mit gleicher y-Achse, damit sie direkt vergleichbar sind
        y_max = 1.08 * max(np.histogram(vorher, bins=bins)[0].max(),
                           np.histogram(np.r_[ok[col], x_man], bins=bins)[0].max())
        for r in (1, 2):
            fig.update_yaxes(range=[0, y_max], row=r, col=k)
            ax = fig.get_subplot(r, k).yaxis.plotly_name
            y_lin.update({f"{ax}.type": "linear", f"{ax}.range": [0, y_max],
                          f"{ax}.dtick": None})
            y_log.update({f"{ax}.type": "log", f"{ax}.dtick": 1,
                          f"{ax}.range": [np.log10(0.5), np.log10(1.5 * y_max)]})

    # Legendeneintrag, der beim Filtern sichtbar bleibt
    add(go.Scatter(x=[None], y=[None], mode="markers", name="manipuliert",
                   marker=dict(color=ORANGE, size=10, symbol="square")), None, 2, 1)

    _achsen_stil(fig)
    fig.update_yaxes(tickvals=list(range(len(zeilen))), ticktext=zeilen,
                     range=[len(zeilen) - 0.5, -0.5], showgrid=False,
                     showticklabels=False, fixedrange=True, row=3)
    fig.update_yaxes(showticklabels=True, tickfont=dict(color=INK, size=12), row=3, col=1)
    for i in range(len(zeilen) - 1):
        fig.add_hline(y=i + 0.5, line=dict(color=GRID, width=1), row=3, col="all")
    fig.update_yaxes(title_text="Original<br>(Anzahl Patienten)", row=1, col=1,
                     title_font=dict(color=INK, size=12))
    fig.update_yaxes(title_text="Nach Manipulation<br>(Anzahl Patienten)", row=2, col=1,
                     title_font=dict(color=INK, size=12))
    fig.update_annotations(font=dict(color=INK, size=14))

    def sichtbar(auswahl):
        return [t is None or auswahl is None or t == auswahl for t in spur_typ]

    n_typ = truth["fehlertyp"].value_counts()
    filter_buttons = [dict(label=f"Alle Fehlertypen ({len(truth)})", method="update",
                           args=[{"visible": sichtbar(None)}])]
    filter_buttons += [dict(label=f"{FEHLERTYPEN[t]} ({n_typ[t]})", method="update",
                            args=[{"visible": sichtbar(t)}]) for t in typen]
    knopf = dict(bgcolor="white", bordercolor="#c9c8c2", font=dict(color=INK, size=12))
    fig.update_layout(
        title=_titel("Panel 1 · Laborwerte vor und nach der Manipulation",
                     f"n = {len(df)}, davon {truth[id_col].nunique()} Zeilen mit "
                     "injiziertem Fehler · unten: jeder Punkt ist ein Patient, "
                     "Mouseover zeigt ID und Wert"),
        barmode="stack", bargap=0, paper_bgcolor=BG, plot_bgcolor=BG, height=900,
        margin=dict(l=70, r=20, t=125, b=70), font=dict(family=FONT),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06,
                    font=dict(color=INK, size=12)),
        updatemenus=[
            dict(type="dropdown", direction="down", x=0.78, xanchor="right", y=1.065,
                 yanchor="bottom", showactive=True, buttons=filter_buttons, **knopf),
            dict(type="buttons", direction="right", x=1.0, xanchor="right", y=1.065,
                 yanchor="bottom", showactive=True, **knopf,
                 buttons=[dict(label="y-Achse linear", method="relayout", args=[y_lin]),
                          dict(label="y-Achse logarithmisch", method="relayout",
                               args=[y_log])])],
        hoverlabel=dict(bgcolor="white", font=dict(color=INK, size=13)))
    return fig


# ---------------------------------------------------------------- Panel 2
SYMBOLE = {"einheitenfehler": "circle", "kommaverschiebung": "square",
           "feldtausch": "diamond", "unmoegliche_kombination": "triangle-up",
           "platzhalterwert": "x", "duplikat": "cross"}


def fig_bivariat(df: pd.DataFrame, truth: pd.DataFrame,
                 clean: pd.DataFrame | None = None, id_col: str = "patient_id",
                 start: tuple[str, str] = ("albumin_g_dl", "bilirubin_mg_dl")) -> go.Figure:
    """Ein Streudiagramm zweier Laborwerte; das Paar ist per Auswahl umschaltbar.

    Blau = unveraendert, Orange = manipuliert (Symbol je Fehlertyp).
    Klick in die Legende blendet einzelne Fehlertypen aus.
    """
    labs = [c for c in LABS if c in df.columns]
    paare = list(combinations(labs, 2))
    if start in paare:
        paare.remove(start)
        paare.insert(0, start)
    typ_von = truth.drop_duplicates(id_col).set_index(id_col)["fehlertyp"]
    typ = df[id_col].map(typ_von)
    typen = [t for t in FEHLERTYPEN if t in set(typ.dropna())]
    masken = [typ.isna().to_numpy()] + [(typ == t).to_numpy() for t in typen]

    # Darstellungsbereich je Laborwert, Werte ausserhalb an den Rand
    x_plot, bereich = {}, {}
    for col in labs:
        log = LABS[col][2] == "log"
        vorher = clean[col] if clean is not None else df.loc[typ.isna(), col]
        ph = df[col].isin(PLATZHALTER) & typ.eq("platzhalterwert")
        lo, hi = _fenster(vorher, df.loc[~ph, col], log)
        rand = (lo * 1.15, hi / 1.15) if log else (lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo))
        x_plot[col] = df[col].astype(float).clip(*rand).to_numpy()
        bereich[col] = [np.log10(lo), np.log10(hi)] if log else [lo, hi]

    def titel(col):
        name, einheit, _ = LABS[col]
        return name + (f" ({einheit})" if einheit else "")

    def daten(cx, cy):
        """x, y, customdata und Mouseover je Spur fuer ein Paar."""
        (nx, ex, _), (ny, ey, _) = LABS[cx], LABS[cy]
        hover = (f"<b>%{{customdata[0]}}</b><br>{nx}: %{{customdata[1]}} {ex}"
                 f"<br>{ny}: %{{customdata[2]}} {ey}")
        return dict(
            x=[x_plot[cx][m] for m in masken], y=[x_plot[cy][m] for m in masken],
            customdata=[np.c_[df.loc[m, id_col], df.loc[m, cx], df.loc[m, cy]]
                        for m in masken],
            hovertemplate=[hover + f"<extra>{n}</extra>"
                           for n in ["unverändert"] + [FEHLERTYPEN[t] for t in typen]])

    def achsen(cx, cy):
        out = {}
        for ax, col in (("xaxis", cx), ("yaxis", cy)):
            log = LABS[col][2] == "log"
            out.update({f"{ax}.type": LABS[col][2], f"{ax}.range": bereich[col],
                        f"{ax}.title.text": titel(col), f"{ax}.dtick": 1 if log else None})
        return out

    d0 = daten(*paare[0])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=d0["x"][0], y=d0["y"][0], customdata=d0["customdata"][0],
        hovertemplate=d0["hovertemplate"][0], mode="markers", name="unverändert",
        marker=dict(color=BLUE, size=6, opacity=0.35)))
    for i, t in enumerate(typen, start=1):
        fig.add_trace(go.Scatter(
            x=d0["x"][i], y=d0["y"][i], customdata=d0["customdata"][i],
            hovertemplate=d0["hovertemplate"][i], mode="markers",
            name=f"{FEHLERTYPEN[t]} ({int(masken[i].sum())})",
            marker=dict(color=ORANGE, size=10, symbol=SYMBOLE.get(t, "circle"),
                        line=dict(color=BG, width=1))))

    buttons = [dict(label=f"{LABS[cx][0]} × {LABS[cy][0]}", method="update",
                    args=[daten(cx, cy), achsen(cx, cy)]) for cx, cy in paare]
    _achsen_stil(fig)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title_font=dict(color=INK, size=13))
    fig.update_yaxes(title_font=dict(color=INK, size=13))
    fig.update_layout(
        title=_titel("Panel 2 · Zwei Laborwerte gegeneinander",
                     "Orange = manipuliert, Symbol je Fehlertyp · Klick in die "
                     "Legende blendet Fehlertypen aus"),
        paper_bgcolor=BG, plot_bgcolor=BG, height=720,
        margin=dict(l=70, r=20, t=125, b=60), font=dict(family=FONT),
        legend=dict(x=1.01, y=1, xanchor="left", font=dict(color=INK, size=12),
                    itemsizing="constant"),
        updatemenus=[dict(type="dropdown", direction="down", x=1.0, xanchor="right",
                          y=1.05, yanchor="bottom", showactive=True, buttons=buttons,
                          bgcolor="white", bordercolor="#c9c8c2",
                          font=dict(color=INK, size=12))],
        hoverlabel=dict(bgcolor="white", font=dict(color=INK, size=13)))
    fig.update_layout(achsen(*paare[0]))
    return fig


# ---------------------------------------------------------------- HTML
def plot_overview_html(df: pd.DataFrame, truth: pd.DataFrame, path=None,
                       clean: pd.DataFrame | None = None, iso: pd.DataFrame | None = None,
                       id_col: str = "patient_id", include_plotlyjs=True):
    """Schreibt den Bericht nach `path`. Gibt die Figuren als Liste zurueck.

    clean: fehlerfreier Datensatz (Zeile "Original" und Originalwerte im Mouseover).
    iso:   wird nicht mehr gezeichnet; der Parameter bleibt nur, damit alte Aufrufe laufen.
    include_plotlyjs: True = HTML laeuft offline (ca. 4,5 MB), "cdn" = kleine Datei.
    """
    figs = [fig_univariat(df, truth, clean, id_col),
            fig_bivariat(df, truth, clean, id_col)]
    if path is not None:
        cfg = {"displaylogo": False}
        teile = [f.to_html(full_html=False, config=cfg,
                           include_plotlyjs=include_plotlyjs if i == 0 else False)
                 for i, f in enumerate(figs)]
        html = ("<!DOCTYPE html>\n<html lang='de'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>OOD-Monitoring: Laborwerte</title>"
                f"<style>body{{margin:0;background:{BG};font-family:{FONT}}}"
                "section{max-width:1900px;margin:0 auto;padding:8px 12px}"
                "section:nth-of-type(2){max-width:1150px}"
                f"section+section{{border-top:1px solid {GRID};margin-top:12px}}</style>"
                "</head><body>"
                + "".join(f"<section>{t}</section>" for t in teile) + "</body></html>")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return figs


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--daten", type=Path, default=Path("daten"))
    ap.add_argument("--out", type=Path, default=Path("folien/laborwerte_interaktiv.html"))
    a = ap.parse_args()
    df = pd.read_csv(a.daten / "hcc_mit_fehlern.csv")
    plot_overview_html(df, pd.read_csv(a.daten / "hcc_ground_truth.csv"), a.out,
                       clean=pd.read_csv(a.daten / "hcc_clean.csv"))
    print("geschrieben:", a.out)