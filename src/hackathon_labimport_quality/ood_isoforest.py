"""Multivariate Ausreisserpruefung mit Isolation Forest (PoC).

    from ood_isoforest import check_isolation_forest
    iso = check_isolation_forest(df)      # Score und Markierung je Zeile
    auffaellig = iso[iso["markiert"]]

Der Isolation Forest sieht alle Laborwerte gleichzeitig und findet damit auch
Kombinationen, die einzeln betrachtet unauffaellig sind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

LABS = ["albumin_g_dl", "thrombozyten_g_l", "bilirubin_mg_dl", "afp_ng_ml", "inr"]
ORDINAL = {"child_pugh": {"A": 0, "B": 1, "C": 2}}


def _features(df: pd.DataFrame, columns, ordinal) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in columns:
        x = df[col].astype(float)
        q1, q2, q3 = np.quantile(x.dropna(), [0.125, 0.5, 0.875])
        schief = (q3 + q1 - 2 * q2) / (q3 - q1) if q3 > q1 else 0
        if schief > 0.2:                       # schiefe Variable: Log-Skala
            boden = x[x > 0].min() / 10        # Werte <= 0 weit nach unten legen
            X[f"log_{col}"] = np.log(x.clip(lower=boden))
        else:
            X[col] = x
    for col, mapping in ordinal.items():
        if col in df.columns:
            X[col] = df[col].map(mapping).astype(float)
    return X.fillna(X.median())


def check_isolation_forest(df: pd.DataFrame, columns=None, ordinal=None,
                           contamination: float | str = 0.1, n_estimators: int = 300,
                           seed: int = 42, id_col: str = "patient_id") -> pd.DataFrame:
    """Isolation Forest ueber Laborwerte (+ Child-Pugh, falls vorhanden).

    contamination: erwarteter Anteil auffaelliger Zeilen (Annahme!) oder "auto".
    Rueckgabe: eine Zeile je Patient mit `score` (hoeher = auffaelliger) und `markiert`.
    """
    columns = [c for c in (columns or LABS) if c in df.columns]
    X = _features(df, columns, ORDINAL if ordinal is None else ordinal)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            random_state=seed).fit(X)
    out = pd.DataFrame({id_col: df[id_col].to_numpy(),
                        "score": np.round(-model.score_samples(X), 4),
                        "markiert": model.predict(X) == -1})
    out.attrs["features"] = list(X.columns)
    out.attrs["contamination"] = contamination
    return out