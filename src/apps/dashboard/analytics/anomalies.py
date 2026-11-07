"""Bibliometric anomaly detection (Isolation Forest over per-article indicators)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_bibliometric_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """Identify multidimensional bibliometric outliers using Isolation Forest.

    Audits publication records based on citation velocity, author team size,
    relevance score, and full-text metadata.
    """
    from sklearn.ensemble import IsolationForest

    if len(df) < 20:
        res = df.copy()
        res["anomaly_score"] = 0.0
        res["is_anomaly"] = False
        res["anomaly_reason"] = "—"
        return res

    res = df.copy()
    years = pd.to_numeric(res.get("year"), errors="coerce").fillna(2015).to_numpy()
    cites = (
        pd.to_numeric(res.get("citation_count"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
        .to_numpy()
    )

    if "reference_count" in res.columns:
        refs = pd.to_numeric(res["reference_count"], errors="coerce").fillna(0.0).to_numpy()
    else:
        refs = np.zeros(len(res))

    if "authors" in res.columns:
        author_counts = (
            res["authors"].apply(lambda a: len(a) if isinstance(a, list) else 1).to_numpy()
        )
    else:
        author_counts = np.ones(len(res))

    if "relevance_score" in res.columns:
        rel = pd.to_numeric(res["relevance_score"], errors="coerce").fillna(0.5).to_numpy()
    else:
        rel = np.full(len(res), 0.5)

    if "has_pdf" in res.columns:
        has_pdf = res["has_pdf"].fillna(False).astype(int).to_numpy()
    else:
        has_pdf = np.zeros(len(res))

    X = np.column_stack([years, cites, refs, author_counts, rel, has_pdf])

    try:
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42).fit(
            X
        )
        preds = model.predict(X)
        scores = -model.decision_function(X)

        res["anomaly_score"] = scores.round(3)
        res["is_anomaly"] = preds == -1

        # Derive explanatory rationales
        reasons = []
        cite_p95 = np.percentile(cites, 95)
        author_p98 = np.percentile(author_counts, 98)

        for i in range(len(res)):
            if preds[i] != -1:
                reasons.append("Typical / Consistent Pattern")
                continue
            r_list = []
            if cites[i] >= cite_p95 and years[i] >= 2018:
                r_list.append(f"Recently hyper-cited ({int(cites[i])} citations)")
            elif cites[i] >= cite_p95:
                r_list.append(f"Extreme citation count ({int(cites[i])} citations)")
            if author_counts[i] >= author_p98 and author_counts[i] >= 10:
                r_list.append(f"Mega-team ({int(author_counts[i])} authors)")
            if years[i] < 1990:
                r_list.append(f"Historical article ({int(years[i])})")
            if rel[i] < 0.25:
                r_list.append(f"Divergent semantic margin ({rel[i]:.2f})")
            if not r_list:
                r_list.append("Joint multidimensional atypicality")
            reasons.append("; ".join(r_list))

        res["anomaly_reason"] = reasons
    except Exception:
        res["anomaly_score"] = 0.0
        res["is_anomaly"] = False
        res["anomaly_reason"] = "Model fit error"

    return res
