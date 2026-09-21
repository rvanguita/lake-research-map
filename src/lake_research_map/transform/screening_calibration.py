"""Human-label calibration utilities for systematic-review screening.

The module is deliberately Streamlit-free. It validates long-form reviewer
decisions, resolves adjudication/consensus, measures agreement and evaluates a
contrastive-margin threshold without reusing its calibration observations for
the reported holdout metrics.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from lake_research_map.transform.bronze_articles import normalize_doi

STRATA = (
    "out_of_scope_deep",
    "borderline_negative",
    "borderline_positive",
    "in_scope_deep",
)

LABEL_ALIASES = {
    "1": "include",
    "1.0": "include",
    "true": "include",
    "include": "include",
    "included": "include",
    "incluir": "include",
    "0": "exclude",
    "0.0": "exclude",
    "false": "exclude",
    "exclude": "exclude",
    "excluded": "exclude",
    "excluir": "exclude",
    "uncertain": "uncertain",
    "unsure": "uncertain",
    "duvida": "uncertain",
    "dúvida": "uncertain",
    "": "uncertain",
}


def _normalize_doi_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    return normalize_doi(str(value))


def _empty_sample() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "doi",
            "title",
            "year",
            "venue",
            "relevance_margin",
            "theme_label",
            "stratum",
            "reviewer",
            "manual_label",
            "reviewer_notes",
            "protocol_version",
        ]
    )


def generate_stratified_screening_sample(
    scored_df: pd.DataFrame,
    n_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample the full margin range and redistribute unused stratum quotas."""
    required = {"relevance_margin", "doi"}
    if scored_df.empty or not required.issubset(scored_df.columns) or n_samples <= 0:
        return _empty_sample()

    valid = scored_df.dropna(subset=["relevance_margin", "doi"]).copy()
    valid["doi"] = valid["doi"].map(_normalize_doi_value)
    valid = valid.dropna(subset=["doi"]).drop_duplicates("doi")
    if valid.empty:
        return _empty_sample()

    margin = pd.to_numeric(valid["relevance_margin"], errors="coerce")
    valid = valid.loc[margin.notna()].copy()
    margin = margin.loc[valid.index]
    masks = {
        "out_of_scope_deep": margin < -0.10,
        "borderline_negative": (margin >= -0.10) & (margin < 0.0),
        "borderline_positive": (margin >= 0.0) & (margin <= 0.10),
        "in_scope_deep": margin > 0.10,
    }

    target = min(int(n_samples), len(valid))
    base, remainder = divmod(target, len(STRATA))
    allocations = {
        name: min(int(masks[name].sum()), base + (index < remainder))
        for index, name in enumerate(STRATA)
    }
    remaining = target - sum(allocations.values())
    while remaining:
        progressed = False
        for name in STRATA:
            capacity = int(masks[name].sum()) - allocations[name]
            if capacity > 0:
                allocations[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    rng = np.random.default_rng(seed)
    sampled: list[pd.DataFrame] = []
    for name in STRATA:
        subset = valid.loc[masks[name]]
        n_take = allocations[name]
        if n_take:
            chosen = rng.choice(subset.index.to_numpy(), size=n_take, replace=False)
            part = subset.loc[chosen].copy()
            part["stratum"] = name
            sampled.append(part)

    if not sampled:
        return _empty_sample()
    result = pd.concat(sampled, ignore_index=True)
    result["reviewer"] = ""
    result["manual_label"] = ""
    result["reviewer_notes"] = ""
    result["protocol_version"] = "v1"
    for column in _empty_sample().columns:
        if column not in result.columns:
            result[column] = pd.NA
    return (
        result[list(_empty_sample().columns)].sort_values("relevance_margin").reset_index(drop=True)
    )


def validate_review_labels(
    labels: pd.DataFrame,
    *,
    known_dois: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize a long-form review file and return row-level audit issues."""
    issue_columns = ["severity", "code", "doi", "detail"]
    if labels.empty:
        issue = pd.DataFrame(
            [["error", "empty_file", None, "O arquivo não contém decisões."]],
            columns=issue_columns,
        )
        return pd.DataFrame(), issue
    missing = {"doi", "manual_label"}.difference(labels.columns)
    if missing:
        issue = pd.DataFrame(
            [["error", "missing_columns", None, ", ".join(sorted(missing))]],
            columns=issue_columns,
        )
        return pd.DataFrame(), issue

    normalized = labels.copy()
    normalized["doi"] = normalized["doi"].map(_normalize_doi_value)
    normalized["reviewer"] = (
        normalized.get("reviewer", pd.Series("reviewer_1", index=normalized.index))
        .fillna("reviewer_1")
        .astype(str)
        .str.strip()
        .replace("", "reviewer_1")
    )
    normalized["protocol_version"] = (
        normalized.get("protocol_version", pd.Series("unversioned", index=normalized.index))
        .fillna("unversioned")
        .astype(str)
        .str.strip()
        .replace("", "unversioned")
    )
    raw_labels = normalized["manual_label"].fillna("").astype(str).str.strip().str.lower()
    normalized["manual_label"] = raw_labels.map(LABEL_ALIASES)

    issues: list[dict[str, object]] = []
    invalid_doi = normalized["doi"].isna()
    for index in normalized.index[invalid_doi]:
        issues.append(
            {
                "severity": "error",
                "code": "invalid_doi",
                "doi": None,
                "detail": f"linha {index + 2}",
            }
        )
    invalid_label = normalized["manual_label"].isna()
    for index in normalized.index[invalid_label]:
        issues.append(
            {
                "severity": "error",
                "code": "invalid_label",
                "doi": normalized.at[index, "doi"],
                "detail": raw_labels.at[index],
            }
        )

    normalized = normalized.loc[~invalid_doi & ~invalid_label].copy()
    if known_dois is not None:
        known = {doi for value in known_dois if (doi := _normalize_doi_value(value))}
        unknown = ~normalized["doi"].isin(known)
        for doi in normalized.loc[unknown, "doi"].drop_duplicates():
            issues.append(
                {
                    "severity": "warning",
                    "code": "unknown_doi",
                    "doi": doi,
                    "detail": "fora do corpus atual",
                }
            )
        normalized = normalized.loc[~unknown].copy()

    conflicts = (
        normalized.groupby(["doi", "reviewer"])["manual_label"].nunique().loc[lambda x: x > 1]
    )
    for doi, reviewer in conflicts.index:
        issues.append(
            {
                "severity": "error",
                "code": "reviewer_conflict",
                "doi": doi,
                "detail": reviewer,
            }
        )
    if len(conflicts):
        conflict_index = pd.MultiIndex.from_tuples(conflicts.index)
        row_index = pd.MultiIndex.from_frame(normalized[["doi", "reviewer"]])
        normalized = normalized.loc[~row_index.isin(conflict_index)].copy()

    normalized = normalized.drop_duplicates(["doi", "reviewer", "manual_label"])
    issue_df = pd.DataFrame(issues, columns=issue_columns)
    return normalized.reset_index(drop=True), issue_df


def resolve_review_consensus(labels: pd.DataFrame) -> pd.DataFrame:
    """Resolve one binary label per DOI using adjudication before consensus."""
    columns = ["doi", "manual_label", "y_true", "resolution", "reviewer_count", "resolved"]
    if labels.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for doi, group in labels.groupby("doi", sort=True):
        binary = group[group["manual_label"].isin(["include", "exclude"])]
        adjudicated = binary[binary["reviewer"].str.casefold().eq("adjudicated")]
        source = adjudicated if not adjudicated.empty else binary
        decisions = source["manual_label"].unique().tolist()
        if len(decisions) == 1:
            label = decisions[0]
            if not adjudicated.empty:
                resolution = "adjudicated"
            elif binary["reviewer"].nunique() == 1:
                resolution = "single_reviewer"
            else:
                resolution = "consensus"
            resolved = True
        else:
            label = "uncertain" if not decisions else "disagreement"
            resolution = label
            resolved = False
        rows.append(
            {
                "doi": doi,
                "manual_label": label,
                "y_true": label == "include" if resolved else pd.NA,
                "resolution": resolution,
                "reviewer_count": int(binary["reviewer"].nunique()),
                "resolved": resolved,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def reviewer_agreement(labels: pd.DataFrame, *, min_overlap: int = 20) -> pd.DataFrame:
    """Return pairwise raw agreement and Cohen's kappa with support checks."""
    columns = ["reviewer_a", "reviewer_b", "n_overlap", "raw_agreement", "kappa", "status"]
    if labels.empty:
        return pd.DataFrame(columns=columns)
    binary = labels[
        labels["manual_label"].isin(["include", "exclude"])
        & ~labels["reviewer"].str.casefold().eq("adjudicated")
    ].copy()
    pivot = binary.pivot_table(
        index="doi", columns="reviewer", values="manual_label", aggfunc="first"
    )
    rows = []
    for reviewer_a, reviewer_b in combinations(pivot.columns, 2):
        pair = pivot[[reviewer_a, reviewer_b]].dropna()
        n_overlap = len(pair)
        raw = float((pair[reviewer_a] == pair[reviewer_b]).mean()) if n_overlap else np.nan
        has_two_classes = pair[reviewer_a].nunique() == 2 and pair[reviewer_b].nunique() == 2
        sufficient = n_overlap >= min_overlap and has_two_classes
        kappa = (
            float(cohen_kappa_score(pair[reviewer_a], pair[reviewer_b])) if sufficient else np.nan
        )
        rows.append(
            {
                "reviewer_a": reviewer_a,
                "reviewer_b": reviewer_b,
                "n_overlap": n_overlap,
                "raw_agreement": raw,
                "kappa": kappa,
                "status": "ok" if sufficient else "insufficient_support",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def evaluate_screening_threshold(
    y_true: np.ndarray | pd.Series | list[bool | int],
    margins: np.ndarray | pd.Series | list[float],
    threshold: float = 0.0,
) -> dict[str, float | int]:
    """Evaluate a margin threshold; positive means included in the review."""
    yt = np.asarray(y_true, dtype=bool)
    margin = np.asarray(margins, dtype=float)
    empty = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "f2": 0.0,
        "workload_reduction": 0.0,
        "threshold": float(threshold),
    }
    if len(yt) != len(margin) or len(yt) == 0:
        return empty

    predicted = margin >= threshold
    tp = int(np.sum(yt & predicted))
    fp = int(np.sum((~yt) & predicted))
    tn = int(np.sum((~yt) & (~predicted)))
    fn = int(np.sum(yt & (~predicted)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    f2 = float(5 * precision * recall / (4 * precision + recall)) if 4 * precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "workload_reduction": float(np.mean(~predicted)),
        "threshold": float(threshold),
    }


def screening_threshold_curve(
    y_true: np.ndarray | pd.Series | list[bool | int],
    margins: np.ndarray | pd.Series | list[float],
) -> pd.DataFrame:
    """Evaluate every observed decision boundary for a precision-recall view."""
    margin = np.asarray(margins, dtype=float)
    if not len(margin):
        return pd.DataFrame()
    thresholds = np.sort(np.unique(margin))
    return pd.DataFrame(
        [evaluate_screening_threshold(y_true, margin, float(threshold)) for threshold in thresholds]
    )


def find_optimal_screening_threshold(
    y_true: np.ndarray | pd.Series | list[bool | int],
    margins: np.ndarray | pd.Series | list[float],
    min_recall: float = 0.98,
) -> dict[str, object]:
    """Maximize specificity subject to recall, then precision and conservatism."""
    yt = np.asarray(y_true, dtype=bool)
    margin = np.asarray(margins, dtype=float)
    curve = screening_threshold_curve(yt, margin)
    feasible = (
        curve[curve["recall"] >= min_recall] if not curve.empty and yt.any() else curve.iloc[0:0]
    )
    if feasible.empty:
        return {
            "optimal_threshold": None,
            "min_recall_target": min_recall,
            "metrics_at_optimal": None,
            "feasible": False,
        }
    best = feasible.sort_values(
        ["specificity", "precision", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    metrics = evaluate_screening_threshold(yt, margin, float(best["threshold"]))
    return {
        "optimal_threshold": float(best["threshold"]),
        "min_recall_target": min_recall,
        "metrics_at_optimal": metrics,
        "feasible": True,
    }


def calibrate_screening_threshold(
    resolved_labels: pd.DataFrame,
    scored_df: pd.DataFrame,
    *,
    min_recall: float = 0.98,
    test_size: float = 0.30,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Fit a candidate threshold on 70% and report metrics on a held-out 30%."""
    result: dict[str, object] = {
        "valid": False,
        "reason": "insufficient_labels",
        "n_resolved": 0,
        "threshold": None,
        "metrics": None,
        "confidence_intervals": {},
        "curve": pd.DataFrame(),
    }
    required_labels = {"doi", "y_true", "resolved"}
    required_scores = {"doi", "relevance_margin"}
    if not required_labels.issubset(resolved_labels.columns) or not required_scores.issubset(
        scored_df.columns
    ):
        return result
    labels = resolved_labels[resolved_labels["resolved"]].copy()
    scores = scored_df[["doi", "relevance_margin"]].copy()
    scores["doi"] = scores["doi"].map(_normalize_doi_value)
    joined = labels.merge(scores, on="doi", how="inner").dropna(
        subset=["y_true", "relevance_margin"]
    )
    joined["y_true"] = joined["y_true"].astype(bool)
    result["n_resolved"] = len(joined)
    class_counts = joined["y_true"].value_counts()
    if len(joined) < 40 or len(class_counts) < 2 or int(class_counts.min()) < 10:
        return result

    train, holdout = train_test_split(
        joined,
        test_size=test_size,
        random_state=seed,
        stratify=joined["y_true"],
    )
    selected = find_optimal_screening_threshold(
        train["y_true"], train["relevance_margin"], min_recall=min_recall
    )
    if not selected["feasible"]:
        result["reason"] = "no_feasible_threshold"
        return result
    threshold = float(selected["optimal_threshold"])
    metrics = evaluate_screening_threshold(
        holdout["y_true"], holdout["relevance_margin"], threshold
    )

    rng = np.random.default_rng(seed)
    positives = holdout[holdout["y_true"]]
    negatives = holdout[~holdout["y_true"]]
    bootstrap_metrics: dict[str, list[float]] = {
        metric: [] for metric in ("precision", "recall", "specificity", "f2", "workload_reduction")
    }
    for _ in range(n_bootstrap):
        sampled = pd.concat(
            [
                positives.iloc[rng.integers(0, len(positives), len(positives))],
                negatives.iloc[rng.integers(0, len(negatives), len(negatives))],
            ],
            ignore_index=True,
        )
        estimate = evaluate_screening_threshold(
            sampled["y_true"], sampled["relevance_margin"], threshold
        )
        for metric in bootstrap_metrics:
            bootstrap_metrics[metric].append(float(estimate[metric]))
    intervals = {
        metric: tuple(np.quantile(values, [0.025, 0.975]).astype(float))
        for metric, values in bootstrap_metrics.items()
    }
    return {
        "valid": True,
        "reason": None,
        "n_resolved": len(joined),
        "n_calibration": len(train),
        "n_holdout": len(holdout),
        "threshold": threshold,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "curve": screening_threshold_curve(holdout["y_true"], holdout["relevance_margin"]),
    }
