"""Vocabulary statistics: Zipf's law, dynamic c-TF-IDF topics, conceptual
atypicality of keyword combinations, and the rising/declining keyword trends.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from apps.dashboard.analytics.counts import (
    benjamini_hochberg,
    explode_keywords,
    linear_slope_with_ci,
    mann_kendall_trend,
    valid_years,
)

# Below this many (keyword, year) rows the trend tab has too little to fit.
MIN_KEYWORD_TREND_ROWS = 30
# How many terms each end of the slope ranking shows.
TREND_EXTREMES = 7
FDR_ALPHA = 0.05


def zipf_law_analysis(df: pd.DataFrame) -> dict:
    """Evaluate Zipf's law for technical vocabulary in titles and abstracts.

    Calculates word frequency vs. rank, fits log(f) = log(C) - gamma * log(r)
    via OLS regression, and returns goodness of fit (R^2), slope gamma (~1.0),
    and a top words table.
    """
    from collections import Counter

    # Aggregate text from title and abstract
    texts = []
    if "title" in df.columns:
        texts.append(df["title"].dropna().astype(str))
    if "abstract" in df.columns:
        texts.append(df["abstract"].dropna().astype(str))

    if not texts:
        return {
            "valid": False,
            "gamma": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "vocab_size": 0,
            "total_tokens": 0,
            "ranks": [],
            "frequencies": [],
            "words": [],
            "top_words_df": pd.DataFrame(),
        }

    full_series = pd.concat(texts, ignore_index=True)
    all_text = " ".join(full_series)
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
    total_tokens = len(tokens)

    stopwords = {
        "the",
        "of",
        "and",
        "in",
        "to",
        "for",
        "with",
        "on",
        "as",
        "by",
        "at",
        "an",
        "be",
        "is",
        "are",
        "from",
        "that",
        "this",
        "these",
        "it",
        "its",
        "or",
        "can",
        "has",
        "have",
        "been",
        "was",
        "were",
        "into",
        "also",
        "such",
        "than",
        "through",
        "which",
        "our",
        "their",
        "paper",
        "presents",
        "proposed",
        "method",
        "based",
        "using",
        "results",
        "study",
        "analysis",
        "model",
        "system",
        "systems",
        "approach",
        "case",
        "different",
        "new",
    }
    filtered_tokens = [w for w in tokens if w not in stopwords]
    counter = Counter(filtered_tokens)

    if len(counter) < 10:
        return {
            "valid": False,
            "gamma": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "vocab_size": len(counter),
            "total_tokens": total_tokens,
            "ranks": [],
            "frequencies": [],
            "words": [],
            "top_words_df": pd.DataFrame(),
        }

    top_items = counter.most_common(500)
    words = [w for w, _ in top_items]
    freqs = np.array([c for _, c in top_items], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    log_r = np.log10(ranks)
    log_f = np.log10(freqs)

    slope, intercept = np.polyfit(log_r, log_f, 1)
    pred = slope * log_r + intercept
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    ss_res = np.sum((log_f - pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    gamma = -float(slope)

    expected_ideal = freqs[0] / ranks

    top_df = pd.DataFrame(
        {
            "Rank (r)": ranks[:30].astype(int),
            "Term": words[:30],
            "Real Frequency (f)": freqs[:30].astype(int),
            "Ideal Zipf forecast": expected_ideal[:30].round(1),
            "Empirical Adjustment": (10 ** pred[:30]).round(1),
        }
    )

    return {
        "valid": True,
        "gamma": gamma,
        "intercept": float(intercept),
        "r_squared": float(r2),
        "vocab_size": len(counter),
        "total_tokens": total_tokens,
        "ranks": ranks.tolist(),
        "frequencies": freqs.tolist(),
        "words": words,
        "expected_zipf": (10**pred).tolist(),
        "top_words_df": top_df,
    }


def dynamic_topic_ctfidf(
    df: pd.DataFrame, time_windows: list[tuple[int, int]] | None = None
) -> dict:
    """Compute Class-based TF-IDF (c-TF-IDF) by research theme across chronological epochs.

    Reveals which technical terms distinguish each theme during each epoch.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if time_windows is None:
        time_windows = [(1990, 2010), (2011, 2018), (2019, 2026)]

    req_cols = ["theme_label", "year"]
    if not all(col in df.columns for col in req_cols):
        return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}

    text_series = df["title"].fillna("")
    if "abstract" in df.columns:
        text_series = text_series + " " + df["abstract"].fillna("")

    valid_df = df.copy()
    valid_df["_full_text"] = text_series
    valid_df["year"] = pd.to_numeric(valid_df["year"], errors="coerce")
    valid_df = valid_df.dropna(subset=["year", "theme_label"])
    valid_df["year"] = valid_df["year"].astype(int)

    epochs_data = []
    summary_rows = []

    global_docs = []
    for start, end in time_windows:
        sub = valid_df[(valid_df["year"] >= start) & (valid_df["year"] <= end)]
        if len(sub) >= 10:
            grouped = sub.groupby("theme_label")["_full_text"].apply(lambda s: " ".join(s))
            if len(grouped) >= 2:
                global_docs.extend(grouped.values)

    if not global_docs:
        return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}

    try:
        vec = CountVectorizer(stop_words="english", min_df=1, max_features=1500)
        vec.fit(global_docs)
        vocab = np.array(vec.get_feature_names_out())
        if len(vocab) < 3:
            return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}
    except Exception:
        return {"valid": False, "epochs": [], "summary_df": pd.DataFrame()}

    for start, end in time_windows:
        epoch_name = f"{start}–{end}"
        sub = valid_df[(valid_df["year"] >= start) & (valid_df["year"] <= end)]
        if len(sub) < 10:
            continue

        grouped = sub.groupby("theme_label")["_full_text"].apply(lambda s: " ".join(s))
        themes = list(grouped.index)
        docs = list(grouped.values)

        if len(themes) < 2:
            continue

        try:
            X = vec.transform(docs).toarray()

            tf = X
            ft = X.sum(axis=0)
            A = X.sum(axis=1).mean()
            idf = np.log(1 + A / (ft + 1e-9))
            ctfidf = tf * idf

            theme_res = {}
            for i, th in enumerate(themes):
                top_indices = np.argsort(-ctfidf[i])[:5]
                top_kws = [vocab[idx] for idx in top_indices if ctfidf[i, idx] > 0]
                theme_res[th] = top_kws
                summary_rows.append(
                    {
                        "Time": epoch_name,
                        "Theme": th,
                        "Characteristic Terms (c-TF-IDF)": ", ".join(top_kws),
                        "Articles": int((sub["theme_label"] == th).sum()),
                    }
                )

            epochs_data.append(
                {
                    "name": epoch_name,
                    "start": start,
                    "end": end,
                    "themes": theme_res,
                }
            )
        except Exception:
            continue

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    return {"valid": len(epochs_data) > 0, "epochs": epochs_data, "summary_df": summary_df}


def conceptual_atypicality_analysis(df: pd.DataFrame, top_n_keywords: int = 50) -> dict:
    """Describe uncommon keyword combinations and their citation association.

    This independence-based co-occurrence diagnostic is exploratory. It is
    not the journal-pair randomized null model introduced by Uzzi et al.
    """
    import itertools
    from collections import Counter

    if df.empty or "keywords" not in df.columns:
        return {
            "valid": False,
            "articles_df": pd.DataFrame(),
            "atypical_pairs": pd.DataFrame(),
            "hit_rate_high_atypical": 0.0,
            "hit_rate_baseline": 0.0,
        }

    # Extract keywords per article
    doc_kws = []
    kw_counts = Counter()
    for kws in df["keywords"]:
        if isinstance(kws, list):
            cleaned = [str(k).strip().lower() for k in kws if str(k).strip()]
            unique_k = sorted(set(cleaned))
            doc_kws.append(unique_k)
            kw_counts.update(unique_k)
        else:
            doc_kws.append([])

    top_vocab = {kw for kw, _ in kw_counts.most_common(top_n_keywords)}
    if len(top_vocab) < 3:
        return {
            "valid": False,
            "articles_df": pd.DataFrame(),
            "atypical_pairs": pd.DataFrame(),
            "hit_rate_high_atypical": 0.0,
            "hit_rate_baseline": 0.0,
        }

    # Count pairs across all articles
    pair_counts = Counter()
    total_pairs = 0
    for k_list in doc_kws:
        filtered = [k for k in k_list if k in top_vocab]
        for pair in itertools.combinations(filtered, 2):
            sorted_pair = tuple(sorted(pair))
            pair_counts[sorted_pair] += 1
            total_pairs += 1

    if total_pairs == 0:
        return {
            "valid": False,
            "articles_df": pd.DataFrame(),
            "atypical_pairs": pd.DataFrame(),
            "hit_rate_high_atypical": 0.0,
            "hit_rate_baseline": 0.0,
        }

    # Compute expected frequencies and z-scores under random null model
    pair_zscores = {}
    pair_rows = []
    for pair, count in pair_counts.items():
        k1, k2 = pair
        n1 = kw_counts[k1]
        n2 = kw_counts[k2]
        expected = (n1 * n2) / max(1, len(doc_kws))
        var = max(0.1, expected * (1.0 - n1 / len(doc_kws)) * (1.0 - n2 / len(doc_kws)))
        z = (count - expected) / np.sqrt(var)
        pair_zscores[pair] = float(z)
        if z < 0:
            pair_rows.append(
                {
                    "Term 1": k1,
                    "Term 2": k2,
                    "Real Co-occurrence": count,
                    "Expected": round(expected, 1),
                    "Z-score (atypicality)": round(z, 2),
                }
            )

    pair_rows.sort(key=lambda x: x["Z-score (atypicality)"])
    atypical_pairs_df = pd.DataFrame(pair_rows).head(20)

    # Score each article
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0).to_numpy()
    cite_p95 = float(np.percentile(cites, 95)) if len(cites) > 0 else 50.0

    art_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        k_list = doc_kws[idx] if idx < len(doc_kws) else []
        filtered = [k for k in k_list if k in top_vocab]
        pairs = [tuple(sorted(p)) for p in itertools.combinations(filtered, 2)]
        zs = [pair_zscores[p] for p in pairs if p in pair_zscores]

        if zs:
            med_z = float(np.median(zs))
            min_z = float(np.min(zs))
        else:
            med_z = 0.0
            min_z = 0.0

        c = float(cites[idx])
        is_hit = c >= cite_p95

        art_rows.append(
            {
                "doi": row.get("doi", ""),
                "title": row.get("title", ""),
                "year": row.get("year", 2020),
                "citations": c,
                "median_z": round(med_z, 2),
                "min_z": round(min_z, 2),
                "is_atypical": min_z < -0.5,
                "is_conventional": med_z > 0.0,
                "is_hit": is_hit,
            }
        )

    art_df = pd.DataFrame(art_rows)
    atypical_mask = art_df["is_atypical"] & art_df["is_conventional"]
    hit_rate_atypical = (
        float(art_df.loc[atypical_mask, "is_hit"].mean() * 100) if atypical_mask.sum() > 0 else 0.0
    )
    hit_rate_baseline = float(art_df["is_hit"].mean() * 100) if len(art_df) > 0 else 0.0

    return {
        "valid": True,
        "articles_df": art_df,
        "atypical_pairs": atypical_pairs_df,
        "hit_rate_high_atypical": round(hit_rate_atypical, 1),
        "hit_rate_baseline": round(hit_rate_baseline, 1),
        "cite_p95_threshold": int(cite_p95),
    }


def keyword_trend_frame(
    articles_df: pd.DataFrame, *, min_year: int, min_rows: int = MIN_KEYWORD_TREND_ROWS
) -> pd.DataFrame | None:
    """One row per (article, keyword) with an integer `year` >= `min_year`, or None.

    Shared by every keyword-trend view, so they all start from the same year
    window. None means there is not enough to analyze, not an error.
    """
    kw_year = explode_keywords(articles_df)
    if kw_year.empty or "year" not in kw_year.columns:
        return None
    kw_year["year"] = valid_years(kw_year, lo=min_year)
    kw_year = kw_year.dropna(subset=["year"]).astype({"year": int})
    if len(kw_year) < min_rows:
        return None
    return kw_year


def keyword_share_slopes(
    kw_year: pd.DataFrame, *, min_occurrences: int, n_extremes: int = TREND_EXTREMES
) -> pd.DataFrame:
    """OLS slope of each eligible keyword's annual share, the `n_extremes` at each end.

    A keyword is eligible with at least `min_occurrences` rows in `kw_year` and
    at least three years in which it appears. Its share is the percentage of all
    keyword mentions of that year. Columns: `keyword`, `slope`, `ci_low`,
    `ci_high`, `err_plus`, `err_minus`, `direction` ("Rising"/"Falling") and
    `excludes_zero`, sorted by slope ascending. Empty when nothing is eligible.
    """
    columns = [
        "keyword",
        "slope",
        "ci_low",
        "ci_high",
        "err_plus",
        "err_minus",
        "direction",
        "excludes_zero",
    ]
    if kw_year.empty:
        return pd.DataFrame(columns=columns)
    counts = kw_year["keyword"].value_counts()
    eligible = counts[counts >= min_occurrences].index
    by_year_total = kw_year.groupby("year").size()
    per_keyword = kw_year[kw_year["keyword"].isin(eligible)].groupby(["keyword", "year"]).size()

    fits: dict[str, dict[str, float]] = {}
    for kw, yearly in per_keyword.groupby(level="keyword"):
        yearly = yearly.droplevel("keyword")
        if len(yearly) < 3:
            continue
        share = (yearly / by_year_total.reindex(yearly.index)).fillna(0.0) * 100
        fit = linear_slope_with_ci(share.index.to_numpy(dtype=float), share.to_numpy(dtype=float))
        if np.isfinite(fit["slope"]):
            fits[str(kw)] = fit
    if not fits:
        return pd.DataFrame(columns=columns)

    slope_series = pd.Series({kw: fit["slope"] for kw, fit in fits.items()}).sort_values()
    top_slopes = pd.concat([slope_series.head(n_extremes), slope_series.tail(n_extremes)])
    # De-duplicated by keyword, not by value: a family smaller than
    # 2 * n_extremes puts a keyword in both halves, while two keywords that
    # merely share a slope are both real rows (`drop_duplicates()` on the
    # values, which this used to call, silently dropped one of them).
    top_slopes = top_slopes[~top_slopes.index.duplicated()]
    df = top_slopes.rename_axis("keyword").reset_index(name="slope")
    df["direction"] = np.where(df["slope"] >= 0, "Rising", "Falling")
    # Error bars carry the sampling spread the ranking hides: on series this
    # short the interval very often straddles zero even at the extremes.
    df["ci_low"] = df["keyword"].map(lambda kw: fits[kw]["ci_low"])
    df["ci_high"] = df["keyword"].map(lambda kw: fits[kw]["ci_high"])
    df["err_plus"] = df["ci_high"] - df["slope"]
    df["err_minus"] = df["slope"] - df["ci_low"]
    df["excludes_zero"] = (df["ci_low"] > 0) | (df["ci_high"] < 0)
    return df.sort_values("slope").reset_index(drop=True)[columns]


def keyword_trend_tests(
    kw_year: pd.DataFrame, *, min_occurrences: int, alpha: float = FDR_ALPHA
) -> pd.DataFrame:
    """Mann-Kendall + Benjamini-Hochberg over every eligible keyword (the whole family).

    The test family is every keyword with at least `min_occurrences` rows, not
    the extremes a chart shows. Adjusting only the top/bottom slopes would
    correct a set already chosen for being extreme, which inflates significance
    instead of controlling it -- so Mann-Kendall runs over the full family,
    Benjamini-Hochberg is applied there, and a caller narrows the table only
    afterwards. Series are zero-filled over the years present in `kw_year`.
    """
    columns = [
        "Keyword",
        "Trend",
        "Raw p-value",
        "Serial-corrected p-value",
        "Sen slope",
        "Adjusted p-value",
        "Adjusted p-value (serial)",
        "FDR significant",
        "FDR significant (serial)",
    ]
    if kw_year.empty:
        return pd.DataFrame(columns=columns)
    counts = kw_year["keyword"].value_counts()
    eligible = counts[counts >= min_occurrences].index
    pivoted = kw_year.groupby(["year", "keyword"]).size().unstack(fill_value=0)
    records = []
    for kw in eligible:
        if kw not in pivoted.columns:
            continue
        res = mann_kendall_trend(pivoted[kw].to_numpy())
        records.append(
            {
                "Keyword": kw,
                "Trend": str(res["trend"]).title(),
                "Raw p-value": float(res["p_value"]),
                # Mann-Kendall assumes independent years, which annual counts
                # are not. The Hamed-Rao variance inflation is reported beside
                # the raw value as a sensitivity rather than swapped in: it
                # changes which terms look significant.
                "Serial-corrected p-value": float(res["p_value_serial_corrected"]),
                "Sen slope": round(float(res["slope"]), 3),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    family = pd.DataFrame(records)
    family["Adjusted p-value"] = benjamini_hochberg(family["Raw p-value"])
    family["Adjusted p-value (serial)"] = benjamini_hochberg(family["Serial-corrected p-value"])
    family["FDR significant"] = family["Adjusted p-value"] < alpha
    family["FDR significant (serial)"] = family["Adjusted p-value (serial)"] < alpha
    return family[columns]


def rising_falling_keywords(
    kw_year: pd.DataFrame, *, min_occurrences: int, n_extremes: int = TREND_EXTREMES
) -> dict:
    """Everything the "Rise vs. decline" view shows, computed once.

    Returns `slopes` (from `keyword_share_slopes`), `family_size` (keywords
    tested), `flagged` (FDR-significant under independence), `survives`
    (significant under both the raw and the serial-corrected test), and
    `tests`: the family-adjusted Mann-Kendall rows narrowed to the charted
    keywords, p-values rounded to 4 places, sorted by Sen slope.
    """
    slopes = keyword_share_slopes(kw_year, min_occurrences=min_occurrences, n_extremes=n_extremes)
    family = keyword_trend_tests(kw_year, min_occurrences=min_occurrences)
    tests = family[family["Keyword"].isin(set(slopes["keyword"]))].copy()
    p_columns = [
        "Raw p-value",
        "Adjusted p-value",
        "Serial-corrected p-value",
        "Adjusted p-value (serial)",
    ]
    tests[p_columns] = tests[p_columns].astype(float).round(4)
    tests = tests.sort_values("Sen slope").reset_index(drop=True)
    return {
        "slopes": slopes,
        "tests": tests,
        "family_size": len(family),
        "flagged": int(family["FDR significant"].sum()),
        "survives": int((family["FDR significant"] & family["FDR significant (serial)"]).sum()),
    }
