"""Method taxonomy: the regex classifiers behind the Synthesis page and the
evidence samplers.

Optimization methods, benchmark feeders, objective functions, uncertainty
paradigms, planning horizons, solvers and complexity classes, each a pattern
table plus the function that counts it over title + abstract
(`taxonomy_haystack`). `TAXONOMY_REGISTRY` lists them for the coverage and
label-precision diagnostics. The tables live in the core because the evidence
samplers draw review strata with exactly the rule the dashboard displays.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# The regexes `optimization_methods_taxonomy` classifies with, hoisted to module
# scope so a sampler drawing strata for human review matches on exactly the same
# rule the dashboard displays. Re-deriving the match from the label text (which
# `evidence taxonomy` did at first) silently produced zero classes and asked
# reviewers to label a stratum the classifier does not recognise.
OPTIMIZATION_METHOD_PATTERNS: dict[str, re.Pattern[str]] = {
    "Multi-objective Optimization": re.compile(r"multi-objective|pareto"),
    "Genetic Algorithms (GA)": re.compile(r"genetic algorithm|\bga\b"),
    "Particle Swarm Optimization (PSO)": re.compile(r"particle swarm|\bpso\b"),
    "Mixed-Integer Linear Programming (MILP)": re.compile(r"milp|mixed-integer linear"),
    "Machine Learning & AI": re.compile(
        r"machine learning|deep learning|reinforcement learning|neural network"
    ),
    "Other Metaheuristics": re.compile(
        r"differential evolution|harmony search|simulated annealing|ant colony"
    ),
    "Stochastic Programming": re.compile(r"stochastic programming|scenario-based"),
    "Robust Optimization": re.compile(r"robust optimization|robust approach"),
    "Conical / Convex Relaxation (SOCP)": re.compile(
        r"second-order cone|conic|convex relaxation|socp"
    ),
}


def taxonomy_haystack(frame: pd.DataFrame) -> pd.Series:
    """Lowercased title + abstract, the exact text the taxonomy matches against.

    Keywords are deliberately excluded because the classifier excludes them;
    a sampler that searched a wider field would build strata the dashboard
    disagrees with.
    """
    return (
        frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("abstract", pd.Series("", index=frame.index)).fillna("").astype(str)
    ).str.lower()


def optimization_methods_taxonomy(df: pd.DataFrame) -> dict:
    """Extract and quantify optimization methods used across the corpus.

    Identifies 8 major mathematical modeling and optimization paradigms:
    - MILP / Mixed-Integer Linear
    - Genetic Algorithms (GA)
    - Particle Swarm Optimization (PSO)
    - Robust Optimization
    - Stochastic Programming
    - Conic / Convex Relaxation (SOCP)
    - Machine Learning / AI / Reinforcement Learning
    - Multi-Objective Optimization
    - Diverse Metaheuristics
    """

    if df.empty:
        return {"summary_df": pd.DataFrame(), "temporal_df": pd.DataFrame()}

    opt_patterns = OPTIMIZATION_METHOD_PATTERNS

    if "_cached_titles_abs" not in df.columns:
        df["_cached_titles_abs"] = taxonomy_haystack(df)
    titles_abs = df["_cached_titles_abs"]

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    summary_rows = []
    temporal_dict = {}

    for m_label, pat in opt_patterns.items():
        matched = titles_abs.str.contains(pat, regex=True)
        count = int(matched.sum())
        pct_recent = float((years[matched] >= 2021).mean() * 100) if count > 0 else 0.0
        mean_cites = float(cites[matched].mean()) if count > 0 else 0.0

        summary_rows.append(
            {
                "method": m_label,
                "articles": count,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_cites, 1),
            }
        )

        # Yearly counts for articles >= 2005
        # Counting rows needs no particular column: keying this on the `id`
        # surrogate coupled the taxonomy to a primary key it never uses,
        # and broke on any frame assembled without one.
        yearly = df[matched & (years >= 2005)].groupby("year").size()
        temporal_dict[m_label] = yearly

    summary_df = pd.DataFrame(summary_rows).sort_values(by="articles", ascending=False)
    temporal_df = pd.DataFrame(temporal_dict).fillna(0).astype(int)
    temporal_df.index = temporal_df.index.astype(int)

    return {
        "summary_df": summary_df,
        "temporal_df": temporal_df,
    }


BENCHMARK_FEEDER_PATTERNS = {
    "IEEE 33-Bus (standard radial)": re.compile(r"33-bus|ieee 33|33 bus|33-node"),
    "IEEE 69-Bus": re.compile(r"69-bus|ieee 69|69 bus|69-node"),
    "Real Utility Networks": re.compile(
        r"real distribution|real-world|practical distribution|actual distribution|utility network"
    ),
    "Regional Systems (Brazil / Europe)": re.compile(
        r"brazilian|european|california|uk distribution|nordic"
    ),
    "IEEE 123-Bus / 119-Bus": re.compile(r"123-bus|ieee 123|119-bus|ieee 119"),
}


def benchmark_feeders_analysis(df: pd.DataFrame) -> dict:
    """Analyze IEEE standard benchmark test feeders and real-world distribution networks."""

    if df.empty:
        return {"feeders_df": pd.DataFrame(), "cross_matrix": pd.DataFrame()}

    feeder_patterns = BENCHMARK_FEEDER_PATTERNS

    resource_patterns = {
        "Solar Generation (PV)": re.compile(r"photovoltaic|\bpv\b|solar"),
        "Storage / Batteries": re.compile(r"energy storage|battery|bess"),
        "Electric Vehicles (EV)": re.compile(r"electric vehicle|\bev\b|v2g|charging"),
        "Network reconfiguration": re.compile(r"reconfiguration|switching|switch"),
    }

    if "_cached_titles_abs" not in df.columns:
        df["_cached_titles_abs"] = taxonomy_haystack(df)
    titles_abs = df["_cached_titles_abs"]

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    cross_data = {r_label: [] for r_label in resource_patterns}

    for f_label, pat in feeder_patterns.items():
        f_match = titles_abs.str.contains(pat, regex=True)
        count = int(f_match.sum())
        pct_recent = float((years[f_match] >= 2021).mean() * 100) if count > 0 else 0.0
        mean_cites = float(cites[f_match].mean()) if count > 0 else 0.0

        rows.append(
            {
                "feeder": f_label,
                "articles": count,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_cites, 1),
            }
        )

        for r_label, r_pat in resource_patterns.items():
            r_match = titles_abs.str.contains(r_pat, regex=True)
            cross_count = int((f_match & r_match).sum())
            cross_data[r_label].append(cross_count)

    feeders_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    cross_matrix = pd.DataFrame(cross_data, index=list(feeder_patterns.keys()))

    return {
        "feeders_df": feeders_df,
        "cross_matrix": cross_matrix,
    }


OBJECTIVE_FUNCTION_PATTERNS = {
    "Economic Costs": r"cost|capex|opex|investment|economic|capital expenditure",
    "Reliability (SAIDI/SAIFI/ENS)": r"reliability|saidi|saifi|ens|energy not supplied|interruption|outage|unserved",
    "Technical losses": r"power loss|energy loss|technical loss|transmission loss|loss reduction",
    "Voltage Profile": r"voltage profile|voltage deviation|voltage stability|power quality|voltage drop|voltage regulation",
    "Decarbonization / Emissions": r"emission|carbon|decarboniz|greenhouse|environmental|co2",
    "Resilience": r"resilience|extreme weather|disaster|blackout|hardening|restoration",
}


def objective_functions_taxonomy(df: pd.DataFrame) -> dict:
    """Analyze optimized objectives and multi-criteria formulations."""
    if df.empty:
        return {
            "summary_df": pd.DataFrame(),
            "co_matrix": pd.DataFrame(),
            "multi_obj_ratio": 0.0,
            "temporal_multiobj": pd.DataFrame(),
        }

    objs = OBJECTIVE_FUNCTION_PATTERNS

    titles_abs = taxonomy_haystack(df)

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    matches = {k: titles_abs.str.contains(pat, regex=True) for k, pat in objs.items()}
    summary_rows = []
    for k, m in matches.items():
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        summary_rows.append(
            {
                "objective": k,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="articles", ascending=False)

    co_matrix = pd.DataFrame(index=list(objs.keys()), columns=list(objs.keys()), dtype=int)
    for k1 in objs:
        for k2 in objs:
            co_matrix.loc[k1, k2] = int((matches[k1] & matches[k2]).sum())

    match_matrix = np.column_stack([matches[k].to_numpy() for k in objs])
    obj_counts = match_matrix.sum(axis=1)
    has_any = obj_counts >= 1
    multi_count = int((obj_counts >= 2).sum())
    multi_ratio = float(multi_count / has_any.sum() * 100) if has_any.sum() > 0 else 0.0

    valid_years = (years >= 2005) & (years <= 2026) & has_any
    temp_df = pd.DataFrame(
        {
            "year": years[valid_years].astype(int),
            "is_multi": (obj_counts[valid_years] >= 2),
        }
    )
    if not temp_df.empty:
        grouped = temp_df.groupby(["year", "is_multi"]).size().unstack(fill_value=0)
        temporal_multiobj = pd.DataFrame(
            {
                "year": grouped.index,
                "mono_objective": grouped.get(False, 0),
                "multi_objective": grouped.get(True, 0),
            }
        ).reset_index(drop=True)
        total = temporal_multiobj["mono_objective"] + temporal_multiobj["multi_objective"]
        temporal_multiobj["pct_multi"] = (
            temporal_multiobj["multi_objective"] / total.clip(lower=1) * 100
        ).round(1)
    else:
        temporal_multiobj = pd.DataFrame()

    return {
        "summary_df": summary_df,
        "co_matrix": co_matrix,
        "multi_obj_ratio": round(multi_ratio, 1),
        "temporal_multiobj": temporal_multiobj,
    }


UNCERTAINTY_PARADIGM_PATTERNS = {
    "Stochastic (Scenarios / Monte Carlo)": r"stochastic|scenario-based|monte carlo|sample average",
    "Robust Optimization (Min-Max)": r"robust optimization|robust approach|uncertainty set|worst-case",
    "Fuzzy Logic": r"fuzzy",
    "Distributionally Robust Optimization (DRO)": r"distributionally robust|wasserstein|ambiguity set",
    "Chance Constraint (Probabilistic)": r"chance-constrained|chance constraint|probabilistic constraint",
    "Deterministic (Fixed Case)": r"deterministic",
}


def uncertainty_paradigms_analysis(df: pd.DataFrame) -> dict:
    """Analyze mathematical paradigms for handling uncertainty in distribution systems."""
    if df.empty:
        return {
            "paradigms_df": pd.DataFrame(),
            "cross_resources": pd.DataFrame(),
            "temporal_paradigms": pd.DataFrame(),
        }

    paradigms = UNCERTAINTY_PARADIGM_PATTERNS

    resources = {
        "Solar Generation (PV)": r"photovoltaic|\bpv\b|solar",
        "Batteries / Storage (BESS)": r"energy storage|battery|bess",
        "Electric Vehicles (EV)": r"electric vehicle|\bev\b|v2g|charging",
        "Demand / Load Uncertainty": r"load uncertainty|demand uncertainty|forecast error|load variation",
    }

    titles_abs = taxonomy_haystack(df)

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    p_matches = {k: titles_abs.str.contains(pat, regex=True) for k, pat in paradigms.items()}
    r_matches = {k: titles_abs.str.contains(pat, regex=True) for k, pat in resources.items()}

    summary_rows = []
    for k, m in p_matches.items():
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        summary_rows.append(
            {
                "paradigm": k,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    paradigms_df = pd.DataFrame(summary_rows).sort_values(by="articles", ascending=False)

    cross_data = {r_label: [] for r_label in resources}
    for _p_label, p_m in p_matches.items():
        for r_label, r_m in r_matches.items():
            cross_data[r_label].append(int((p_m & r_m).sum()))

    cross_resources = pd.DataFrame(cross_data, index=list(paradigms.keys()))

    temporal_dict = {}
    for p_label, p_m in p_matches.items():
        filt = p_m & (years >= 2005) & (years <= 2026)
        if filt.any():
            yearly = df[filt].groupby(years[filt].astype(int)).size()
            temporal_dict[p_label] = yearly

    temporal_paradigms = pd.DataFrame(temporal_dict).fillna(0).astype(int)
    if not temporal_paradigms.empty:
        temporal_paradigms.index.name = "year"
        temporal_paradigms = temporal_paradigms.reset_index()

    return {
        "paradigms_df": paradigms_df,
        "cross_resources": cross_resources,
        "temporal_paradigms": temporal_paradigms,
    }


PLANNING_HORIZON_PATTERNS = {
    "Multistage Dynamic Expansion": r"multi-stage|multistage|multi-year|sequential expansion|expansion planning|dynamic planning",
    "Co-Optimization Planning + Operation": r"co-optimi|planning and operation|representative days|representative periods|operational constraints|chronological",
    "Static Planning (Target Year)": r"static planning|single-stage|target year|snapshot",
}


def planning_time_horizons_analysis(df: pd.DataFrame) -> dict:
    """Analyze static vs. multi-stage dynamic planning and operation co-optimization."""
    if df.empty:
        return {"horizons_df": pd.DataFrame()}

    horizons = PLANNING_HORIZON_PATTERNS

    titles_abs = taxonomy_haystack(df)

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    for h_label, pat in horizons.items():
        m = titles_abs.str.contains(pat, regex=True)
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        rows.append(
            {
                "horizon": h_label,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    horizons_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    return {"horizons_df": horizons_df}


SOLVER_PATTERNS = {
    "GAMS / AMPL (Algebraic Modelers)": (r"gams|ampl", "Algebraic Modeler"),
    "MATLAB / Simulink": (r"matlab|simulink", "Scripting & Simulation"),
    "CPLEX (IBM)": (r"cplex", "Commercial Exact Solver"),
    "DIgSILENT PowerFactory": (r"digsilent|powerfactory", "Specialized Electric Simulator"),
    "Gurobi Optimizer": (r"gurobi", "Commercial Exact Solver"),
    "OpenDSS (EPRI)": (r"opendss|open dss", "Distribution Simulator"),
    "Python (Pyomo / Pandapower)": (r"python|pyomo|pandapower", "Scripting & Open Source"),
    "PSCAD / EMTP": (r"pscad|emtp", "Transient Simulator"),
}


def computational_solvers_analysis(df: pd.DataFrame) -> dict:
    """Analyze mathematical solvers and simulation platforms utilized in the literature."""
    if df.empty:
        return {"solvers_df": pd.DataFrame(), "ecosystem_df": pd.DataFrame()}

    solvers = SOLVER_PATTERNS

    titles_abs = taxonomy_haystack(df)

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    for s_label, (pat, cat) in solvers.items():
        m = titles_abs.str.contains(pat, regex=True)
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        rows.append(
            {
                "tool": s_label,
                "category": cat,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

    solvers_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    eco_df = (
        solvers_df.groupby("category")["articles"]
        .sum()
        .reset_index()
        .sort_values(by="articles", ascending=False)
    )

    return {
        "solvers_df": solvers_df,
        "ecosystem_df": eco_df,
    }


COMPLEXITY_CLASS_PATTERNS = {
    "Linear Programming / MILP (Exact)": r"milp|mixed-integer linear|\blp\b|linear programming",
    "Convex / Conical Relaxation (SOCP/SDP)": r"second-order cone|socp|semidefinite|convex relaxation|conic",
    "Non-Linear Prog (NLP / MINLP)": r"minlp|mixed-integer nonlinear|nonlinear programming|\bnlp\b|non-convex",
    "Metaheuristics (GA, PSO, DE, ACO)": r"genetic algorithm|particle swarm|\bpso\b|\bga\b|differential evolution|ant colony|harmony search|simulated annealing",
    "AI & Reinforcement Learning": r"machine learning|deep learning|reinforcement learning|neural network|q-learning",
}


# Every regex taxonomy the Engineering-evidence page displays, so coverage can
# be reported the same way for all of them and a reviewed sample can be scored
# against the rule the dashboard actually applies. `WP-12` requires each
# displayed taxonomy to disclose its evaluated coverage; a registry is what
# makes "each" enumerable instead of a promise.
TAXONOMY_REGISTRY: dict[str, dict] = {
    "Optimization methods": OPTIMIZATION_METHOD_PATTERNS,
    "Objective functions": OBJECTIVE_FUNCTION_PATTERNS,
    "Uncertainty paradigms": UNCERTAINTY_PARADIGM_PATTERNS,
    "Planning horizons": PLANNING_HORIZON_PATTERNS,
    "Computational solvers": SOLVER_PATTERNS,
    "Mathematical complexity": COMPLEXITY_CLASS_PATTERNS,
    "Benchmark feeders": BENCHMARK_FEEDER_PATTERNS,
}


def _pattern_of(value) -> str:
    """Take the regex out of a registry entry.

    `SOLVER_PATTERNS` pairs each regex with an ecosystem label, so entries are
    not uniformly bare patterns.
    """
    return value[0] if isinstance(value, tuple) else value


def taxonomy_coverage(frame: pd.DataFrame, patterns: dict) -> dict:
    """How much of the corpus a regex taxonomy actually classifies.

    A frequency chart drawn only over matched articles silently answers "of the
    ones I recognised, which is most common?" while looking like it answers
    "what does this corpus do?". The share it never matched is the difference
    between those two questions, and on this corpus it is large.

    Multi-label counts matter for the same reason: these regexes are not
    mutually exclusive, so the per-class counts sum to more than the classified
    population and cannot be read as a partition.
    """
    if frame.empty or not patterns:
        return {
            "population": 0,
            "classified": 0,
            "unclassified": 0,
            "coverage": float("nan"),
            "multi_label": 0,
            "per_class": {},
        }

    haystack = taxonomy_haystack(frame)
    hits = {
        label: haystack.str.contains(_pattern_of(value), regex=True)
        for label, value in patterns.items()
    }
    matrix = pd.DataFrame(hits, index=frame.index)
    per_row = matrix.sum(axis=1)
    population = int(len(frame))
    classified = int((per_row > 0).sum())
    return {
        "population": population,
        "classified": classified,
        "unclassified": population - classified,
        "coverage": classified / population,
        "multi_label": int((per_row > 1).sum()),
        "per_class": {label: int(series.sum()) for label, series in hits.items()},
    }


def taxonomy_precision_from_labels(
    frame: pd.DataFrame, patterns: dict, labels: pd.DataFrame
) -> pd.DataFrame:
    """Score a regex taxonomy against reviewed labels, per class.

    `labels` carries the review workflow's own columns -- `subject_id` shaped
    `<class>::<doi>` and `label` in {present, absent, ambiguous}. Ambiguous
    rows are excluded from precision and recall and counted separately: a
    reviewer who could not decide is evidence about the class boundary, not a
    negative.

    Returns an empty frame when there are no usable labels, which is the
    current state -- the point of this function is that the panel upgrades
    itself from exploratory to measured the moment a review lands, with no
    further code change.
    """
    columns = [
        "class",
        "labelled",
        "ambiguous",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]
    if frame.empty or labels.empty or not {"subject_id", "label"}.issubset(labels.columns):
        return pd.DataFrame(columns=columns)

    parsed = labels.dropna(subset=["subject_id", "label"]).copy()
    split = parsed["subject_id"].astype(str).str.split("::", n=1, expand=True)
    if split.shape[1] < 2:
        return pd.DataFrame(columns=columns)
    parsed["class"] = split[0].str.strip()
    parsed["doi"] = split[1].str.strip()
    parsed["label"] = parsed["label"].astype(str).str.strip().str.casefold()

    haystack = taxonomy_haystack(frame)
    doi_index = frame.get("doi")
    if doi_index is None:
        return pd.DataFrame(columns=columns)
    predicted = {
        label: set(
            frame.loc[haystack.str.contains(_pattern_of(value), regex=True), "doi"].astype(str)
        )
        for label, value in patterns.items()
    }

    rows = []
    for class_name, group in parsed.groupby("class"):
        if class_name not in predicted:
            continue
        decided = group[group["label"].isin({"present", "absent"})]
        positives = set(decided.loc[decided["label"] == "present", "doi"])
        negatives = set(decided.loc[decided["label"] == "absent", "doi"])
        matched = predicted[class_name]
        tp = len(positives & matched)
        fp = len(negatives & matched)
        fn = len(positives - matched)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and (precision + recall) > 0
            else float("nan")
        )
        rows.append(
            {
                "class": class_name,
                "labelled": int(len(decided)),
                "ambiguous": int((group["label"] == "ambiguous").sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def mathematical_complexity_spectrum(df: pd.DataFrame) -> dict:
    """Classify papers into mathematical complexity classes."""
    if df.empty:
        return {"spectrum_df": pd.DataFrame(), "temporal_spectrum": pd.DataFrame()}

    classes = COMPLEXITY_CLASS_PATTERNS

    titles_abs = taxonomy_haystack(df)

    years = pd.to_numeric(df.get("year"), errors="coerce").fillna(2015)
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0)

    rows = []
    temporal_dict = {}

    for c_label, pat in classes.items():
        m = titles_abs.str.contains(pat, regex=True)
        cnt = int(m.sum())
        pct_recent = float((years[m] >= 2021).mean() * 100) if cnt > 0 else 0.0
        mean_c = float(cites[m].mean()) if cnt > 0 else 0.0
        rows.append(
            {
                "complexity_class": c_label,
                "articles": cnt,
                "pct_recent": round(pct_recent, 1),
                "mean_citations": round(mean_c, 1),
            }
        )

        filt = m & (years >= 2005) & (years <= 2026)
        if filt.any():
            yearly = df[filt].groupby(years[filt].astype(int)).size()
            temporal_dict[c_label] = yearly

    spectrum_df = pd.DataFrame(rows).sort_values(by="articles", ascending=False)
    temporal_spectrum = pd.DataFrame(temporal_dict).fillna(0).astype(int)
    if not temporal_spectrum.empty:
        temporal_spectrum.index.name = "year"
        temporal_spectrum = temporal_spectrum.reset_index()

    return {
        "spectrum_df": spectrum_df,
        "temporal_spectrum": temporal_spectrum,
    }
