"""Author-level aggregations: explosion by position, the author-year matrix,
productivity concentration, impact indices and the co-authorship network.

Author identity comes from `transform.authors.canonical_author`, the same
normalization the evidence samplers use, so a name is merged the same way on
every page and in every review export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apps.dashboard.analytics.counts import source_means, valid_years
from transform.authors import canonical_author


def author_count_series(df: pd.DataFrame) -> pd.Series:
    """Authors per row, from the `authors` list column.

    A list counts by its length; a non-null non-list scalar (a defensive
    fallback for a stray single-author value that never got wrapped in a
    list) counts as 1; null counts as 0. Shared by every page that needs an
    "authors per article" distribution or mean, so this edge case is handled
    the same way everywhere instead of drifting between pages.
    """
    if "authors" not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return df["authors"].apply(
        lambda a: len(a) if isinstance(a, list) else (1 if pd.notna(a) else 0)
    )


# Team-size trends start here: earlier years hold a handful of articles each,
# so a yearly mean there is one paper's team, not a trend.
TEAM_TREND_MIN_YEAR = 2000


def team_size_summary(df: pd.DataFrame) -> dict:
    """Authors-per-article statistics over the articles that list any author.

    Returns `n_articles`, `mean` (dict ieee/elsevier/total from `source_means`,
    None where a source is absent), `median` (overall), `solo_share` (fraction
    of single-author articles) and `distribution` (columns `authors`,
    `articles`, one row per observed team size). `n_articles == 0` means the
    layer has no author lists.
    """
    with_authors = df.assign(n_authors=author_count_series(df))
    with_authors = with_authors[with_authors["n_authors"] > 0]
    empty_distribution = pd.DataFrame(columns=["authors", "articles"])
    if with_authors.empty:
        return {
            "n_articles": 0,
            "mean": source_means(with_authors, "n_authors"),
            "median": None,
            "solo_share": None,
            "distribution": empty_distribution,
        }
    distribution = (
        with_authors["n_authors"]
        .value_counts()
        .sort_index()
        .rename_axis("authors")
        .reset_index(name="articles")
    )
    return {
        "n_articles": len(with_authors),
        "mean": source_means(with_authors, "n_authors"),
        "median": float(with_authors["n_authors"].median()),
        "solo_share": float((with_authors["n_authors"] == 1).mean()),
        "distribution": distribution,
    }


def team_size_by_year(df: pd.DataFrame, *, min_year: int = TEAM_TREND_MIN_YEAR) -> pd.DataFrame:
    """Mean authors per article for each publication year >= `min_year`.

    Columns `year`, `mean_authors`, plus `source` (one row per year and source)
    when the frame carries one. Articles without any author are left out, as
    in `team_size_summary`. It is a *mean*: the page used to label this series
    "Median authors" while computing `.mean()`.
    """
    working = df.assign(n_authors=author_count_series(df))
    working = working[working["n_authors"] > 0].copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"]).astype({"year": int})
    working = working[working["year"] >= min_year]
    keys = ["year", "source"] if "source" in working.columns else ["year"]
    if working.empty:
        return pd.DataFrame(columns=[*keys, "mean_authors"])
    return (
        working.groupby(keys)["n_authors"]
        .mean()
        .reset_index(name="mean_authors")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def cognitive_distance_by_article(
    author_rows: pd.DataFrame, articles: pd.DataFrame
) -> pd.DataFrame:
    """Mean pairwise distance between co-authors' semantic centroids, per article.

    `author_rows` needs `doi` and `author_key`; `articles` needs `doi`, `map_x`,
    `map_y` and `citation_count` (rows missing any of them are dropped), and may
    carry `title` and `year`. Each author's centroid is the mean map position of
    their articles *in that scope*; an article's cognitive distance is the mean
    Euclidean distance over every pair of its distinct authors, so it needs at
    least two. Columns: `doi`, `title` (first 80 characters), `year`,
    `cognitive_distance` (rounded to 3 places), `citation_count`, `team_size`.

    Vectorised as a self-merge on DOI rather than one lookup per article: the
    loop it replaces filtered the whole article frame once per DOI.
    """
    columns = ["doi", "title", "year", "cognitive_distance", "citation_count", "team_size"]
    needed = {"doi", "map_x", "map_y", "citation_count"}
    if author_rows.empty or not needed <= set(articles.columns):
        return pd.DataFrame(columns=columns)
    valid = articles.dropna(subset=["map_x", "map_y", "citation_count"]).drop_duplicates("doi")
    pairs_in = (
        author_rows[["doi", "author_key"]]
        .drop_duplicates()
        .merge(valid[["doi", "map_x", "map_y"]], on="doi")
    )
    if pairs_in.empty:
        return pd.DataFrame(columns=columns)
    centroids = pairs_in.groupby("author_key")[["map_x", "map_y"]].mean()
    members = pairs_in[["doi", "author_key"]].join(centroids, on="author_key")
    team_size = members.groupby("doi")["author_key"].nunique()
    multi = members[members["doi"].isin(team_size[team_size >= 2].index)]
    if multi.empty:
        return pd.DataFrame(columns=columns)

    pairs = multi.merge(multi, on="doi", suffixes=("_a", "_b"))
    pairs = pairs[pairs["author_key_a"] < pairs["author_key_b"]]
    pairs["distance"] = np.hypot(
        pairs["map_x_a"] - pairs["map_x_b"], pairs["map_y_a"] - pairs["map_y_b"]
    )
    mean_distance = pairs.groupby("doi")["distance"].mean()

    meta = valid.set_index("doi").reindex(mean_distance.index)
    title = (
        meta["title"].astype(str).str.slice(0, 80)
        if "title" in meta.columns
        else pd.Series("—", index=meta.index)
    )
    year = meta["year"] if "year" in meta.columns else pd.Series(np.nan, index=meta.index)
    return pd.DataFrame(
        {
            "doi": mean_distance.index,
            "title": title.to_numpy(),
            "year": year.to_numpy(),
            "cognitive_distance": mean_distance.round(3).to_numpy(),
            "citation_count": meta["citation_count"].astype(int).to_numpy(),
            "team_size": team_size.reindex(mean_distance.index).astype(int).to_numpy(),
        },
        columns=columns,
    ).reset_index(drop=True)


def cognitive_distance_analysis(
    author_rows: pd.DataFrame,
    articles: pd.DataFrame,
    *,
    min_articles: int = 10,
    min_teams: int = 5,
) -> dict:
    """Cognitive distance per article plus its correlation with citations.

    `status` is "ok", "few_articles" (fewer than `min_articles` articles carry a
    map position and a citation count) or "few_teams" (fewer than `min_teams`
    multi-author articles could be positioned). On "ok" it also returns
    `articles` (from `cognitive_distance_by_article`), `mean_distance`,
    `pearson` and `spearman` (distance vs. citations).
    """
    needed = {"map_x", "map_y", "citation_count"}
    if not needed <= set(articles.columns):
        return {"status": "few_articles"}
    if len(articles.dropna(subset=list(needed))) < min_articles:
        return {"status": "few_articles"}
    dist_df = cognitive_distance_by_article(author_rows, articles)
    if len(dist_df) < min_teams:
        return {"status": "few_teams"}
    distance = dist_df["cognitive_distance"].astype(float)
    citations = dist_df["citation_count"].astype(float)
    return {
        "status": "ok",
        "articles": dist_df,
        "mean_distance": float(distance.mean()),
        "pearson": float(distance.corr(citations, method="pearson")),
        "spearman": float(distance.corr(citations, method="spearman")),
    }


def explode_authors_with_position(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, author), plus the author's 0-based byline position.

    Position 0 is the first-listed author, 1 the second, etc. -- the order the
    source `.bib`/CSV author field was written in (not alphabetical; see
    `transform.bronze_articles._split_bibtex_authors`/`_split_ieee_csv_authors`).
    It's a positional count only, not a claim about authorship role -- what a
    given position conventionally means varies by field, and this corpus
    doesn't record roles.
    """
    columns = ["author", "position", "year", "source", "venue", "doi"]
    if df.empty or "authors" not in df.columns:
        return pd.DataFrame(columns=columns)
    keep = [c for c in ("year", "source", "venue", "doi", "citation_count") if c in df.columns]
    working = df[["authors", *keep]].copy()
    working["authors"] = working["authors"].apply(
        lambda lst: list(enumerate(lst)) if isinstance(lst, list) else np.nan
    )
    working = working.explode("authors")
    working = working[working["authors"].apply(lambda v: isinstance(v, tuple))]
    if working.empty:
        return pd.DataFrame(columns=columns)
    # `.explode()` repeats the original row index for every exploded entry, so a
    # positional (not index-aligned) assignment is required here -- joining two
    # frames that both carry duplicate index labels would cross-join within each
    # duplicated label instead of pairing rows one-to-one.
    tuples = working["authors"].tolist()
    working = working.drop(columns=["authors"])
    working["position"] = [t[0] for t in tuples]
    working["author"] = [t[1] for t in tuples]
    working = working[working["author"].notna() & (working["author"].astype(str).str.strip() != "")]
    return working.reset_index(drop=True)


def explode_authors(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article, author), keeping year/source/citation_count/venue."""
    return explode_authors_with_position(df).drop(columns=["position"])


def author_display_name(names: pd.Series) -> str:
    """Pick the most informative spelling among variants that share a key."""
    if names.empty:
        return ""
    return max(names.unique(), key=len)


_MATRIX_SUMMARY_COLS = ("total", "ieee_total", "elsevier_total")


def author_year_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical author, one column per valid year, plus source totals.

    Explodes `authors`, folds names via `canonical_author` (see its docstring
    for the "initial surname" identity-merge caveat -- any page showing this
    table must disclose it), and counts distinct articles per author per year
    via `doi` where available (falls back to row count otherwise, since two
    exploded rows for the same article/author pair would otherwise double-count
    a co-authored paper). Sorted descending by `total` per the page's "highest
    to lowest" requirement. Years outside `valid_years`' plausible window are
    dropped before pivoting, same as every other year-based aggregation here.

    `ieee_total`/`elsevier_total` break `total` down by source (0 when a
    `source` column isn't present), using the same distinct-DOI counting rule
    as the year columns -- the "3 real series" convention this dashboard uses
    everywhere else (see `source_counts_by`).
    """
    exploded = explode_authors(df)
    if exploded.empty:
        return pd.DataFrame(columns=["author", *_MATRIX_SUMMARY_COLS])

    exploded["author_key"] = exploded["author"].apply(canonical_author)
    exploded = exploded[exploded["author_key"] != ""]
    exploded["year"] = valid_years(exploded)
    exploded = exploded.dropna(subset=["year"]).astype({"year": int})
    if exploded.empty:
        return pd.DataFrame(columns=["author", *_MATRIX_SUMMARY_COLS])

    display_names = exploded.groupby("author_key")["author"].apply(author_display_name)
    exploded["author_display"] = exploded["author_key"].map(display_names)

    count_col = "doi" if "doi" in exploded.columns else "author"
    agg = "nunique" if count_col == "doi" else "size"
    pivot = (
        exploded.groupby(["author_display", "year"])[count_col]
        .agg(agg)
        .unstack(fill_value=0)
        .astype(int)
    )
    pivot.columns = [str(int(c)) for c in pivot.columns]
    pivot["total"] = pivot.sum(axis=1)

    if "source" in exploded.columns:
        src_pivot = (
            exploded.groupby(["author_display", "source"])[count_col].agg(agg).unstack(fill_value=0)
        )
        for src in ("ieee", "elsevier"):
            if src not in src_pivot.columns:
                src_pivot[src] = 0
        pivot["ieee_total"] = src_pivot["ieee"].reindex(pivot.index, fill_value=0).astype(int)
        pivot["elsevier_total"] = (
            src_pivot["elsevier"].reindex(pivot.index, fill_value=0).astype(int)
        )
    else:
        pivot["ieee_total"] = 0
        pivot["elsevier_total"] = 0

    pivot = pivot.sort_values("total", ascending=False)
    pivot.index.name = "author"
    year_cols = sorted((c for c in pivot.columns if c not in _MATRIX_SUMMARY_COLS), key=int)
    return pivot.reset_index()[["author", *_MATRIX_SUMMARY_COLS, *year_cols]]


def _exploded_author_years(df: pd.DataFrame, max_position: int | None = None) -> pd.DataFrame:
    """Shared prep for author-by-year aggregations: explode, canonicalize, valid years only.

    `max_position`, when given, restricts to authors at or before that 0-based
    byline position (e.g. `max_position=1` keeps only the 1st/2nd author of
    each article) -- see `explode_authors_with_position`.
    """
    exploded = (
        explode_authors_with_position(df) if max_position is not None else explode_authors(df)
    )
    if max_position is not None and not exploded.empty:
        exploded = exploded[exploded["position"] <= max_position]
    if exploded.empty:
        return exploded
    unique_authors = exploded["author"].unique()
    canonical_map = {a: canonical_author(a) for a in unique_authors}
    exploded["author_key"] = exploded["author"].map(canonical_map)
    exploded = exploded[exploded["author_key"] != ""]
    exploded["year"] = valid_years(exploded)
    return exploded.dropna(subset=["year"]).astype({"year": int})


def researchers_by_year(df: pd.DataFrame, max_position: int | None = None) -> pd.DataFrame:
    """Distinct canonical-author count per year, split ieee/elsevier/total.

    Unlike `source_counts_by` (which counts rows), `total` here is counted
    independently as the number of distinct authors active that year
    regardless of source -- an author publishing in both IEEE and Elsevier
    the same year must count once in `total`, not twice. `ieee`/`elsevier`
    default to 0 when a `source` column isn't present. `max_position` restricts
    to authors at or before that byline position (see `_exploded_author_years`).
    """
    exploded = _exploded_author_years(df, max_position=max_position)
    if exploded.empty:
        return pd.DataFrame(columns=["year", "ieee", "elsevier", "total"])

    years = sorted(exploded["year"].unique())
    result = pd.DataFrame({"year": years})
    has_source = "source" in exploded.columns
    for src in ("ieee", "elsevier"):
        if has_source:
            counts = exploded[exploded["source"] == src].groupby("year")["author_key"].nunique()
            result[src] = result["year"].map(counts).fillna(0).astype(int)
        else:
            result[src] = 0
    total_counts = exploded.groupby("year")["author_key"].nunique()
    result["total"] = result["year"].map(total_counts).fillna(0).astype(int)
    return result


def cumulative_researchers(df: pd.DataFrame, max_position: int | None = None) -> pd.DataFrame:
    """Cumulative count of distinct researchers introduced by each year.

    Each canonical author is counted once, in the year of their earliest
    valid-year appearance (within that source, for `ieee`/`elsevier`; across
    all sources, for `total`) -- summing each year's *active* researcher
    count would double-count an author active across multiple years, which
    isn't what a cumulative researcher count should mean. As with every other
    ieee/elsevier/total triple here, `total` isn't required to equal
    `ieee + elsevier` -- an author's first IEEE year and first Elsevier year
    can differ from their first-ever appearance. `max_position` restricts to
    authors at or before that byline position (see `_exploded_author_years`).
    """
    exploded = _exploded_author_years(df, max_position=max_position)
    if exploded.empty:
        return pd.DataFrame(columns=["year", "ieee", "elsevier", "total"])

    years = sorted(exploded["year"].unique())

    def cumulative_new(sub: pd.DataFrame) -> pd.Series:
        if sub.empty:
            return pd.Series(0, index=years, dtype="int64")
        first_year = sub.groupby("author_key")["year"].min()
        by_year = first_year.value_counts().reindex(years, fill_value=0)
        return by_year.cumsum()

    result = pd.DataFrame({"year": years})
    has_source = "source" in exploded.columns
    for src in ("ieee", "elsevier"):
        sub = exploded[exploded["source"] == src] if has_source else exploded.iloc[0:0]
        result[src] = cumulative_new(sub).to_numpy()
    result["total"] = cumulative_new(exploded).to_numpy()
    return result


def gini_coefficient(values: pd.Series) -> float:
    """Gini coefficient of a distribution of non-negative values (0..1).

    0 = every author has the same output; close to 1 = output is concentrated
    in very few authors. Standard mean-absolute-difference formulation, no
    external stats dependency needed. Returns 0.0 for fewer than 2 authors or
    an all-zero series (nothing to be unequal about).
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    arr = arr[arr >= 0]
    n = arr.size
    if n < 2 or arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    index = np.arange(1, n + 1, dtype="float64")
    return float((2 * (index * sorted_arr).sum() / (n * sorted_arr.sum())) - (n + 1) / n)


def lorenz_curve(values: pd.Series) -> pd.DataFrame:
    """Cumulative share of output vs. cumulative share of authors, sorted ascending.

    Includes the (0, 0) origin point. `charts.lorenz_chart` plots this against
    the perfect-equality diagonal as a second real trace (never a reference
    line, per the chart contract).
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype="float64")
    arr = np.sort(arr[arr >= 0])
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return pd.DataFrame({"share_of_authors": [0.0], "share_of_output": [0.0]})

    cum_output = arr.cumsum() / arr.sum()
    cum_authors = np.arange(1, n + 1, dtype="float64") / n
    return pd.DataFrame(
        {
            "share_of_authors": [0.0, *cum_authors.tolist()],
            "share_of_output": [0.0, *cum_output.tolist()],
        }
    )


def author_productivity_trend(matrix: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Linear-trend classification of yearly output for the top `top_n` authors.

    Fits `count ~ year` with `numpy.polyfit` (degree 1) per author, over that
    author's own active years only (a career that ended in 2015 shouldn't be
    scored against years it has no data for). Classifies the slope as
    "growing" / "stable" / "falling" against a small fixed threshold (0.15
    articles/year) rather than a significance test -- most authors here have
    under 15 active years, too short a series for a p-value to be meaningful
    (the same reasoning `forecasting.py` uses to prefer plain regression over
    a heavier model). Sorted descending by `total`, matching the main table.
    """
    columns = [
        "author",
        "total",
        "first_year",
        "last_year",
        "active_years",
        "slope",
        "trend",
    ]
    if matrix.empty:
        return pd.DataFrame(columns=columns)

    year_cols = [c for c in matrix.columns if c not in ("author", *_MATRIX_SUMMARY_COLS)]
    top = matrix.sort_values("total", ascending=False).head(top_n)

    rows = []
    for _, row in top.iterrows():
        active = [(int(y), row[y]) for y in year_cols if row[y] > 0]
        if len(active) < 2:
            first_year = active[0][0] if active else None
            rows.append(
                {
                    "author": row["author"],
                    "total": int(row["total"]),
                    "first_year": first_year,
                    "last_year": first_year,
                    "active_years": len(active),
                    "slope": 0.0,
                    "trend": "insufficient data",
                }
            )
            continue
        # Build the FULL year range (first_year to last_year) filling gaps
        # with zeros so that periods of inactivity count against the slope
        # instead of being silently omitted from the regression.
        active_map = dict(active)
        first_y, last_y = min(active_map), max(active_map)
        full_years = list(range(first_y, last_y + 1))
        full_counts = [active_map.get(y, 0) for y in full_years]
        slope = float(_linear_slope(full_years, full_counts))
        if slope > 0.15:
            trend = "growing"
        elif slope < -0.15:
            trend = "falling"
        else:
            trend = "stable"
        rows.append(
            {
                "author": row["author"],
                "total": int(row["total"]),
                "first_year": first_y,
                "last_year": last_y,
                "active_years": len(active),
                "slope": round(slope, 3),
                "trend": trend,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("total", ascending=False)


def _linear_slope(x: list[int], y: list[int]) -> float:
    """`numpy.polyfit` degree-1 slope, isolated for testability."""
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def output_impact_correlation(
    author_rows: pd.DataFrame, impact_col: str = "citation_count"
) -> dict[str, float | int | None]:
    """Pearson and Spearman correlation between an author's total output and
    their mean `impact_col`, plus the sample size actually used.

    Rows with a null `impact_col` are dropped before averaging -- per this
    corpus's documented data pitfall, null means "not collected", not zero
    impact, and must not be averaged in as 0. Returns `None`s when fewer than
    3 authors have usable data (a correlation over 1-2 points is noise).
    """
    result: dict[str, float | int | None] = {"pearson": None, "spearman": None, "n": 0}
    if impact_col not in author_rows.columns or "author_display" not in author_rows.columns:
        return result

    by_author = author_rows.groupby("author_display").agg(
        articles=("author_display", "size"), mean_impact=(impact_col, "mean")
    )
    by_author = by_author.dropna(subset=["mean_impact"])
    result["n"] = int(len(by_author))
    if len(by_author) < 3:
        return result

    result["pearson"] = float(
        by_author["articles"].corr(by_author["mean_impact"], method="pearson")
    )
    result["spearman"] = float(
        by_author["articles"].corr(by_author["mean_impact"], method="spearman")
    )
    return result


def coauthorship_community_detection(graph) -> dict[str, int]:
    """Detect research communities in a coauthorship network using Louvain modularity."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}
    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        node_comm = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                node_comm[node] = comm_id
        return node_comm
    except Exception:
        return {
            node: idx for idx, comp in enumerate(nx.connected_components(graph)) for node in comp
        }


def graph_advanced_metrics(graph) -> dict:
    """Calculate betweenness, closeness, pagerank, clustering, density, and small-world metrics."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {
            "density": 0.0,
            "avg_clustering": 0.0,
            "betweenness": {},
            "closeness": {},
            "pagerank": {},
            "degree": {},
            "avg_path_length": None,
            "small_world_sigma": None,
        }
    # In coauthorship networks, edge weight = collaboration intensity (higher
    # means closer).  Betweenness and closeness treat weight as *distance*, so
    # we invert: distance = 1 / intensity.  PageRank treats weight as strength
    # (no inversion needed).
    dist_graph = graph.copy()
    for _u, _v, d in dist_graph.edges(data=True):
        w = d.get("weight", 1)
        d["distance"] = 1.0 / w if w > 0 else 1.0
    betweenness = nx.betweenness_centrality(dist_graph, weight="distance")
    closeness = nx.closeness_centrality(dist_graph, distance="distance")
    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except Exception:
        pagerank = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}

    degree = dict(graph.degree(weight="weight"))
    density = float(nx.density(graph))
    avg_clustering = float(nx.average_clustering(graph, weight="weight"))

    # Small-world analysis on the largest connected component (LCC)
    avg_path_length = None
    small_world_sigma = None
    if graph.number_of_nodes() >= 4 and graph.number_of_edges() >= 3:
        components = list(nx.connected_components(graph))
        if components:
            largest_cc = max(components, key=len)
            sub_lcc = graph.subgraph(largest_cc)
            if sub_lcc.number_of_nodes() >= 3:
                try:
                    avg_path_length = float(nx.average_shortest_path_length(sub_lcc))
                    n_lcc = sub_lcc.number_of_nodes()
                    m_lcc = sub_lcc.number_of_edges()
                    k_mean = (2.0 * m_lcc) / n_lcc
                    if k_mean > 1.0 and n_lcc > k_mean:
                        c_rand = k_mean / n_lcc
                        l_rand = np.log(n_lcc) / np.log(k_mean)
                        c_actual = float(nx.average_clustering(sub_lcc))
                        if c_rand > 0 and l_rand > 0 and avg_path_length > 0:
                            gamma = c_actual / c_rand
                            lambda_val = avg_path_length / l_rand
                            if lambda_val > 0:
                                small_world_sigma = float(gamma / lambda_val)
                except Exception:
                    pass

    return {
        "density": density,
        "avg_clustering": avg_clustering,
        "betweenness": betweenness,
        "closeness": closeness,
        "pagerank": pagerank,
        "degree": degree,
        "avg_path_length": avg_path_length,
        "small_world_sigma": small_world_sigma,
    }


def network_null_model_diagnostics(graph, *, n_simulations: int = 100, seed: int = 42) -> dict:
    """Compare clustering with degree-preserving rewired null networks."""
    import networkx as nx

    if graph.number_of_nodes() < 4 or graph.number_of_edges() < 3:
        return {"valid": False, "reason": "insufficient_network"}
    observed = float(nx.average_clustering(graph))
    null_values: list[float] = []
    null_assortativity: list[float] = []
    rng = np.random.default_rng(seed)
    swaps = max(graph.number_of_edges() * 5, 1)
    for _ in range(max(1, n_simulations)):
        candidate = nx.Graph(graph)
        try:
            nx.double_edge_swap(
                candidate,
                nswap=swaps,
                max_tries=max(swaps * 20, 100),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            continue
        null_values.append(float(nx.average_clustering(candidate)))
        null_assortativity.append(float(nx.degree_assortativity_coefficient(candidate)))
    if not null_values:
        return {"valid": False, "reason": "rewiring_failed"}
    null_array = np.asarray(null_values)
    null_std = float(null_array.std(ddof=1)) if len(null_array) > 1 else 0.0

    # Assortativity rides along on the same rewired ensemble: the null is
    # already built, and a bare assortativity coefficient says nothing without
    # one -- degree sequence alone forces some of it.
    observed_assortativity = float(nx.degree_assortativity_coefficient(graph))
    assort_array = np.asarray([value for value in null_assortativity if np.isfinite(value)])
    assort_std = float(assort_array.std(ddof=1)) if len(assort_array) > 1 else 0.0
    assort_mean = float(assort_array.mean()) if len(assort_array) else float("nan")

    return {
        "valid": True,
        "observed_clustering": observed,
        "null_mean": float(null_array.mean()),
        "null_std": null_std,
        "z_score": (observed - float(null_array.mean())) / null_std if null_std > 0 else None,
        "empirical_p_value": float((1 + np.sum(null_array >= observed)) / (len(null_array) + 1)),
        "simulations": len(null_array),
        "observed_assortativity": observed_assortativity
        if np.isfinite(observed_assortativity)
        else None,
        "assortativity_null_mean": assort_mean if np.isfinite(assort_mean) else None,
        "assortativity_z_score": (
            (observed_assortativity - assort_mean) / assort_std
            if assort_std > 0 and np.isfinite(observed_assortativity) and np.isfinite(assort_mean)
            else None
        ),
        **_robustness_under_removal(graph),
    }


def _robustness_under_removal(graph, *, fraction: float = 0.1, seed: int = 42) -> dict:
    """How much of the network survives losing its hubs versus random nodes.

    A collaboration network held together by a few hub authors fragments under
    targeted removal while barely noticing random loss. The gap between the two
    is the structural claim; either number alone is just a graph size.
    """
    import networkx as nx

    total = graph.number_of_nodes()
    remove = max(1, int(round(total * fraction)))
    if total - remove < 2:
        return {"robustness_targeted": None, "robustness_random": None, "robustness_removed": 0}

    def giant_share(candidate) -> float:
        components = list(nx.connected_components(candidate))
        return float(max(len(c) for c in components) / total) if components else 0.0

    hubs = [node for node, _ in sorted(graph.degree, key=lambda kv: -kv[1])[:remove]]
    targeted = graph.copy()
    targeted.remove_nodes_from(hubs)

    rng = np.random.default_rng(seed)
    random_shares = []
    for _ in range(20):
        victim = graph.copy()
        victim.remove_nodes_from(rng.choice(list(graph.nodes), size=remove, replace=False).tolist())
        random_shares.append(giant_share(victim))

    return {
        "robustness_targeted": giant_share(targeted),
        "robustness_random": float(np.mean(random_shares)),
        "robustness_removed": remove,
    }


def periodized_collaboration_ties(author_rows: pd.DataFrame, *, n_periods: int = 3) -> pd.DataFrame:
    """Split coauthor ties into periods and count new versus repeated ones.

    The existing recurrent-edge count is static: it says how many pairs ever
    published twice, which cannot distinguish a field that keeps recruiting new
    collaborators from one that has closed into fixed teams. Splitting by the
    year a tie first appears is what makes that visible.
    """
    required = {"doi", "year"}
    author_column = "author_display" if "author_display" in author_rows.columns else "author"
    if author_rows.empty or not required.issubset(author_rows.columns):
        return pd.DataFrame()

    rows = author_rows.dropna(subset=["doi", "year", author_column]).copy()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce")
    rows = rows.dropna(subset=["year"])
    if rows.empty:
        return pd.DataFrame()

    # Every unordered author pair on each paper, with that paper's year.
    ties: dict[tuple[str, str], list[int]] = {}
    for (_doi, year), group in rows.groupby(["doi", "year"]):
        names = sorted({str(name) for name in group[author_column]})
        if len(names) < 2:
            continue
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                ties.setdefault((left, right), []).append(int(year))
    if not ties:
        return pd.DataFrame()

    years = np.sort(rows["year"].unique())
    if len(years) < n_periods:
        n_periods = max(1, len(years))
    edges = np.array_split(years, n_periods)

    records = []
    for block in edges:
        if not len(block):
            continue
        low, high = int(block.min()), int(block.max())
        new_ties = 0
        repeated = 0
        for appearances in ties.values():
            within = [y for y in appearances if low <= y <= high]
            if not within:
                continue
            # "New" means the pair's first-ever collaboration falls in this
            # period; anything earlier makes it a returning partnership.
            if min(appearances) >= low:
                new_ties += 1
            else:
                repeated += 1
        total = new_ties + repeated
        records.append(
            {
                "period": f"{low}–{high}",
                "new_ties": new_ties,
                "repeated_ties": repeated,
                "total_ties": total,
                "new_share": float(new_ties / total) if total else float("nan"),
            }
        )
    return pd.DataFrame(records)


def analyze_coauthorship_partners(
    graph,
    author_rows: pd.DataFrame,
    communities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Analyze unique vs. recurrent coauthorships for authors in the network and corpus.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph of top authors (nodes are author display names, edges have 'weight').
    author_rows : pd.DataFrame
        DataFrame with at least 'author_display' (or 'author') and 'doi' columns.
    communities : dict[str, int] | None
        Optional mapping of author -> community id. If None, Louvain communities are computed.

    Returns
    -------
    pd.DataFrame
        DataFrame with detailed breakdown of unique and recurrent collaborators
        both within the top-authors network subgraph and across the global corpus.
    """
    import collections

    columns = [
        "author",
        "articles",
        "community",
        "community_id",
        "network_unique_count",
        "network_recurrent_count",
        "network_recurrent_names",
        "network_occasional_count",
        "network_occasional_names",
        "global_unique_count",
        "global_recurrent_count",
        "global_recurrent_names",
        "global_top_partners",
    ]

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    if communities is None:
        communities = coauthorship_community_detection(graph)

    author_col = "author_display" if "author_display" in author_rows.columns else "author"
    has_author_col = author_col in author_rows.columns
    has_doi = "doi" in author_rows.columns

    # Pre-aggregate global collaborations across the corpus
    doi_to_authors: dict[str, set[str]] = collections.defaultdict(set)
    author_to_dois: dict[str, set[str]] = collections.defaultdict(set)
    if has_author_col and has_doi:
        valid = author_rows[[author_col, "doi"]].dropna()
        for a, d in zip(valid[author_col].astype(str), valid["doi"].astype(str), strict=True):
            doi_to_authors[d].add(a)
            author_to_dois[a].add(d)

    rows = []
    for u in graph.nodes():
        # --- 1. In-network metrics ---
        neighbors = list(graph.neighbors(u))
        network_unique_count = len(neighbors)

        # Recurrent in network (weight >= 2)
        recurrent_net = [
            (v, int(graph[u][v].get("weight", 1)))
            for v in neighbors
            if graph[u][v].get("weight", 1) >= 2
        ]
        recurrent_net.sort(key=lambda x: (-x[1], x[0]))
        network_recurrent_count = len(recurrent_net)
        network_recurrent_names = (
            ", ".join(f"{v} ({w})" for v, w in recurrent_net) if recurrent_net else "—"
        )

        # Occasional in network (weight == 1)
        occasional_net = sorted([v for v in neighbors if graph[u][v].get("weight", 1) == 1])
        network_occasional_count = len(occasional_net)
        network_occasional_names = ", ".join(occasional_net) if occasional_net else "—"

        # Community info (1-indexed)
        comm_idx = communities.get(u, 0) + 1
        comm_label = f"#{comm_idx}"

        # --- 2. Global corpus metrics ---
        u_dois = author_to_dois.get(u, set())
        articles_count = len(u_dois)

        partner_counts: dict[str, int] = collections.defaultdict(int)
        for d in u_dois:
            for coauthor in doi_to_authors.get(d, set()):
                if coauthor != u:
                    partner_counts[coauthor] += 1

        global_unique_count = len(partner_counts)
        recurrent_glob = {k: v for k, v in partner_counts.items() if v >= 2}
        global_recurrent_count = len(recurrent_glob)

        sorted_recurrent_glob = sorted(recurrent_glob.items(), key=lambda x: (-x[1], x[0]))
        global_recurrent_names = (
            ", ".join(f"{k} ({v})" for k, v in sorted_recurrent_glob[:5])
            if sorted_recurrent_glob
            else "—"
        )

        sorted_all_glob = sorted(partner_counts.items(), key=lambda x: (-x[1], x[0]))
        global_top_partners = (
            ", ".join(f"{k} ({v})" for k, v in sorted_all_glob[:5]) if sorted_all_glob else "—"
        )

        rows.append(
            {
                "author": u,
                "articles": articles_count,
                "community": comm_label,
                "community_id": comm_idx,
                "network_unique_count": network_unique_count,
                "network_recurrent_count": network_recurrent_count,
                "network_recurrent_names": network_recurrent_names,
                "network_occasional_count": network_occasional_count,
                "network_occasional_names": network_occasional_names,
                "global_unique_count": global_unique_count,
                "global_recurrent_count": global_recurrent_count,
                "global_recurrent_names": global_recurrent_names,
                "global_top_partners": global_top_partners,
            }
        )

    res_df = pd.DataFrame(rows, columns=columns)
    if not res_df.empty:
        res_df = res_df.sort_values(
            by=["articles", "network_unique_count", "network_recurrent_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return res_df


def author_impact_advanced_indices(df: pd.DataFrame, min_papers: int = 2) -> pd.DataFrame:
    """Compute h-index, Egghe's g-index, Zhang's e-index, and i10-index for researchers.

    - h-index: max h such that h papers have >= h citations.
    - g-index (Egghe 2006): largest g such that top g papers accumulate >= g^2 citations.
    - e-index (Zhang 2009): sqrt(sum(c_i - h) for i in top h papers) measuring excess citations beyond h-core.
    - i10-index: count of papers with >= 10 citations.
    """
    if df.empty or "authors" not in df.columns:
        return pd.DataFrame()

    author_citations: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        raw_c = row.get("citation_count")
        cites = int(raw_c) if pd.notna(raw_c) else 0
        authors = row.get("authors")
        if isinstance(authors, list):
            for a in authors:
                a_clean = str(a).strip()
                if a_clean:
                    author_citations.setdefault(a_clean, []).append(cites)

    records = []
    for author, cites_list in author_citations.items():
        if len(cites_list) < min_papers:
            continue
        cites_sorted = sorted(cites_list, reverse=True)
        n = len(cites_sorted)

        # h-index
        h = 0
        for i, c in enumerate(cites_sorted):
            if c >= i + 1:
                h = i + 1
            else:
                break

        # g-index (Egghe)
        cumsum = np.cumsum(cites_sorted)
        g = 0
        for i in range(n):
            if cumsum[i] >= (i + 1) ** 2:
                g = i + 1

        # e-index (Zhang excess citations)
        e2 = sum(cites_sorted[i] - h for i in range(h)) if h > 0 else 0
        e = float(np.sqrt(max(0, e2)))

        # i10-index
        i10 = sum(1 for c in cites_sorted if c >= 10)

        total_c = sum(cites_sorted)
        mean_c = total_c / n if n > 0 else 0.0

        records.append(
            {
                "author": author,
                "papers": n,
                "total_citations": total_c,
                "mean_citations": round(mean_c, 1),
                "h_index": h,
                "g_index": g,
                "e_index": round(e, 2),
                "i10_index": i10,
                "g_h_diff": g - h,
            }
        )

    return pd.DataFrame(records).sort_values(by=["g_index", "h_index"], ascending=[False, False])


def author_m_quotient_analysis(
    df: pd.DataFrame, *, observation_year: int, min_papers: int = 2
) -> pd.DataFrame:
    """Compute career-length normalized impact (Hirsch's m-quotient = h / career_years).

    Distinguishes rapidly emerging high-velocity researchers from established veterans.
    """
    if df.empty or "authors" not in df.columns:
        return pd.DataFrame()

    author_records: dict[str, dict] = {}

    for _, row in df.iterrows():
        raw_c = row.get("citation_count")
        cites = int(raw_c) if pd.notna(raw_c) else 0
        raw_y = row.get("year")
        y = int(raw_y) if pd.notna(raw_y) else 2015
        authors = row.get("authors")
        if isinstance(authors, list):
            for a in authors:
                a_clean = str(a).strip()
                if a_clean:
                    rec = author_records.setdefault(a_clean, {"cites": [], "years": []})
                    rec["cites"].append(cites)
                    rec["years"].append(y)

    rows = []
    for author, data in author_records.items():
        c_list = data["cites"]
        if len(c_list) < min_papers:
            continue
        c_sorted = sorted(c_list, reverse=True)
        h = 0
        for i, c in enumerate(c_sorted):
            if c >= i + 1:
                h = i + 1
            else:
                break

        first_year = min(data["years"])
        career_span = max(1, observation_year - first_year + 1)
        m_quotient = round(h / career_span, 2)

        rows.append(
            {
                "author": author,
                "papers": len(c_list),
                "total_citations": sum(c_list),
                "h_index": h,
                "first_year": first_year,
                "career_span_years": career_span,
                "m_quotient": m_quotient,
            }
        )

    m_df = pd.DataFrame(rows).sort_values(by="m_quotient", ascending=False)
    return m_df
