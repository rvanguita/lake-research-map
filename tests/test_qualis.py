import pandas as pd

from apps.dashboard.qualis import load_qualis_reference, match_venues_to_qualis


def _qualis_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "issn": "0142-0615",
                "titulo": "INTERNATIONAL JOURNAL OF ELECTRICAL POWER & ENERGY SYSTEMS",
                "estrato": "A1",
            },
            {
                "issn": "0378-7796",
                "titulo": "ELECTRIC POWER SYSTEMS RESEARCH (PRINT)",
                "estrato": "A2",
            },
            {"issn": "0960-1481", "titulo": "RENEWABLE ENERGY", "estrato": "A1"},
        ]
    )


def test_match_venues_to_qualis_exact_match():
    result = match_venues_to_qualis(["Renewable Energy"], _qualis_df()).set_index("venue")
    assert result.loc["Renewable Energy", "estrato"] == "A1"
    assert result.loc["Renewable Energy", "score"] == 100.0


def test_match_venues_to_qualis_handles_ampersand_and_suffix_variants():
    result = match_venues_to_qualis(
        [
            "International Journal of Electrical Power and Energy Systems",
            "Electric Power Systems Research",
        ],
        _qualis_df(),
    ).set_index("venue")

    assert (
        result.loc["International Journal of Electrical Power and Energy Systems", "estrato"]
        == "A1"
    )
    assert result.loc["Electric Power Systems Research", "estrato"] == "A2"


def test_match_venues_to_qualis_unmatched_venue_is_not_a_guess():
    result = match_venues_to_qualis(["Journal of Completely Unrelated Topic"], _qualis_df())
    row = result.iloc[0]
    assert row["estrato"] is None
    assert row["matched_title"] is None


def test_match_venues_to_qualis_empty_reference_returns_unclassified():
    empty_ref = pd.DataFrame(columns=["titulo", "estrato"])
    result = match_venues_to_qualis(["Renewable Energy"], empty_ref)
    assert result.iloc[0]["estrato"] is None


def test_load_qualis_reference_accepts_prefiltered_export_without_area_column(
    tmp_path, monkeypatch
):
    workbook = tmp_path / "capes.xlsx"
    workbook.touch()
    source = pd.DataFrame([{"ISSN": "1234-5678", "Título": "Renewable Energy", "Estrato": "A1"}])
    monkeypatch.setattr("pandas.read_excel", lambda *args, **kwargs: source)

    result = load_qualis_reference(workbook)

    assert result.to_dict("records") == [
        {"issn": "1234-5678", "titulo": "Renewable Energy", "estrato": "A1"}
    ]


def test_load_qualis_reference_filters_full_export_by_area(tmp_path, monkeypatch):
    workbook = tmp_path / "capes.xlsx"
    workbook.touch()
    source = pd.DataFrame(
        [
            {
                "ISSN": "1234-5678",
                "Título": "Renewable Energy",
                "Estrato": "A1",
                "Área de Avaliação": "ENGENHARIAS IV",
            },
            {
                "ISSN": "9876-5432",
                "Título": "Other Journal",
                "Estrato": "B4",
                "Área de Avaliação": "MEDICINA II",
            },
        ]
    )
    monkeypatch.setattr("pandas.read_excel", lambda *args, **kwargs: source)

    result = load_qualis_reference(workbook)

    assert result["issn"].tolist() == ["1234-5678"]


def _tree(root) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def test_missing_workbook_returns_an_empty_reference_and_writes_nothing(tmp_path):
    # The dashboard container ships without `data/`: a missing workbook is an
    # expected state, not a FileNotFoundError.
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = load_qualis_reference(data_dir / "capes-qualis.xlsx")

    assert result.empty
    assert list(result.columns) == ["issn", "titulo", "estrato"]
    assert _tree(tmp_path) == {"data"}


def test_reading_a_workbook_writes_no_cache_next_to_it(tmp_path):
    # `data/` is read-only input: the parsed subset is cached in memory by
    # `loaders._qualis_reference`, never as a file beside the source.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workbook = data_dir / "capes-qualis.xlsx"
    pd.DataFrame([{"ISSN": "1234-5678", "Título": "Renewable Energy", "Estrato": "A1"}]).to_excel(
        workbook, sheet_name="RelatorioQualis", index=False
    )
    before = _tree(tmp_path)

    result = load_qualis_reference(workbook)

    assert result["estrato"].tolist() == ["A1"]
    assert _tree(tmp_path) == before


def test_unavailable_reference_leaves_every_venue_unclassified():
    empty = load_qualis_reference("/nonexistent/capes-qualis.xlsx")

    result = match_venues_to_qualis(["Renewable Energy"], empty)

    assert result.iloc[0]["estrato"] is None
