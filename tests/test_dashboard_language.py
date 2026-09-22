from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

DASHBOARD_ROOT = Path("src/lake_research_map/dashboard")
# Nouns and adjectives alone were not enough: the leftover that survived the
# first sweep was "não como identidade confirmada", whose every word is a
# common Portuguese function word rather than a domain term. The closed-class
# words below (pronouns, prepositions, copulas) are what actually distinguish a
# Portuguese sentence from an English one carrying a Portuguese noun.
PORTUGUESE_UI_WORDS = re.compile(
    r"\b(?:arquivo|busca|campo|cobertura|complexidade|curada|desde|equidade|erro|"
    r"etapas|ferramental|horizonte|incerteza|modelagem|paradigma|preenchimento|"
    r"recomendado|severidade|similaridade|vetorial|justificativa|atualizado|"
    r"como|identidade|confirmada|confirmado|nao|não|sao|são|está|estão|pelo|pela|"
    r"apenas|ainda|cada|mais|menos|entre|sobre|quando|onde|porque|foram|"
    r"deve|pode|seja|sem|muito|todos|todas)\b",
    re.IGNORECASE,
)
EMPTY_HEADING = re.compile(r"st\.(?:header|subheader)\(\s*(['\"])\1\s*\)")

# `metric_row` takes (label, value[, delta]) tuples and passes the label
# straight to `st.metric`, so an empty first element renders a number with no
# name on it. That is the same NFR-07 failure as an empty heading, but it
# cannot be seen by looking for `st.metric` -- the call site is a tuple.
LABELLED_TUPLE_CALLS = {"metric_row"}


def _python_sources() -> list[Path]:
    # qualis.py must retain the official CAPES column names verbatim.
    return sorted(path for path in DASHBOARD_ROOT.rglob("*.py") if path.name != "qualis.py")


def test_dashboard_source_has_no_portuguese_ui_copy_or_empty_headings():
    failures = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        for match in PORTUGUESE_UI_WORDS.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            failures.append(f"{path}:{line}: {match.group(0)!r}")
        if match := EMPTY_HEADING.search(source):
            line = source.count("\n", 0, match.start()) + 1
            failures.append(f"{path}:{line}: empty Streamlit heading")
        failures.extend(_empty_metric_labels(path, source))
    assert not failures, "\n".join(failures)


def _empty_metric_labels(path: Path, source: str) -> list[str]:
    """Find `metric_row([... ("", value, delta) ...])` entries."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        if name not in LABELLED_TUPLE_CALLS:
            continue
        for argument in node.args:
            entries = argument.elts if isinstance(argument, ast.List | ast.Tuple) else []
            for entry in entries:
                if not isinstance(entry, ast.Tuple) or not entry.elts:
                    continue
                label = entry.elts[0]
                if isinstance(label, ast.Constant) and label.value == "":
                    found.append(f"{path}:{entry.lineno}: empty metric label")
    return found


def test_adjacent_dashboard_strings_do_not_start_a_lowercase_clause_after_period():
    failures = []
    ignored = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
    for path in _python_sources():
        previous = None
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in ignored:
                continue
            if token.type == tokenize.STRING and previous and previous.type == tokenize.STRING:
                try:
                    left = ast.literal_eval(previous.string)
                    right = ast.literal_eval(token.string)
                except (SyntaxError, ValueError):
                    pass
                else:
                    if (
                        isinstance(left, str)
                        and isinstance(right, str)
                        and left.endswith(". ")
                        and right[:1].islower()
                    ):
                        failures.append(f"{path}:{token.start[0]}: lowercase clause {right[:40]!r}")
            previous = token
    assert not failures, "\n".join(failures)
