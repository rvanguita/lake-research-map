"""Pure pandas aggregations shared by every dashboard page, one module per subject.

No `streamlit` import anywhere in this package on purpose -- these functions are
plain data transforms and can be exercised/cached independently of the UI layer.
Pages call these instead of open-coding a `groupby`/`explode`, so the same year
window, the same "count by source" shape, and the same author-name normalization
are used everywhere.

There are no re-exports: import from the submodule that owns the function
(`counts`, `authors`, `citations`, `bibliometrics`, `vocabulary`, `anomalies`).
The method taxonomy lives in `transform.taxonomy`, author identity in
`transform.authors`, and the semantic-stability diagnostics in
`transform.semantics`, because the core uses them to build a version.
"""
