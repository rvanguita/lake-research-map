"""Developer tools that are not part of the pipeline.

The hero-image generator and the pre-commit guard live here so the repository
has one source tree. Nothing in the pipeline imports this package, and
`ingest/snapshots.py::PRESENTATION_ONLY` excludes it from the code fingerprint:
editing a developer tool must not mint a new dataset version.
"""
