"""The composition root: the `lake-research-map` command line.

`cli.main` builds the parser, takes the pipeline lock for the commands that
change what the next publication would contain, and dispatches to the stage
runner (`pipeline.run`) or to one command group. Each group module exposes
`configure(subparsers)` and `run(args)`, as `apps.api.cli` and `apps.qdrant.cli`
do. This package is the only place outside `apps/` allowed to import an
application; nothing in the core imports it.
"""
