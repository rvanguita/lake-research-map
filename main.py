"""Root entry point for the Streamlit dashboard.

    uv run streamlit run main.py

`app.main()` is called as a function so it re-executes on every Streamlit
rerun. (Importing a module for its top-level side effects would only render on
the first run -- Python caches imports, so every later rerun would no-op.)
"""

from apps.dashboard.app import main

main()
