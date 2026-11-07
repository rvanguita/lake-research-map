FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies first, in a layer of their own: it depends only on the lockfile,
# so editing the code reuses it instead of reinstalling every wheel.
# `--no-install-project` skips building this project, which is what needs src/
# and README.md (hatchling reads the readme into the package metadata).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the project itself; only this layer and the ones below rebuild on a code change.
COPY README.md main.py ./
COPY src ./src
RUN uv sync --frozen --no-dev

# The fixed dark theme and `gatherUsageStats = false` (PRD NFR-07). Streamlit
# reads `.streamlit/config.toml` from the working directory; without it the
# container rendered in Streamlit's default theme and sent usage statistics.
COPY .streamlit/config.toml ./.streamlit/config.toml

ENV PATH="/app/.venv/bin:$PATH"
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "main.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
