# Pinned by digest so a rebuild gets the same base, not whatever 3.13-slim points
# at that day; bump it deliberately (`docker pull python:3.13-slim` + inspect).
FROM python:3.13-slim@sha256:09a17cad02668ae61ece136b4fd36ac3849e14f670ebedb637ea3e15580525bc

# Same uv as the one that wrote uv.lock, so `--frozen` reads it the same way.
RUN pip install --no-cache-dir uv==0.13.7

# The dashboard runs unprivileged. /app and the model cache (the compose
# `model_cache` volume mounts on /opt/model-cache; an empty named volume takes
# the ownership of the directory it is mounted over) belong to that user.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app /opt/model-cache \
    && chown app:app /app /opt/model-cache

WORKDIR /app
USER app
# No uv cache in the image: it would only duplicate the wheels already in .venv.
ENV UV_NO_CACHE=1

# Dependencies first, in a layer of their own: it depends only on the lockfile,
# so editing the code reuses it instead of reinstalling every wheel.
# `--no-install-project` skips building this project, which is what needs src/
# and README.md (hatchling reads the readme into the package metadata).
COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the project itself; only this layer and the ones below rebuild on a code change.
COPY --chown=app:app README.md main.py ./
COPY --chown=app:app src ./src
RUN uv sync --frozen --no-dev

# The fixed dark theme and `gatherUsageStats = false` (PRD NFR-07). Streamlit
# reads `.streamlit/config.toml` from the working directory; without it the
# container rendered in Streamlit's default theme and sent usage statistics.
COPY --chown=app:app .streamlit/config.toml ./.streamlit/config.toml

ENV PATH="/app/.venv/bin:$PATH"
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# `--server.address` outranks `.streamlit/config.toml`'s loopback address: inside
# the container the port must accept the published connection; compose publishes
# it on the host's 127.0.0.1 only.
ENTRYPOINT ["streamlit", "run", "main.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
