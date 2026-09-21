FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY main.py ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "main.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
