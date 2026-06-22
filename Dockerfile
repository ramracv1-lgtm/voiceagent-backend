FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Hugging Face Spaces routes traffic to the port declared as app_port in the Space's
# README frontmatter (set to 7860 below, HF's conventional default) — combined.py reads
# $PORT for the FastAPI side-car, so both have to agree on the same value.
ENV PORT=7860
EXPOSE 7860

CMD ["uv", "run", "python", "-m", "app.combined", "start"]
