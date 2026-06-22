web: uv run uvicorn app.server:app --host 0.0.0.0 --port $PORT
worker: uv run python -m app.agent start
combined: uv run python -m app.combined start
