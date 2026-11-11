# FastAPI prediction service (slim: no torch)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ ./src/
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
# 1 worker: prometheus-fastapi-instrumentator metrics are per-process
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
