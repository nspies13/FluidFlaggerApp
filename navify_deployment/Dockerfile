FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY checksums/ ./checksums/
COPY scripts/verify_models.py ./scripts/verify_models.py

RUN python scripts/verify_models.py

RUN groupadd -g 1000 appuser \
    && useradd -u 1000 -g appuser -s /usr/sbin/nologin -M appuser \
    && chown -R appuser:appuser /app

USER 1000
EXPOSE 8080

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info", "--timeout-keep-alive", "120"]
