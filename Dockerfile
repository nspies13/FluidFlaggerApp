# ==========================================================================
# FluidFlagger — multi-target Dockerfile
#
# Targets (build with: docker build --target <name> .)
#   inference  — navify Algorithm Suite API (models baked in)
#   nomodel    — Gradio UI for Self-Test & Review (no models required)
#   train      — Model training job
#
# navify requirements (inference target):
#   1. Non-root user (UID 1000)
#   2. Exactly one EXPOSE port (8080)
#   3. Health probes: GET /health/ready, GET /health/live
#   4. Calculation endpoint: POST /predict


# ==========================================================================

# -- pip-api: install API-only deps ----------------------------------------
FROM python:3.11-slim AS pip-api

WORKDIR /build
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# -- pip-full: install all deps (adds Gradio + Optuna) ---------------------
FROM python:3.11-slim AS pip-full

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# -- os-base: shared runtime layer (system libs + user, no packages yet) ---
FROM python:3.11-slim AS os-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user (shared across all targets)
RUN groupadd -g 1000 appuser \
    && useradd -u 1000 -g appuser -s /usr/sbin/nologin -M appuser


# -- app-core: source code + data (no models, no pip packages) ------------
FROM os-base AS app-core

COPY src/ ./src/
COPY data/fluid_concentrations.tsv ./data/fluid_concentrations.tsv


# ==========================================================================
# TARGET: inference — navify Algorithm Suite API container
# ==========================================================================
FROM app-core AS inference

COPY --from=pip-api /install /usr/local
COPY models/ ./models/

RUN chown -R appuser:appuser /app
USER 1000

EXPOSE 8080

CMD ["uvicorn", "src.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--log-level", "info", \
     "--timeout-keep-alive", "120"]


# ==========================================================================
# TARGET: nomodel — Gradio UI (Self-Test & Review, no models needed)
# ==========================================================================
FROM app-core AS nomodel

COPY --from=pip-full /install /usr/local
COPY app.py .

RUN chown -R appuser:appuser /app
USER 1000

EXPOSE 7860

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info"]


# ==========================================================================
# TARGET: train — model training job
# ==========================================================================
FROM app-core AS train

COPY --from=pip-full /install /usr/local
COPY data/ ./data/

RUN chown -R appuser:appuser /app
USER 1000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]


# ==========================================================================
# TARGET: hf — Full Gradio app + models for Hugging Face Spaces deployment
# ==========================================================================
FROM app-core AS hf

COPY --from=pip-full /install /usr/local
COPY app.py .
COPY data/ ./data/

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('nickspies/fluidflagger-models', local_dir='./models')" \
    && chown -R appuser:appuser /app
USER 1000

EXPOSE 7860

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info"]
