FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Keep dependency layers small enough for registry uploads to complete through
# the Navify gateway. Versions remain pinned in requirements.txt.
RUN pip install --no-cache-dir --no-compile \
    fastapi==0.116.1 \
    "uvicorn[standard]==0.35.0"
RUN pip install --no-cache-dir --no-compile \
    numpy==2.2.6
RUN pip install --no-cache-dir --no-compile \
    scipy==1.14.1
RUN pip install --no-cache-dir --no-compile \
    pandas==2.3.2
RUN pip install --no-cache-dir --no-compile \
    scikit-learn==1.8.0 \
    joblib==1.4.2
RUN pip install --no-cache-dir --no-compile \
    lightgbm==4.6.0

COPY src/ ./src/
# Copy each model family independently so Navify's registry gateway never has
# to receive all compressed model artifacts in one upload.
COPY models/bmp_D5LR_* ./models/
COPY models/bmp_D5NS_* ./models/
COPY models/bmp_D5W_* ./models/
COPY models/bmp_D5halfNS_* ./models/
COPY models/bmp_D5halfNSwK_* ./models/
COPY models/bmp_LR_* ./models/
COPY models/bmp_NS_* ./models/
COPY models/bmp_Water_* ./models/
COPY models/bmp_halfNS_* ./models/
COPY data/ ./data/
COPY checksums/ ./checksums/
COPY scripts/verify_models.py ./scripts/verify_models.py

RUN python scripts/verify_models.py

RUN groupadd -g 1000 appuser \
    && useradd -u 1000 -g appuser -s /usr/sbin/nologin -M appuser

USER 1000
EXPOSE 8080

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info", "--timeout-keep-alive", "120"]
