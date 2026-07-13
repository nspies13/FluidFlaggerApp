#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-fluidflagger-bmp}"
TAG="${TAG:-1.0.0}"
IMAGE="${IMAGE_NAME}:${TAG}"
CONTAINER_NAME="${CONTAINER_NAME:-fluidflagger-bmp-smoke}"
PORT="${PORT:-8080}"

cd "$(dirname "$0")/.."

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" -p "${PORT}:8080" "$IMAGE" >/dev/null

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${PORT}/health/ready" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "http://localhost:${PORT}/health/live" >/dev/null
curl -fsS "http://localhost:${PORT}/health/ready" >/dev/null

curl -fsS \
  -H "Content-Type: application/json" \
  --data-binary @examples/realtime_request.json \
  "http://localhost:${PORT}/predict" >/tmp/fluidflagger_realtime_response.json

curl -fsS \
  -H "Content-Type: application/json" \
  --data-binary @examples/retrospective_request.json \
  "http://localhost:${PORT}/predict" >/tmp/fluidflagger_retrospective_response.json

STATUS="$(curl -sS -o /tmp/fluidflagger_invalid_response.json -w '%{http_code}' \
  -H "Content-Type: application/json" \
  --data-binary @examples/invalid_partial_post_request.json \
  "http://localhost:${PORT}/predict")"

test "$STATUS" = "400"

echo "Smoke test passed for $IMAGE on localhost:$PORT"
