#!/usr/bin/env bash
set -euo pipefail

: "${TENANT_ALIAS:?Set TENANT_ALIAS}"
: "${TOKEN:?Set TOKEN to a Navify bearer token}"

IMAGE_NAME="${IMAGE_NAME:-fluidflagger-bmp}"
TAG="${TAG:-1.0.0}"
REGISTRY="${REGISTRY:-acr.us.prod.algosuite.navify.com}"
LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
REMOTE_IMAGE="${REGISTRY}/${TENANT_ALIAS}/${IMAGE_NAME}:${TAG}"

cd "$(dirname "$0")/.."

echo "$TOKEN" | docker login "$REGISTRY" -u Navify --password-stdin

IMAGE_NAME="$IMAGE_NAME" TAG="$TAG" ./scripts/build_local.sh

docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"

echo "Pushed $REMOTE_IMAGE"
