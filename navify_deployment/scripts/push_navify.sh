#!/usr/bin/env bash
set -euo pipefail

: "${TENANT_ALIAS:?Set TENANT_ALIAS}"
: "${TOKEN:?Set TOKEN to a Navify bearer token}"

IMAGE_NAME="${IMAGE_NAME:-fluidflagger-bmp}"
TAG="${TAG:-1.0.0}"
REGISTRY="${REGISTRY:-acr.prod.algosuite.navify.com}"
REMOTE_IMAGE="${REGISTRY}/${TENANT_ALIAS}/${IMAGE_NAME}:${TAG}"

cd "$(dirname "$0")/.."

echo "$TOKEN" | docker login "$REGISTRY" -u Navify --password-stdin

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --output "type=image,name=${REMOTE_IMAGE},push=true,oci-mediatypes=true" \
  .

echo "Pushed $REMOTE_IMAGE"
