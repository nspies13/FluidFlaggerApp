#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-fluidflagger}"
TAG="${TAG:-1.0.0}"
IMAGE="${IMAGE_NAME}:${TAG}"

cd "$(dirname "$0")/.."

docker buildx build --platform linux/amd64 --load -t "$IMAGE" .

USER_VALUE="$(docker image inspect "$IMAGE" --format '{{.Config.User}}')"
PORTS_VALUE="$(docker image inspect "$IMAGE" --format '{{json .Config.ExposedPorts}}')"
ARCH_VALUE="$(docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}')"

test -n "$USER_VALUE"
test "$USER_VALUE" != "0"
test "$USER_VALUE" != "root"
test "$PORTS_VALUE" = '{"8080/tcp":{}}'
test "$ARCH_VALUE" = "linux/amd64"

echo "Built and verified $IMAGE"
echo "User=$USER_VALUE Ports=$PORTS_VALUE Arch=$ARCH_VALUE"
