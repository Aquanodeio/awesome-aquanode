#!/usr/bin/env bash
set -xeuo pipefail

VERSION="${1:-0.3}"
IMAGE="vandit1604/flux-schnell-stablediff:$VERSION"

# Enable BuildKit explicitly (works with both local + CI)
export DOCKER_BUILDKIT=1

echo "[*] Building $IMAGE"
docker build -t "$IMAGE" .

echo "[*] Pushing $IMAGE"
docker push "$IMAGE"

