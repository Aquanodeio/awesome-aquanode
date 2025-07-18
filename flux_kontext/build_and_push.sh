#!/bin/bash

# Usage: ./build_and_push.sh 0.1
# If no tag passed, default to "latest"
TAG=${1:-latest}
IMAGE_NAME="arpbansal/flux-kontext:$TAG"

echo "🔨 Building image: $IMAGE_NAME"
docker build -t $IMAGE_NAME flux_kontext/

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Exiting."
    exit 1
fi

echo "🚀 Pushing image to Docker Hub: $IMAGE_NAME"
docker push $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed $IMAGE_NAME"
else
    echo "❌ Push failed."
    exit 1
fi
