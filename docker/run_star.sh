#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
set -euo pipefail

# Default X11 vars so GUI apps (e.g., rviz2) can authenticate to host display.
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/tmp/.Xauthority}"

BASE_LAYER_IMAGE="base-uv-dev:latest"  # <-- your base image with ROS, Gazebo, and other heavy dependencies pre-installed
BASE_LAYER_DOCKERFILE="Dockerfile.cuda128_humble_uv"  # <-- Dockerfile for the base layer (should be in project root)

IMAGE_NAME="star"
TAG="latest"
SERVICE_NAME="star_dev"                 # <-- your compose service name
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

if docker image inspect $BASE_LAYER_IMAGE >/dev/null 2>&1; then
    echo "base layer exists, skipping build"
else
    echo "base layer not found, rebuilding ${BASE_LAYER_IMAGE}..."
    cd "$SCRIPT_DIR"
    docker build -t "$BASE_LAYER_IMAGE" -f "$BASE_LAYER_DOCKERFILE" .
fi

# Check for --reset flag to determine if we should remove existing containers/volumes
RESET=false
REBUILD=false
if [[ "${1:-}" == "--reset" ]]; then
  echo "Running with --reset. Existing containers/volumes will be removed and recreated."
  RESET=true
elif [[ "${1:-}" == "--rebuild" ]]; then
  echo "Running with --rebuild. Existing containers/volumes will be reused, but image will be rebuilt."
  REBUILD=true
else
  echo "Existing containers/volumes will be reused if they exist."
fi

# Prefer Docker Compose v2 ("docker compose")
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose -f "$COMPOSE_FILE")
else
  echo "[ERROR] Docker Compose v2 not found. Install it with:"
  echo "  sudo apt-get update && sudo apt-get install -y docker-compose"
  exit 1
fi

echo "Configuration complete. Recreating container to apply latest mounts/env..."
if [[ "$REBUILD" == true ]]; then
  echo "Rebuilding image ${IMAGE_NAME}:${TAG}..."
  "${COMPOSE_CMD[@]}" build --no-cache
  "${COMPOSE_CMD[@]}" up -d
elif [[ "$RESET" == true ]]; then
  echo "Resetting containers/volumes for ${SERVICE_NAME}..."
  # Stop + remove existing containers for this compose project (does NOT remove images or named volumes)
  "${COMPOSE_CMD[@]}" down #--remove-orphans
  # Start / recreate with latest compose config
  "${COMPOSE_CMD[@]}" up -d --force-recreate
else
  # Build image if missing
  if [[ -z "$(docker images -q "${IMAGE_NAME}:${TAG}" 2>/dev/null)" ]]; then
    echo "Image ${IMAGE_NAME}:${TAG} not found locally — building now."
    "${COMPOSE_CMD[@]}" build
  else
    echo "Image ${IMAGE_NAME}:${TAG} already exists — skipping build."
  fi

  # Non-destructive: just start any stopped containers, but don't recreate if already running
  "${COMPOSE_CMD[@]}" up -d
fi

# Attach a shell (more reliable than docker exec with container_name)
"${COMPOSE_CMD[@]}" exec -it "$SERVICE_NAME" bash