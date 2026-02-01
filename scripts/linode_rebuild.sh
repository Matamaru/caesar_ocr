#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CAESAR_ENV_FILE:-}" ]]; then
  echo "Set CAESAR_ENV_FILE to the env file path, e.g. /etc/caesar-ocr.env" >&2
  exit 1
fi

echo "Stopping and removing containers..."
docker rm -f $(docker ps -aq) 2>/dev/null || true

echo "Removing images..."
docker rmi -f $(docker images -q) 2>/dev/null || true

echo "Pruning build cache and volumes..."
docker builder prune -af || true
docker volume prune -f || true

echo "Pulling latest code..."
git pull

echo "Building image..."
docker build -t caesar-ocr:latest .

echo "Starting container..."
docker run -d --name caesar-ocr \
  -p 127.0.0.1:8000:8000 \
  --env-file "$CAESAR_ENV_FILE" \
  caesar-ocr:latest

echo "Done."
