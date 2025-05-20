#!/bin/bash

# === USER CONTAINER LAUNCHER ===
GPU_ID=$1
USERNAME=$2

if [ -z "$GPU_ID" ] || [ -z "$USERNAME" ]; then
    echo "Usage: $0 <GPU_ID> <USERNAME>"
    exit 1
fi

# === Compute or generate port ===
BASE_PORT=11434
USER_INDEX=$(echo "$USERNAME" | grep -o '[0-9]*$')

if [ -n "$USER_INDEX" ]; then
    PORT=$((BASE_PORT + USER_INDEX))
    echo "ℹ️ Using calculated port $PORT for user $USERNAME (based on index $USER_INDEX)"
else
    # Generate a random free port in 20000–30000
    echo "🔀 No numeric suffix found. Assigning random available port for $USERNAME..."
    while true; do
        PORT=$(( ( RANDOM % 10000 ) + 20000 ))
        if ! lsof -i tcp:${PORT} >/dev/null 2>&1; then
            echo "✅ Assigned random free port $PORT"
            break
        fi
    done
fi

CONTAINER_NAME="ollama-${USERNAME}"

# === Check for existing container ===
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "⚠️  Container $CONTAINER_NAME already exists. Stopping and removing it..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
    echo "🗑️  Removed old container $CONTAINER_NAME"
fi

# === Check if port is free (again, for safety) ===
if lsof -i tcp:${PORT} | grep LISTEN; then
    echo "❌ Port $PORT is already in use. Cannot start container."
    exit 1
fi

# === Start container ===
echo "🚀 Starting container $CONTAINER_NAME on GPU $GPU_ID (port $PORT)..."
docker run -d --gpus "device=${GPU_ID}" \
  -p ${PORT}:11434 \
  -v /opt/ollama-models:/root/.ollama \
  -v /data/ICTA_workshop/data/images:/home/data \
  --name "$CONTAINER_NAME" \
  ollama/ollama

if [ $? -eq 0 ]; then
    echo "✅ Ollama container for $USERNAME started successfully on port $PORT."
else
    echo "❌ Failed to start Ollama container for $USERNAME."
fi
