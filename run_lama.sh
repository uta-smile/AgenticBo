#!/usr/bin/env bash

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$ROOT/.cache/models/Qwen3.5-9B-Q4_K_M.gguf"

echo "Project root: $ROOT"
echo "Model: $MODEL"

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

llama.exe serve \
  -m "$MODEL" \
  --alias "qwen3.5:9b" \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8080