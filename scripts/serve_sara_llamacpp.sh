#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Reserve GPU 3 for Sara; use GPUs 0–2 for Boltz while this server is running.
export CUDA_VISIBLE_DEVICES="${SARA_GPU:-3}"
exec .cache/llama.cpp/build/bin/llama-server \
  --model .cache/models/Qwen3.5-9B-Q4_K_M.gguf \
  --alias qwen3.5:9b --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 1 --threads 6 \
  --batch-size 256 --ubatch-size 128 --n-gpu-layers 99 \
  --flash-attn off --jinja --reasoning on --reasoning-budget 256 "$@"
