#!/usr/bin/env bash
# Start the OpenAI-compatible vLLM server for the audio-LLM transcription backend
# (transcription.backend: llm). Needs the vLLM env: bash scripts/install.sh --vllm
#
#   bash scripts/serve_llm.sh                          # Qwen3-Omni-30B-A3B-Instruct, BF16 (>= 80 GB GPU)
#   QUANT=fp8 bash scripts/serve_llm.sh                # FP8 weights: fits 48 GB GPUs
#   MODEL=<hf id> PORT=8000 bash scripts/serve_llm.sh
#
# Then keep transcription.llm.url = http://localhost:$PORT/v1 in configs/config.yaml.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
MAX_SEQS="${MAX_SEQS:-64}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP="${TP:-1}"
if [ -z "${CONDA_DIR:-}" ]; then
  if [ -d /workspace/miniforge3 ]; then CONDA_DIR=/workspace/miniforge3; else CONDA_DIR="$HOME/miniforge3"; fi
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate egyspeech-vllm
[ -d /workspace ] && export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

ARGS=(serve "$MODEL" --port "$PORT" --dtype bfloat16 --max-model-len "$MAX_LEN"
      --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL" -tp "$TP"
      --limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}')
[ -n "${QUANT:-}" ] && ARGS+=(--quantization "$QUANT")
echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
