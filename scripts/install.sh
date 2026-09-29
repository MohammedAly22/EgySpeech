#!/usr/bin/env bash
# EgySpeech environment installer (RunPod or any Linux + NVIDIA GPU).
#
#   bash scripts/install.sh            # main + nemo + qwen envs
#   bash scripts/install.sh --vllm     # also the vLLM env for the audio-LLM backend
#
# Installs Miniforge (conda) if needed — into /workspace/miniforge3 on RunPod (the
# persistent volume), else ~/miniforge3 — and creates:
#   egyspeech       download, separation, segmentation, quality, alignment, analysis, publish
#   egyspeech-nemo  Sortformer diarization, TitaNet speaker embeddings, Parakeet/FastConformer ASR
#   egyspeech-qwen  QwenCleo-ASR (default ASR backend)
#   egyspeech-vllm  (optional) vLLM server for the audio LLM (Qwen3-Omni)
# Re-running is safe: existing envs are updated in place.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_VLLM=0
for arg in "$@"; do
  case "$arg" in
    --vllm) WITH_VLLM=1 ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done

if [ -z "${CONDA_DIR:-}" ]; then
  if [ -d /workspace ]; then CONDA_DIR=/workspace/miniforge3; else CONDA_DIR="$HOME/miniforge3"; fi
fi

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo ">>> installing Miniforge into $CONDA_DIR"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
# One pip cache for all envs (torch is downloaded once), on the same volume as conda.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$CONDA_DIR")/.pip_cache}"

# ---- torch build for this GPU ------------------------------------------------
# torch 2.11 is the newest release with a matching torchaudio (needed by QwenCleo).
# Blackwell GPUs (compute capability >= 10: RTX 50xx, RTX PRO 4500/6000, B200 ...)
# need CUDA >= 12.8 builds, which need a recent driver.
TORCH_VERSION=2.11.0
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
CC="${CC:-0}"; DRIVER="${DRIVER:-0}"
if [ "$CC" -ge 10 ]; then
  if [ "$DRIVER" -ge 580 ]; then CU=cu130; elif [ "$DRIVER" -ge 575 ]; then CU=cu129; else CU=cu128; fi
else
  CU=cu126
fi
TORCH_INDEX="https://download.pytorch.org/whl/$CU"
echo ">>> GPU compute capability ${CC}.x, driver ${DRIVER}: torch ${TORCH_VERSION}+${CU}"

make_env() {  # name python [extra conda-forge packages...]
  local name="$1" py="$2"; shift 2
  if ! conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo ">>> creating conda env $name"
    conda create -y -q -n "$name" -c conda-forge --override-channels "python=$py" "$@" >/dev/null
  elif [ "$#" -gt 0 ]; then
    conda install -y -q -n "$name" -c conda-forge --override-channels "$@" >/dev/null
  fi
}

install_torch() {  # env
  conda run -n "$1" --no-capture-output pip install -q \
    "torch==${TORCH_VERSION}+${CU}" "torchaudio==${TORCH_VERSION}+${CU}" --index-url "$TORCH_INDEX"
}

gpu_check() {  # env
  conda run -n "$1" --no-capture-output python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("  torch", torch.__version__, "| cuda", ok, "|", torch.cuda.get_device_name(0) if ok else "no GPU")
if ok:
    x = torch.randn(512, 512, device="cuda")
    assert float((x @ x).abs().mean()) > 0
    print("  GPU kernel check OK")
EOF
}

# ---- main --------------------------------------------------------------------
make_env egyspeech 3.12 ffmpeg
install_torch egyspeech
conda run -n egyspeech --no-capture-output pip install -q -r "$REPO_DIR/envs/main.txt"
# audio-separator[gpu] may pull a torch from PyPI: put the matching CUDA build back.
install_torch egyspeech
conda run -n egyspeech --no-capture-output pip install -q -e "$REPO_DIR" --no-deps
conda run -n egyspeech --no-capture-output python -m ipykernel install --user --name egyspeech --display-name "Python (egyspeech)"
gpu_check egyspeech

# ---- nemo --------------------------------------------------------------------
make_env egyspeech-nemo 3.12 ffmpeg
install_torch egyspeech-nemo
conda run -n egyspeech-nemo --no-capture-output pip install -q -r "$REPO_DIR/envs/nemo.txt"
install_torch egyspeech-nemo
conda run -n egyspeech-nemo --no-capture-output pip install -q -e "$REPO_DIR" --no-deps
gpu_check egyspeech-nemo

# ---- qwen (QwenCleo-ASR) -----------------------------------------------------
make_env egyspeech-qwen 3.12 ffmpeg sox
install_torch egyspeech-qwen
conda run -n egyspeech-qwen --no-capture-output pip install -q -r "$REPO_DIR/envs/qwen.txt"
conda run -n egyspeech-qwen --no-capture-output pip install -q qwencleo-asr --no-deps
install_torch egyspeech-qwen
conda run -n egyspeech-qwen --no-capture-output pip install -q -e "$REPO_DIR" --no-deps
gpu_check egyspeech-qwen

# ---- vllm (optional) ---------------------------------------------------------
if [ "$WITH_VLLM" -eq 1 ]; then
  make_env egyspeech-vllm 3.12 ffmpeg
  # vLLM brings the torch build it was compiled against.
  conda run -n egyspeech-vllm --no-capture-output pip install -q -r "$REPO_DIR/envs/vllm.txt"
  gpu_check egyspeech-vllm
fi

cat <<EOF

Done. Conda: $CONDA_DIR
  source $CONDA_DIR/etc/profile.d/conda.sh && conda activate egyspeech
  cd $REPO_DIR && python -m egyspeech.cli --help
Jupyter kernel: "Python (egyspeech)"
EOF
