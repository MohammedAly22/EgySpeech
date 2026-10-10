"""Run each step inside the conda env it needs (they have conflicting dependencies).

Every step is a module `egyspeech.steps.<name>` with `main(cfg)`; the runner starts
`<conda_dir>/envs/<env>/bin/python -m egyspeech.steps.<name> --config <file>` and
streams its output. Steps skip work that is already done, so any step can be
re-run or resumed after an interruption.
"""

import os
import subprocess
import sys
from pathlib import Path

from egyspeech.config import REPO_ROOT, Section

# step -> env key in cfg.envs ("transcribe"/"verify" depend on the ASR backend)
STEP_ENV = {
    "download": "main",
    "index": "main",
    "separate": "main",
    "diarize": "nemo",
    "segment": "main",
    "quality": "main",
    "speaker_check": "nemo",
    "filter": "main",
    "review": "main",
    "tune": "main",
    "push_chunks": "main",
    "pull_chunks": "main",
    "transcribe": None,
    "verify": None,
    "align": "main",
    "cluster": "main",
    "balance": "main",
    "analysis": "main",
    "publish": "main",
}
STEPS = list(STEP_ENV)
# `run --stage local`: episodes on disk -> diarization chunks + global speaker IDs (a small GPU is enough),
#                      then `push_chunks`;
# `run --stage gpu`:   after `pull_chunks` on a GPU machine: quality -> filter (+ speaker cap) -> transcribe
#                      -> align -> balance (splits) -> analysis -> publish.
STAGES = {
    "local": ["index", "separate", "diarize", "segment", "speaker_check", "cluster"],
    "gpu": ["quality", "filter", "transcribe", "verify", "align", "balance", "analysis", "publish"],
}
STAGES["all"] = STAGES["local"] + STAGES["gpu"]
RESTART = 75  # a step exits with this to be restarted in a fresh process (e.g. broken GPU state)
BACKEND_ENV = {"qwencleo": "qwen", "cohere": "main", "parakeet": "nemo", "llm": "main"}


def env_for(step: str, cfg: Section) -> str:
    key = STEP_ENV[step]
    if key is None:
        backend = cfg.transcription.backend if step == "transcribe" else cfg.transcription.verify.backend
        key = BACKEND_ENV[backend] if backend else "main"  # verify disabled: no-op in the main env
    return cfg.envs[key]


def env_python(env_name: str, cfg: Section) -> str:
    """Interpreter of a conda env; the current one if we are already inside it."""
    if Path(sys.prefix).name == env_name:
        return sys.executable
    python = Path(cfg.envs.conda_dir) / "envs" / env_name / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"{python} not found: run scripts/install.sh (env {env_name})")
    return str(python)


def run_step(step: str, cfg: Section, extra: list[str] | None = None) -> int:
    if step not in STEP_ENV:
        raise ValueError(f"unknown step {step!r}; steps: {', '.join(STEPS)}")
    env_name = env_for(step, cfg)
    python = env_python(env_name, cfg)
    cmd = [python, "-m", f"egyspeech.steps.{step}", "--config", cfg["_path"], *(extra or [])]
    print(f"\n==> [{step}] env={env_name}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT)}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # cloud templates often set HF_HUB_ENABLE_HF_TRANSFER=1; downloads then fail in envs without hf_transfer
    if env.get("HF_HUB_ENABLE_HF_TRANSFER") not in (None, "", "0") and not any(
            (Path(python).parent.parent / "lib").glob("python3*/site-packages/hf_transfer")):
        env.pop("HF_HUB_ENABLE_HF_TRANSFER")
    # ffmpeg / sox of the env on PATH, as `conda activate` would do
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    code = 0
    for attempt in range(2, 102):
        code = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)
        if code != RESTART:
            return code
        print(f"\n==> [{step}] restarting in a fresh process (attempt {attempt})", flush=True)
    return code
