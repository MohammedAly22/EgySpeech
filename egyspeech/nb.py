"""Notebook helpers: run steps with live output, load results, listen to clips."""

import subprocess
import sys

import numpy as np
import pandas as pd

from egyspeech.config import REPO_ROOT, load_config
from egyspeech.io import iter_jsonl_dir, read_audio, read_jsonl
from egyspeech.paths import Layout


def run(step: str, *extra: str, config: str | None = None) -> int:
    """Run `python -m egyspeech.cli <step>` and stream its output into the notebook."""
    cmd = [sys.executable, "-m", "egyspeech.cli", step, *extra]
    if config:
        cmd += ["--config", config]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    if code:
        raise RuntimeError(f"step {step} failed with exit code {code}")
    return code


def cfg_and_layout(config: str | None = None):
    cfg = load_config(config)
    return cfg, Layout(cfg.work_dir)


def clips_frame(lay: Layout) -> pd.DataFrame:
    rows = list(iter_jsonl_dir(lay.meta / "chunks"))
    q = {r["id"]: r for r in iter_jsonl_dir(lay.meta / "quality")}
    for r in rows:
        r.update(q.get(r["id"], {}))
    df = pd.DataFrame(rows)
    sims = {}
    for f in (lay.meta / "speakers").glob("*.npz"):
        z = np.load(f)
        sims.update(zip(z["ids"].tolist(), z["window_similarity"].tolist(), strict=True))
    if len(df):
        df["window_similarity"] = df["id"].map(sims)
    return df


def play(path: str, label: str = "", max_sec: float | None = None):
    """Audio player for a clip (or the first max_sec seconds of an episode)."""
    import soundfile as sf
    from IPython.display import Audio, Markdown, display

    if label:
        display(Markdown(label))
    if max_sec:
        with sf.SoundFile(str(path)) as f:
            wav, sr = f.read(int(max_sec * f.samplerate), dtype="float32", always_2d=True).mean(axis=1), f.samplerate
    else:
        wav, sr = read_audio(path)
    display(Audio(wav, rate=sr))


def transcripts_frame(lay: Layout, backend: str) -> pd.DataFrame:
    return pd.DataFrame(list(iter_jsonl_dir(lay.meta / "transcripts" / backend)))


def final_frame(lay: Layout) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        for r in read_jsonl(lay.split(split)):
            r["split"] = split
            rows.append(r)
    return pd.DataFrame(rows)
