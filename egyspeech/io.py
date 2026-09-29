"""Small I/O helpers: atomic writes (every step is resumable), jsonl, audio."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    tmp = atomic_path(path)
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl_dir(directory: Path) -> Iterator[dict[str, Any]]:
    for p in sorted(directory.glob("*.jsonl")):
        yield from read_jsonl(p)


def write_json(path: str | Path, obj: Any):
    path = Path(path)
    tmp = atomic_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ffmpeg_bin() -> str:
    """ffmpeg from PATH, else next to this interpreter (conda-forge ffmpeg in the env)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(sys.prefix) / "bin" / "ffmpeg"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("ffmpeg not found: run scripts/install.sh (installs ffmpeg into the env)")


def decode_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Any audio file -> mono float32 at sample_rate (ffmpeg)."""
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", str(sample_rate),
           "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def write_audio(path: str | Path, wav: np.ndarray, sample_rate: int, subtype: str | None = None):
    path = Path(path)
    tmp = atomic_path(path).with_suffix(path.suffix)
    fmt = path.suffix.lstrip(".").upper()
    sf.write(str(tmp), wav, sample_rate, format=fmt, subtype=subtype or ("PCM_16" if fmt in ("FLAC", "WAV") else None))
    os.replace(tmp, path)


def resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return wav
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target)
    return resample_poly(wav, target // g, sr // g).astype(np.float32)
