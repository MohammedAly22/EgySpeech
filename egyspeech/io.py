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


def decode_audio(path: str | Path, sample_rate: int, expected_sec: float | None = None) -> np.ndarray:
    """Any audio file -> mono float32 at sample_rate (ffmpeg).

    Decoded straight into one preallocated array (sized from expected_sec), so a 4 h
    episode needs its own size in RAM, not twice that.
    """
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", str(sample_rate),
           "-f", "f32le", "-"]
    buf = np.empty(int((expected_sec or 600) * sample_rate) + sample_rate, dtype=np.float32)
    filled = 0  # bytes
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdout is not None and proc.stderr is not None
        while True:
            view = memoryview(buf).cast("B")
            if filled == view.nbytes:  # longer than expected: grow
                view.release()
                grown = np.empty(len(buf) * 3 // 2, dtype=np.float32)
                grown[: len(buf)] = buf
                buf = grown
                continue
            k = proc.stdout.readinto(view[filled:])
            view.release()
            if not k:
                break
            filled += k
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg could not decode {path}: {err.strip()[-300:]}")
    n = filled // 4
    return buf[:n] if n > 0.9 * len(buf) else buf[:n].copy()


class ClipReader:
    """Time ranges of one audio file, resampled to sample_rate.

    Seeks with soundfile (FLAC / WAV / MP3 / OGG) so clips are read without decoding
    the whole episode; other formats are decoded once with ffmpeg.
    """

    def __init__(self, path: str | Path, sample_rate: int):
        self.sr = sample_rate
        self.full: np.ndarray | None = None
        try:
            self.f: sf.SoundFile | None = sf.SoundFile(str(path))
            self.sr_in = self.f.samplerate
        except (RuntimeError, OSError):  # format libsndfile cannot read
            self.f = None
            self.sr_in = sample_rate
            self.full = decode_audio(path, sample_rate)

    def read(self, start: float, end: float) -> np.ndarray:
        a, b = max(0, int(round(start * self.sr_in))), int(round(end * self.sr_in))
        if self.f is None:
            assert self.full is not None
            return self.full[a:b]
        self.f.seek(min(a, self.f.frames))
        x = self.f.read(max(0, b - a), dtype="float32", always_2d=True).mean(axis=1)
        return resample(x, self.sr_in, self.sr)

    def close(self):
        if self.f is not None:
            self.f.close()

    def __enter__(self) -> "ClipReader":
        return self

    def __exit__(self, *exc):
        self.close()


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
