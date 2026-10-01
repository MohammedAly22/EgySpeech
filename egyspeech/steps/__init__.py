"""Pipeline steps. Each module exposes main(cfg, args) and runs as `python -m egyspeech.steps.<name>`."""

import argparse
from collections.abc import Callable
from pathlib import Path

from egyspeech.config import Section, load_config
from egyspeech.io import read_jsonl, write_jsonl
from egyspeech.paths import Layout
from egyspeech.progress import setup_logging


def layout(cfg: Section) -> Layout:
    return Layout(cfg.work_dir)


def raw_dir(cfg: Section) -> Path:
    """Folder of downloaded episodes (download.dir; default <work_dir>/raw_download)."""
    d = cfg.download.get("dir")
    return Path(d) if d else Path(cfg.work_dir) / "raw_download"


def videos(cfg: Section) -> list[dict]:
    """Episodes on disk, as indexed by the `index` step."""
    return read_jsonl(layout(cfg).videos)


def video_ids(cfg: Section) -> list[str]:
    return [v["id"] for v in videos(cfg)]


def speech_audio(cfg: Section, video: dict) -> Path:
    """Audio the speech steps read: the vocal stem when separation is on, else the episode itself."""
    if cfg.separation.mode == "never":
        return Path(video["path"])
    return layout(cfg).vocals(video["id"])


def select(cfg: Section, args: argparse.Namespace, ready: Callable[[dict], bool],
           done: Callable[[dict], bool]) -> tuple[list[dict], int]:
    """(videos to process, videos already done) among those whose inputs are ready; applies --limit / --force."""
    candidates = [v for v in videos(cfg) if ready(v)]
    pending = [v for v in candidates if args.force or not done(v)]
    n_done = len(candidates) - len(pending)
    if args.limit:
        pending = pending[: args.limit]
    return pending, n_done


def total_sec(rows: list[dict]) -> float:
    return float(sum(r.get("duration") or 0.0 for r in rows))


def write_failures(cfg: Section, step: str, failures: list[dict]) -> None:
    """meta/failures/<step>.jsonl lists what failed in the last run (those videos are retried next run)."""
    path = layout(cfg).failures(step)
    if failures or path.exists():
        write_jsonl(path, failures)


def step_main(fn: Callable[[Section, argparse.Namespace], None],
              extra: Callable[[argparse.ArgumentParser], None] | None = None):
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--limit", type=int, default=None, help="process at most N pending videos (testing)")
    p.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    fn(cfg, args)
