"""Pipeline steps. Each module exposes main(cfg) and runs as `python -m egyspeech.steps.<name>`."""

import argparse
import logging
from collections.abc import Callable

from egyspeech.config import Section, load_config
from egyspeech.io import read_jsonl
from egyspeech.paths import Layout


def layout(cfg: Section) -> Layout:
    return Layout(cfg.work_dir)


def video_ids(cfg: Section) -> list[str]:
    return [v["id"] for v in read_jsonl(layout(cfg).videos)]


def step_main(fn: Callable[[Section, argparse.Namespace], None], extra: Callable[[argparse.ArgumentParser], None] | None = None):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--limit", type=int, default=None, help="process at most N pending videos (testing)")
    p.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    fn(cfg, args)
