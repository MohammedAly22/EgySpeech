"""EgySpeech command line.

    python -m egyspeech.cli download                       # links file -> FLAC episodes
    python -m egyspeech.cli run --stage local [--limit 5]  # episodes -> clean single-speaker clips
    python -m egyspeech.cli review                         # listening page for the clips
    python -m egyspeech.cli push_chunks                    # clips -> private Hugging Face dataset
    python -m egyspeech.cli pull_chunks                    # (GPU machine) dataset -> clips on disk
    python -m egyspeech.cli run --stage gpu                # transcribe ... publish
    python -m egyspeech.cli <step> [--limit N] [--force] [step options, e.g. segment --watch]
    python -m egyspeech.cli status                         # progress of every step

Every step resumes where it stopped and runs in its own conda env (egyspeech/runner.py).
"""

import argparse
import sys

from egyspeech.config import load_config
from egyspeech.runner import STAGES, STEPS, run_step


def status(cfg) -> None:
    from egyspeech.io import read_json, read_jsonl
    from egyspeech.paths import Layout

    lay = Layout(cfg.work_dir)
    vids = read_jsonl(lay.videos)
    total_h = sum(v.get("duration") or 0 for v in vids) / 3600
    backend = cfg.transcription.backend

    def count(fn) -> str:
        done = [v for v in vids if fn(v["id"])]
        return f"{len(done):>6}/{len(vids)} videos  {sum(v.get('duration') or 0 for v in done) / 3600:8.1f} h"

    def failed(step: str) -> str:
        n = len(read_jsonl(lay.failures(step)))
        return f"   ({n} failed in the last run)" if n else ""

    print(f"work dir: {lay.root}")
    print(f"  indexed            {len(vids):>6} videos  {total_h:8.1f} h")
    if cfg.separation.mode != "never":
        print(f"  separated          {count(lambda v: lay.vocals(v).exists())}{failed('separate')}")
    print(f"  diarized           {count(lambda v: lay.diar_json(v).exists())}{failed('diarize')}")
    print(f"  segmented          {count(lambda v: lay.chunks_meta(v).exists())}{failed('segment')}")
    print(f"  quality scored     {count(lambda v: lay.quality(v).exists())}{failed('quality')}")
    print(f"  speaker checked    {count(lambda v: lay.speaker_emb(v).exists())}{failed('speaker_check')}")
    if lay.filter_report.exists():
        r = read_json(lay.filter_report)
        print(f"  filter             {r['kept_clips']:,}/{r['clips']:,} clips, {r['kept_hours']:.1f}/{r['hours']:.1f} h kept")
    if (lay.hub / "manifest.json").exists():
        m = read_json(lay.hub / "manifest.json")
        print(f"  on the Hub         {len(m['videos'])} videos, {m['clips']:,} clips, {m['hours']:.1f} h  ({m['repo_id']})")
    print(f"  transcribed ({backend}) {count(lambda v: lay.transcripts(backend, v).exists())}")
    print(f"  aligned            {count(lambda v: lay.aligned(v).exists())}")
    if (lay.final / "balance_summary.json").exists():
        b = read_json(lay.final / "balance_summary.json")
        print(f"  final              {b['total_hours']} h, {b['speakers']} speakers")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="egyspeech", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "status", *STEPS])
    p.add_argument("--config", default=None, help="default: configs/config.yaml")
    p.add_argument("--stage", default="all", choices=list(STAGES), help="steps of `run` (default: all)")
    p.add_argument("--steps", default=None, help="comma-separated steps for `run` (overrides --stage)")
    p.add_argument("--from", dest="from_step", default=None, choices=STEPS)
    p.add_argument("--to", dest="to_step", default=None, choices=STEPS)
    p.add_argument("--limit", type=int, default=None, help="process at most N pending videos per step (testing)")
    p.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    a, passthrough = p.parse_known_args(argv)
    cfg = load_config(a.config)
    if a.command == "status":
        status(cfg)
        return 0
    extra = (["--limit", str(a.limit)] if a.limit else []) + (["--force"] if a.force else [])
    if a.command != "run":
        return run_step(a.command, cfg, extra + passthrough)
    if passthrough:
        p.error(f"unknown options for `run`: {' '.join(passthrough)} (step options go with the step command)")
    steps = a.steps.split(",") if a.steps else list(STAGES[a.stage])
    if a.from_step:
        steps = steps[steps.index(a.from_step):]
    if a.to_step:
        steps = steps[: steps.index(a.to_step) + 1]
    for step in steps:
        code = run_step(step, cfg, extra)
        if code != 0:
            print(f"\n[{step}] failed with exit code {code}; fix the issue and re-run: finished work is kept.")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
