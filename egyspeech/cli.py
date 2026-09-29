"""EgySpeech command line.

    python -m egyspeech.cli run                        # every step, in order (resumes where it stopped)
    python -m egyspeech.cli run --from segment         # from a step on
    python -m egyspeech.cli run --steps quality,filter # selected steps
    python -m egyspeech.cli <step> [--limit N] [--force]
    python -m egyspeech.cli status                     # progress of every step

Each step runs in its own conda env (see egyspeech/runner.py).
"""

import argparse
import sys

from egyspeech.config import load_config
from egyspeech.runner import STEPS, run_step


def status(cfg) -> None:
    from egyspeech.io import read_json, read_jsonl
    from egyspeech.paths import Layout

    lay = Layout(cfg.work_dir)
    vids = [v["id"] for v in read_jsonl(lay.videos)]
    fmt = cfg.download.format
    backend = cfg.transcription.backend

    def count(fn) -> str:
        return f"{sum(fn(v) for v in vids)}/{len(vids)}"

    print(f"work dir: {lay.root}")
    print(f"  videos collected   {len(vids)}")
    print(f"  downloaded         {count(lambda v: lay.raw_audio(v, fmt).exists())}")
    print(f"  separated          {count(lambda v: lay.vocals(v).exists())}")
    print(f"  diarized           {count(lambda v: lay.diar_json(v).exists())}")
    print(f"  segmented          {count(lambda v: lay.chunks_meta(v).exists())}")
    print(f"  quality scored     {count(lambda v: lay.quality(v).exists())}")
    print(f"  speaker checked    {count(lambda v: lay.speaker_emb(v).exists())}")
    if lay.filter_report.exists():
        r = read_json(lay.filter_report)
        print(f"  filter             {r['kept_hours']:.1f} / {r['hours']:.1f} h kept")
    print(f"  transcribed ({backend}) {count(lambda v: lay.transcripts(backend, v).exists())}")
    print(f"  aligned            {count(lambda v: lay.aligned(v).exists())}")
    if (lay.final / "balance_summary.json").exists():
        b = read_json(lay.final / "balance_summary.json")
        print(f"  final              {b['total_hours']} h, {b['speakers']} speakers")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="egyspeech", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "status", *STEPS])
    p.add_argument("--config", default=None, help="default: configs/config.yaml")
    p.add_argument("--steps", default=None, help="comma-separated steps for `run`")
    p.add_argument("--from", dest="from_step", default=None, choices=STEPS)
    p.add_argument("--to", dest="to_step", default=None, choices=STEPS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.command == "status":
        status(cfg)
        return 0
    extra = (["--limit", str(a.limit)] if a.limit else []) + (["--force"] if a.force else [])
    if a.command != "run":
        return run_step(a.command, cfg, extra)
    steps = a.steps.split(",") if a.steps else list(STEPS)
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
