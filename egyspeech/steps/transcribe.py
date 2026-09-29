"""Step 9 — transcribe: ASR on the clips that passed the filter (only those).

Output per video: meta/transcripts/<backend>/<vid>.jsonl with
    {"id", "text", ["text_tagged", "tags"], "ok", "flags", "chars_per_sec", "arabic_ratio"}
`ok` is False when the transcript looks wrong: empty/truncated, looping
(hallucination), implausible speaking rate, or mostly non-Arabic.
"""

import logging
from collections import defaultdict

from egyspeech.asr import get_backend
from egyspeech.config import Section
from egyspeech.io import read_jsonl, write_jsonl
from egyspeech.steps import layout, step_main
from egyspeech.text import arabic_ratio, has_repetition_loop, letters_count

logger = logging.getLogger("transcribe")


def sanity(text: str, duration: float, s: Section) -> tuple[list[str], float, float]:
    flags = []
    cps = letters_count(text) / max(duration, 1e-6)
    ar = arabic_ratio(text)
    if not text.strip():
        flags.append("empty")
    if cps < s.min_chars_per_sec:
        flags.append("too_slow")
    if cps > s.max_chars_per_sec:
        flags.append("too_fast")
    if ar < s.min_arabic_ratio:
        flags.append("not_arabic")
    if has_repetition_loop(text):
        flags.append("repetition")
    return flags, round(cps, 2), round(ar, 3)


def run(cfg: Section, args, backend_name: str):
    lay = layout(cfg)
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(lay.filtered):
        by_video[r["video_id"]].append(r)
    if not by_video:
        raise SystemExit("no filtered clips: run the filter step first")
    pending = [v for v in by_video if args.force or not lay.transcripts(backend_name, v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"backend={backend_name}: {len(by_video) - len(pending)} videos done, {len(pending)} to transcribe")
    if not pending:
        return
    backend = get_backend(backend_name, cfg)
    s = cfg.transcription.sanity
    n_ok = n_all = 0
    for i, vid in enumerate(pending, 1):
        clips = by_video[vid]
        results = backend.transcribe([c["path"] for c in clips], [c["duration"] for c in clips])
        rows = []
        for c, res in zip(clips, results, strict=True):
            flags, cps, ar = sanity(res["text"], c["duration"], s)
            row = {"id": c["id"], "text": res["text"], "ok": not flags, "flags": flags,
                   "chars_per_sec": cps, "arabic_ratio": ar, "backend": backend_name}
            if "text_tagged" in res:
                row["text_tagged"] = res["text_tagged"]
                row["tags"] = res["tags"]
            rows.append(row)
        write_jsonl(lay.transcripts(backend_name, vid), rows)
        ok = sum(r["ok"] for r in rows)
        n_ok += ok
        n_all += len(rows)
        logger.info(f"[{i}/{len(pending)}] {vid}: {len(rows)} clips, {ok} pass sanity")
    logger.info(f"this run: {n_ok}/{n_all} transcripts pass the sanity checks")


def main(cfg: Section, args):
    run(cfg, args, args.backend or cfg.transcription.backend)


if __name__ == "__main__":
    step_main(main, lambda p: p.add_argument("--backend", default=None))
