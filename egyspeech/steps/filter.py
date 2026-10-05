"""Step 8 — filter: keep clean, single-speaker clips (thresholds from config; re-run freely).

Only these clips are transcribed, so ASR time is never spent on noisy / music /
multi-speaker audio.
"""

import logging
from collections import Counter, defaultdict

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_json, read_jsonl, write_json, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main, video_ids

logger = logging.getLogger("filter")


def load_joined(cfg: Section) -> list[dict]:
    """chunks + quality + speaker consistency for every processed video."""
    lay = layout(cfg)
    vids = [v for v in video_ids(cfg)
            if lay.chunks_meta(v).exists() and lay.quality(v).exists() and lay.speaker_emb(v).exists()]
    rows = []
    with StepBar("filter (load)", len(vids)) as bar:
        for vid in vids:
            q = {r["id"]: r for r in read_jsonl(lay.quality(vid))}
            z = np.load(lay.speaker_emb(vid))
            sim = dict(zip(z["ids"].tolist(), z["window_similarity"].tolist(), strict=True))
            for c in read_jsonl(lay.chunks_meta(vid)):
                if c["id"] in q and c["id"] in sim:
                    rows.append({**c, **q[c["id"]], "window_similarity": round(float(sim[c["id"]]), 4)})
            bar.advance()
    return rows


# config key -> (clip metric, "min" = reject below / "max" = reject above, rejection reason).
# A threshold set to null in the config disables that rule (the `tune` page writes these keys).
RULES = {
    "min_dnsmos_bak": ("dnsmos_bak", "min", "dnsmos_bak"),
    "min_window_similarity": ("window_similarity", "min", "multi_speaker"),
    "min_dnsmos_sig": ("dnsmos_sig", "min", "dnsmos_sig"),
    "min_dnsmos_ovrl": ("dnsmos_ovrl", "min", "dnsmos_ovrl"),
    "min_dnsmos_p808": ("dnsmos_p808", "min", "dnsmos_p808"),
    "min_utmos": ("utmos", "min", "utmos"),
    "min_speech_ratio": ("speech_ratio", "min", "low_speech"),
    "max_clip_ratio": ("clip_ratio", "max", "clipping"),
    "max_edge_db": ("edge_db", "max", "edge_not_silent"),
}


def edge_db(r: dict) -> float | None:
    """The louder of the two clip edges, dB relative to the clip's speech (near 0 = speech at the cut)."""
    edges = [e for e in (r.get("edge_start_db"), r.get("edge_end_db")) if e is not None]
    return max(edges) if edges else None


def reasons(r: dict, f: Section, s: Section) -> list[str]:
    out = []
    lo, hi = (s.diar_min_sec, s.diar_max_sec) if s.get("method") == "diarization" else (s.min_sec, s.max_sec)
    if not lo - 0.05 <= r["duration"] <= hi + 0.05:
        out.append("duration")
    for key, (metric, side, why) in RULES.items():
        limit = f.get(key)
        x = edge_db(r) if metric == "edge_db" else r.get(metric)
        if limit is None or x is None:
            continue
        if (x < limit) if side == "min" else (x > limit):
            out.append(why)
    if f.get("drop_weak_cuts") and r.get("weak_cut"):
        out.append("weak_cut")
    return out


def clip_score(r: dict) -> float:
    """Ranking inside a speaker's clips when capping: cleaner background and a purer voice first."""
    return (r.get("dnsmos_bak") or 0.0) + (r.get("dnsmos_sig") or 0.0) + 2.0 * (r.get("window_similarity") or 0.0)


def cap_speaker(rows: list[dict], cap_sec: float) -> list[dict]:
    """At most cap_sec of one speaker: best clips first, round-robin over the speaker's episodes."""
    if sum(r["duration"] for r in rows) <= cap_sec:
        return rows
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=clip_score, reverse=True):
        by_video[r["video_id"]].append(r)
    queues = sorted(by_video.values(), key=len, reverse=True)
    chosen, total = [], 0.0
    while queues and total < cap_sec:
        nxt = []
        for q in queues:
            if total >= cap_sec:
                break
            r = q.pop(0)
            chosen.append(r)
            total += r["duration"]
            if q:
                nxt.append(q)
        queues = nxt
    return chosen


def speaker_stats(rows: list[dict]) -> dict:
    hours: dict[str, float] = defaultdict(float)
    gender: dict[str, str] = {}
    for r in rows:
        hours[r.get("speaker_id") or "?"] += r["duration"] / 3600
        gender[r.get("speaker_id") or "?"] = r.get("gender") or "?"
    h = np.array(list(hours.values())) if hours else np.zeros(1)
    by_g = defaultdict(lambda: [0, 0.0])
    for s, x in hours.items():
        by_g[gender[s]][0] += 1
        by_g[gender[s]][1] += x
    return {"clips": len(rows), "hours": round(float(h.sum()), 2), "speakers": len(hours),
            "mean_hours": round(float(h.mean()), 3), "median_hours": round(float(np.median(h)), 3),
            "min_hours": round(float(h.min()), 4), "max_hours": round(float(h.max()), 3),
            "gender": {g: {"speakers": n, "hours": round(x, 2)} for g, (n, x) in sorted(by_g.items())}}


def log_stats(title: str, st: dict) -> None:
    g = ", ".join(f"{k} {v['speakers']} speakers / {v['hours']:.1f} h" for k, v in st["gender"].items())
    logger.info(f"{title}: {st['clips']:,} clips, {st['hours']:.1f} h, {st['speakers']:,} speakers | hours per "
                f"speaker: mean {st['mean_hours']:.2f}, median {st['median_hours']:.2f}, min "
                f"{st['min_hours'] * 60:.1f} min, max {st['max_hours']:.2f} | {g}")


def main(cfg: Section, args):
    lay = layout(cfg)
    rows = load_joined(cfg)
    if not rows:
        raise SystemExit("nothing to filter: run quality first (and segment / speaker_check, or pull_chunks)")
    assignment, genders = {}, {}
    if lay.speakers.exists():
        spk = read_json(lay.speakers)
        assignment = spk["assignment"]
        genders = {s["id"]: s["gender"] for s in spk["speakers"]}
    for r in rows:
        r["speaker_id"] = assignment.get(r["id"])
        r["gender"] = genders.get(r["speaker_id"])
    counts: Counter[str] = Counter()
    passed = []
    for r in rows:
        why = reasons(r, cfg.filter, cfg.segmentation)
        counts.update(why)
        if not why:
            passed.append(r)

    max_hours = args.max_hours if args.max_hours is not None else cfg.filter.get("max_hours_per_speaker")
    if max_hours and not assignment:
        raise SystemExit("speaker cap needs speaker IDs (meta/speakers.json): run the cluster step or pull_chunks")
    if max_hours:
        by_spk: dict[str, list[dict]] = defaultdict(list)
        for r in passed:
            by_spk[r["speaker_id"] or r["id"]].append(r)
        kept = [r for rs in by_spk.values() for r in cap_speaker(rs, float(max_hours) * 3600)]
    else:
        kept = passed

    st_all, st_q, st_k = speaker_stats(rows), speaker_stats(passed), speaker_stats(kept)
    log_stats("all clips", st_all)
    log_stats("quality passed", st_q)
    logger.info(f"quality rejections: {dict(counts)}")
    if max_hours:
        capped = sum(1 for rs in by_spk.values() if sum(r["duration"] for r in rs) > float(max_hours) * 3600)
        log_stats(f"capped at {max_hours:g} h/speaker ({capped} speakers capped)", st_k)
    if args.dry_run:
        logger.info("--dry-run: nothing written")
        return

    write_jsonl(lay.filtered, kept)
    if cfg.storage.delete_rejected_clips:  # only clips that failed the quality filter (the cap can change)
        from pathlib import Path

        passed_ids = {r["id"] for r in passed}
        removed = 0
        for r in rows:
            if r["id"] not in passed_ids and Path(r["path"]).exists():
                Path(r["path"]).unlink()
                removed += 1
        logger.info(f"deleted the audio of {removed} rejected clips (storage.delete_rejected_clips)")
    report = {"clips": len(rows), "hours": st_all["hours"], "kept_clips": len(kept), "kept_hours": st_k["hours"],
              "quality_passed_hours": st_q["hours"], "rejections": dict(counts), "max_hours_per_speaker": max_hours,
              "speakers_all": st_all, "speakers_quality": st_q, "speakers_kept": st_k,
              "thresholds": dict(cfg.filter)}
    write_json(lay.filter_report, report)
    logger.info(f"wrote {lay.filtered} ({len(kept):,} clips, {st_k['hours']:.1f} h): next `transcribe`")


if __name__ == "__main__":
    def _args(p):
        p.add_argument("--max-hours", type=float, default=None,
                       help="cap per speaker in hours (default: filter.max_hours_per_speaker; 0 = no cap)")
        p.add_argument("--dry-run", action="store_true", help="only print the statistics")

    step_main(main, _args)
