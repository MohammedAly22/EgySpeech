"""Step 8 — filter: keep clean, single-speaker clips (thresholds from config; re-run freely).

Only these clips are transcribed, so ASR time is never spent on noisy / music /
multi-speaker audio.
"""

import logging
from collections import Counter

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_jsonl, write_json, write_jsonl
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
    if not s.min_sec <= r["duration"] <= s.max_sec:
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


def main(cfg: Section, args):
    lay = layout(cfg)
    rows = load_joined(cfg)
    if not rows:
        raise SystemExit("nothing to filter: run segment, quality and speaker_check first")
    counts: Counter[str] = Counter()
    kept = []
    for r in rows:
        why = reasons(r, cfg.filter, cfg.segmentation)
        counts.update(why)
        if not why:
            kept.append(r)
    write_jsonl(lay.filtered, kept)
    if cfg.storage.delete_rejected_clips:
        from pathlib import Path

        kept_ids = {r["id"] for r in kept}
        removed = 0
        for r in rows:
            if r["id"] not in kept_ids and Path(r["path"]).exists():
                Path(r["path"]).unlink()
                removed += 1
        logger.info(f"deleted the audio of {removed} rejected clips (storage.delete_rejected_clips)")
    total_h = sum(r["duration"] for r in rows) / 3600
    kept_h = sum(r["duration"] for r in kept) / 3600
    report = {"clips": len(rows), "hours": round(total_h, 3), "kept_clips": len(kept),
              "kept_hours": round(kept_h, 3), "rejections": dict(counts), "thresholds": dict(cfg.filter)}
    write_json(lay.filter_report, report)
    logger.info(f"kept {len(kept)}/{len(rows)} clips = {kept_h:.1f}/{total_h:.1f} h; rejections: {dict(counts)}")


if __name__ == "__main__":
    step_main(main)
