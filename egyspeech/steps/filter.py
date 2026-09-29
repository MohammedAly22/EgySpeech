"""Step 8 — filter: keep clean, single-speaker clips (thresholds from config; re-run freely).

Only these clips are transcribed, so ASR time is never spent on noisy / music /
multi-speaker audio.
"""

import logging
from collections import Counter

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_jsonl, write_json, write_jsonl
from egyspeech.steps import layout, step_main, video_ids

logger = logging.getLogger("filter")


def load_joined(cfg: Section) -> list[dict]:
    """chunks + quality + speaker consistency for every processed video."""
    lay = layout(cfg)
    rows = []
    for vid in video_ids(cfg):
        if not (lay.chunks_meta(vid).exists() and lay.quality(vid).exists() and lay.speaker_emb(vid).exists()):
            continue
        q = {r["id"]: r for r in read_jsonl(lay.quality(vid))}
        z = np.load(lay.speaker_emb(vid))
        sim = dict(zip(z["ids"].tolist(), z["window_similarity"].tolist(), strict=True))
        for c in read_jsonl(lay.chunks_meta(vid)):
            if c["id"] in q and c["id"] in sim:
                rows.append({**c, **q[c["id"]], "window_similarity": round(float(sim[c["id"]]), 4)})
    return rows


def reasons(r: dict, f: Section, s: Section) -> list[str]:
    out = []
    if not s.min_sec <= r["duration"] <= s.max_sec:
        out.append("duration")
    if r["dnsmos_ovrl"] < f.min_dnsmos_ovrl:
        out.append("dnsmos_ovrl")
    if r["dnsmos_sig"] < f.min_dnsmos_sig:
        out.append("dnsmos_sig")
    if r["dnsmos_bak"] < f.min_dnsmos_bak:
        out.append("dnsmos_bak")
    if r["utmos"] < f.min_utmos:
        out.append("utmos")
    if r["window_similarity"] < f.min_window_similarity:
        out.append("multi_speaker")
    if r["clip_ratio"] > f.max_clip_ratio:
        out.append("clipping")
    if r["speech_ratio"] < f.min_speech_ratio:
        out.append("low_speech")
    if f.drop_weak_cuts and r["weak_cut"]:
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
