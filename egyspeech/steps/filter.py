"""Step 8 — filter: keep clean, single-speaker clips (thresholds from config; re-run freely).

Two stages:
  1. hard limits (filter.hard): unambiguous defects; any one rejects the clip
     (far too noisy, a clearly different voice, clipping, mostly silence, ...);
  2. weighted risk (filter.risk): each metric adds risk from 0 (at its `good` value)
     to its `weight` (at its `bad` value, linear in between). The clip is rejected when
     the total reaches filter.max_risk. One borderline score is tolerated; several
     weak signs together (a bit of background + a loud edge + a low UTMOS) are not.

Only kept clips are transcribed, so ASR time is never spent on noisy / music /
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


HARD_RULES = {  # config key -> (metric, reject when the metric is below / above, reason)
    "min_dnsmos_ovrl": ("dnsmos_ovrl", "below", "dnsmos_ovrl"),
    "min_dnsmos_sig": ("dnsmos_sig", "below", "dnsmos_sig"),
    "min_dnsmos_bak": ("dnsmos_bak", "below", "dnsmos_bak"),
    "min_utmos": ("utmos", "below", "utmos"),
    "min_window_similarity": ("window_similarity", "below", "multi_speaker"),
    "max_clip_ratio": ("clip_ratio", "above", "clipping"),
    "min_speech_ratio": ("speech_ratio", "below", "low_speech"),
    "max_edge_db": ("edge_db", "above", "edge_not_silent"),
}


def metrics(r: dict) -> dict:
    """The row plus derived metrics (edge_db = the louder of the two clip edges)."""
    edges = [e for e in (r.get("edge_start_db"), r.get("edge_end_db")) if e is not None]
    return {**r, "edge_db": max(edges) if edges else None}


def risk(r: dict, f: Section) -> tuple[float, dict[str, float]]:
    """(total risk, contribution per metric) of a clip under filter.risk."""
    m = metrics(r)
    contrib = {}
    for name, spec in (f.get("risk") or {}).items():
        x = m.get(name)
        if x is None:
            continue
        good, bad, weight = float(spec["good"]), float(spec["bad"]), float(spec["weight"])
        share = (good - x) / (good - bad)  # 0 at good, 1 at bad (works for both directions)
        contrib[name] = round(weight * min(1.0, max(0.0, share)), 3)
    return round(sum(contrib.values()), 3), contrib


def reasons(r: dict, f: Section, s: Section) -> list[str]:
    """Why a clip is rejected (empty = kept): hard limits, then the combined risk."""
    m = metrics(r)
    out = []
    if not s.min_sec <= r["duration"] <= s.max_sec:
        out.append("duration")
    for key, value in (f.get("hard") or {}).items():
        metric, side, why = HARD_RULES[key]
        x = m.get(metric)
        if x is not None and (x < value if side == "below" else x > value):
            out.append(why)
    if f.drop_weak_cuts and r.get("weak_cut"):
        out.append("weak_cut")
    if not out and risk(r, f)[0] >= f.max_risk:
        out.append("combined_risk")
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
        r["filter_risk"] = risk(r, cfg.filter)[0]
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
