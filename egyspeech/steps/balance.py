"""Step 13 — balance: final selection, speaker balancing and splits.

A clip enters the final dataset only if it passed every stage: quality filter,
transcript sanity, forced alignment (score, all words timed, no word cut at the
edges) and — when enabled — the two-ASR agreement check. Then:
  * each speaker is capped at min(max_hours_per_speaker, max_speaker_share x total),
    keeping that speaker's best clips (quality + alignment) spread over their videos;
  * `test_speakers` speakers are held out entirely (unseen-speaker test split),
    and a small seen-speaker validation split is sampled.
"""

import logging
import random
from collections import defaultdict

from egyspeech.config import Section
from egyspeech.io import read_json, read_jsonl, write_json, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main
from egyspeech.text import cer, is_code_switched, latin_words, normalize_for_compare

logger = logging.getLogger("balance")


def candidates(cfg: Section) -> tuple[list[dict], dict]:
    lay = layout(cfg)
    clips = {r["id"]: r for r in read_jsonl(lay.filtered)}
    spk = read_json(lay.speakers)
    assign, speakers = spk["assignment"], {s["id"]: s for s in spk["speakers"]}
    videos = {v["id"]: v for v in read_jsonl(lay.videos)}
    backend = cfg.transcription.backend
    verify = cfg.transcription.verify.backend
    stats = defaultdict(int)
    rows = []
    by_video: dict[str, list[dict]] = defaultdict(list)
    for c in clips.values():
        by_video[c["video_id"]].append(c)
    bar = StepBar("balance (collect)", len(by_video)).start()
    for vid in sorted(by_video):
        bar.advance()
        trans = {t["id"]: t for t in read_jsonl(lay.transcripts(backend, vid))}
        aligned = {a["id"]: a for a in read_jsonl(lay.aligned(vid))}
        ver = {t["id"]: t for t in read_jsonl(lay.transcripts(verify, vid))} if verify else {}
        for c in by_video[vid]:
            cid = c["id"]
            t = trans.get(cid)
            if t is None:
                stats["not_transcribed"] += 1
                continue
            if not t["ok"]:
                stats["transcript_sanity"] += 1
                continue
            al = aligned.get(cid)
            if al is None or not al["align_ok"]:
                stats["alignment"] += 1
                continue
            agreement = None
            if verify:
                v = ver.get(cid)
                if v is None:
                    stats["not_verified"] += 1
                    continue
                agreement = cer(normalize_for_compare(t["text"]), normalize_for_compare(v["text"]))
                if agreement > cfg.transcription.verify.max_cer:
                    stats["asr_disagreement"] += 1
                    continue
            g = assign.get(cid)
            if g is None:
                stats["no_speaker"] += 1
                continue
            vm = videos.get(vid, {})
            rows.append({
                **c, **{k: t[k] for k in ("text", "text_tagged", "tags") if k in t},
                "asr_backend": backend, "asr_cer_agreement": agreement,
                "words": al["words"], "align_score": al["align_score"],
                "speaker_id": g, "gender": speakers[g]["gender"],
                "channel": vm.get("channel"), "video_title": vm.get("title"), "video_url": vm.get("url"),
                "code_switched": is_code_switched(t["text"]), "latin_words": len(latin_words(t["text"])),
            })
    bar.close()
    return rows, dict(stats)


def quality_score(r: dict) -> float:
    return (r["utmos"] - 3.0) + (r["dnsmos_ovrl"] - 3.0) + 0.5 * (r["align_score"] + 1.0) - 0.5 * r["weak_cut"]


def cap_speaker(rows: list[dict], cap_sec: float) -> list[dict]:
    """Best clips first, round-robin over the speaker's videos (content diversity)."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=quality_score, reverse=True):
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


def main(cfg: Section, args):
    lay = layout(cfg)
    b = cfg.balance
    rows, stats = candidates(cfg)
    if not rows:
        raise SystemExit("no candidate clips (run transcribe + align first)")
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    total_sec = sum(r["duration"] for r in rows)

    # iterate the share cap: capping big speakers shrinks the total
    cap_sec = b.max_hours_per_speaker * 3600
    for _ in range(5):
        capped = {s: min(sum(r["duration"] for r in rs), cap_sec) for s, rs in by_spk.items()}
        share_cap = b.max_speaker_share * sum(capped.values())
        new_cap = min(b.max_hours_per_speaker * 3600, share_cap) if b.max_speaker_share else cap_sec
        if abs(new_cap - cap_sec) < 1.0:
            break
        cap_sec = new_cap
    selected = {s: cap_speaker(rs, cap_sec) for s, rs in by_spk.items()}
    final = [r for rs in selected.values() for r in rs]
    if b.target_total_hours:
        final.sort(key=quality_score, reverse=True)
        keep, acc = [], 0.0
        for r in final:
            if acc >= b.target_total_hours * 3600:
                break
            keep.append(r)
            acc += r["duration"]
        final = keep

    # splits: unseen test speakers (mid-sized, both genders when possible) + seen validation
    rng = random.Random(1234)
    spk_hours = defaultdict(float)
    for r in final:
        spk_hours[r["speaker_id"]] += r["duration"] / 3600
    pool = [s for s, h in spk_hours.items() if h >= 0.05]
    pool.sort(key=lambda s: spk_hours[s])
    mid = pool[len(pool) // 4 : 3 * len(pool) // 4] or pool
    genders = defaultdict(list)
    for s in mid:
        genders[next(r["gender"] for r in by_spk[s])].append(s)
    test_spk: list[str] = []
    while len(test_spk) < min(b.test_speakers, max(0, len(pool) - 1)) and any(genders.values()):
        for g in list(genders):
            if genders[g] and len(test_spk) < b.test_speakers:
                test_spk.append(genders[g].pop(rng.randrange(len(genders[g]))))
    test, train_val = [], []
    per_test = defaultdict(float)
    for r in sorted(final, key=quality_score, reverse=True):
        if r["speaker_id"] in test_spk:
            if per_test[r["speaker_id"]] < b.test_max_minutes_per_speaker * 60:
                test.append(r)
                per_test[r["speaker_id"]] += r["duration"]
        else:
            train_val.append(r)
    rng.shuffle(train_val)
    n_val = int(len(train_val) * b.validation_fraction)
    validation, train = train_val[:n_val], train_val[n_val:]
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.sort(key=lambda r: r["id"])
        write_jsonl(lay.split(name), split)

    def h(rs):
        return round(sum(r["duration"] for r in rs) / 3600, 3)

    summary = {
        "candidates": len(rows), "candidate_hours": round(total_sec / 3600, 3), "rejections": stats,
        "speaker_cap_hours": round(cap_sec / 3600, 3),
        "splits": {n: {"clips": len(s), "hours": h(s), "speakers": len({r["speaker_id"] for r in s})}
                   for n, s in (("train", train), ("validation", validation), ("test", test))},
        "total_hours": h(train + validation + test),
        "speakers": len({r["speaker_id"] for r in final}),
    }
    write_json(lay.final / "balance_summary.json", summary)
    logger.info(f"final: {summary['total_hours']} h, {summary['speakers']} speakers, cap "
                f"{summary['speaker_cap_hours']} h/speaker | {summary['splits']}")


if __name__ == "__main__":
    step_main(main)
