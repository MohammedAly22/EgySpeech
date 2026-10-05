"""pull_chunks — download the chunk dataset from the Hub onto this machine (e.g. a GPU pod).

Writes the chunks to chunks/<vid>/<id>.flac and rebuilds what the later steps read:
meta/videos.jsonl, meta/chunks/, meta/speakers/<vid>.npz (TitaNet embeddings, voice
consistency), meta/speakers.json (global speaker IDs + gender) and meta/quality/ when
the dataset already has DNSMOS / UTMOS. Then: `quality` -> `filter --max-hours N` ->
`transcribe` ... `publish`. Shard by shard (download, unpack, delete): resumable, and
needs little more disk than the chunks themselves.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, read_json, read_jsonl, write_json, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main
from egyspeech.steps.push_chunks import QUALITY_KEYS, VIDEO_KEYS

logger = logging.getLogger("pull_chunks")
SPEAKER_KEYS = ("speaker_id", "gender")


def unpack(shard: Path, cfg: Section) -> tuple[int, float, dict[str, tuple[str, str | None]]]:
    """Write one shard's chunks and metadata; returns (chunks, seconds, {chunk id: (speaker id, gender)})."""
    import pyarrow.parquet as pq

    lay = layout(cfg)
    by_video: dict[str, list[dict]] = {}
    for batch in pq.ParquetFile(str(shard)).iter_batches(batch_size=256):
        for r in batch.to_pylist():
            vid = r["video_id"]
            path = lay.chunk_dir(vid) / f"{r['id']}.{cfg.segmentation.audio_format}"
            tmp = atomic_path(path).with_suffix(path.suffix)
            tmp.write_bytes(r.pop("audio")["bytes"])
            tmp.replace(path)
            r["path"] = str(path)
            by_video.setdefault(vid, []).append(r)
    n = sec = 0
    speakers = {}
    for vid, rows in by_video.items():
        rows.sort(key=lambda r: r["start"])
        drop = (*QUALITY_KEYS, *VIDEO_KEYS, *SPEAKER_KEYS, "speaker_embedding", "window_similarity")
        chunk_rows = [{k: v for k, v in r.items() if k not in drop} for r in rows]
        if all(r.get("dnsmos_bak") is not None for r in rows):  # quality already computed
            write_jsonl(lay.quality(vid), [{"id": r["id"], **{k: r[k] for k in QUALITY_KEYS}} for r in rows])
        tmp = atomic_path(lay.speaker_emb(vid)).with_suffix(".npz")
        np.savez(tmp, ids=np.array([r["id"] for r in rows]),
                 embeddings=np.array([r["speaker_embedding"] for r in rows], dtype=np.float16),
                 window_similarity=np.array([r["window_similarity"] for r in rows], dtype=np.float32),
                 local_speaker=np.array([r["local_speaker"] for r in rows]))
        tmp.replace(lay.speaker_emb(vid))
        write_jsonl(lay.chunks_meta(vid), chunk_rows)  # written last: the video is complete
        for r in rows:
            speakers[r["id"]] = (r.get("speaker_id"), r.get("gender"))
        n += len(rows)
        sec += sum(r["duration"] for r in rows)
    return n, sec, speakers


def write_speakers(cfg: Section, assignment: dict[str, tuple[str, str | None]]) -> None:
    """meta/speakers.json from the per-chunk speaker IDs (same format as the cluster step)."""
    lay = layout(cfg)
    hours: dict[str, float] = defaultdict(float)
    n_clips: dict[str, int] = defaultdict(int)
    vids: dict[str, set] = defaultdict(set)
    gender: dict[str, str | None] = {}
    for vid_rows in (read_jsonl(lay.chunks_meta(v["id"])) for v in read_jsonl(lay.videos)
                     if lay.chunks_meta(v["id"]).exists()):
        for r in vid_rows:
            sid, g = assignment.get(r["id"], (None, None))
            if sid:
                hours[sid] += r["duration"] / 3600
                n_clips[sid] += 1
                vids[sid].add(r["video_id"])
                gender[sid] = g
    speakers = [{"id": s, "gender": gender[s], "hours": round(hours[s], 4), "n_clips": n_clips[s],
                 "n_videos": len(vids[s]), "videos": sorted(vids[s])} for s in sorted(hours, key=lambda s: -hours[s])]
    write_json(lay.speakers, {"speakers": speakers, "assignment": {c: s for c, (s, _) in assignment.items() if s}})


def main(cfg: Section, args):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    lay = layout(cfg)
    repo = cfg.hub.chunks_repo_id
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                   if f.startswith("chunks/") and f.endswith(".parquet"))
    dl_dir = lay.hub / "download"
    state_path = lay.hub / "pulled.json"
    state = read_json(state_path) if state_path.exists() and not args.force else {"repo_id": repo, "shards": []}
    assign_path = lay.hub / "pulled_speakers.json"
    assignment = {k: tuple(v) for k, v in read_json(assign_path).items()} if assign_path.exists() and not args.force \
        else {}

    videos_file = hf_hub_download(repo, "metadata/videos.jsonl", repo_type="dataset", local_dir=str(dl_dir))
    known = {v["id"]: v for v in read_jsonl(lay.videos)}
    known.update({v["id"]: v for v in read_jsonl(videos_file)})
    write_jsonl(lay.videos, sorted(known.values(), key=lambda v: v["id"]))

    todo = [f for f in files if f not in state["shards"]]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"{repo}: {len(files)} chunk shards, {len(files) - len(todo)} already here, {len(todo)} to download")
    clips = 0
    with StepBar("pull_chunks", len(todo), unit="shards") as bar:
        for f in todo:
            bar.status(f"downloading {f}")
            local = Path(hf_hub_download(repo, f, repo_type="dataset", local_dir=str(dl_dir)))
            bar.status(f"unpacking {f}")
            n, _, spk = unpack(local, cfg)
            assignment.update(spk)
            clips += n
            local.unlink()
            state["shards"].append(f)
            write_json(assign_path, assignment)
            write_json(state_path, state)
            bar.advance()
    write_speakers(cfg, assignment)
    spk = read_json(lay.speakers)["speakers"]
    hours = sum(s["hours"] for s in spk)
    logger.info(f"pulled {clips:,} chunks this run; {len(assignment):,} chunks, {hours:.1f} h, {len(spk):,} speakers "
                "on disk. Next: python -m egyspeech.cli quality")


if __name__ == "__main__":
    step_main(main)
