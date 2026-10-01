"""push_chunks — upload the filtered clips (no transcripts yet) to a private Hugging Face dataset.

Every row is one clip: the audio (24 kHz FLAC, embedded) plus every column the later
steps need — segmentation info, quality scores, voice consistency, the TitaNet speaker
embedding, and the video's title / playlist / channel. On a GPU machine, `pull_chunks`
turns the dataset back into the pipeline's files and `transcribe` ... `publish` continue.

Shards are Parquet files holding whole videos. They are written, uploaded in commits of
hub.shards_per_commit and deleted locally, so the upload needs only a few GB of free
disk. hub/manifest.json records the uploaded videos: re-running continues where it
stopped and only adds new videos. --force deletes the remote shards and starts over
(needed after changing filter thresholds).

Authenticate first: `hf auth login` (or set HF_TOKEN).
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_json, read_jsonl, write_json, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main, videos

logger = logging.getLogger("push_chunks")

QUALITY_KEYS = ("dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "utmos")
VIDEO_KEYS = ("video_title", "playlist", "channel")


def chunk_features(sample_rate: int):
    from datasets import Audio, Features, List, Value

    f32, f64, i32, s, b = Value("float32"), Value("float64"), Value("int32"), Value("string"), Value("bool")
    return Features({
        "id": s, "audio": Audio(sampling_rate=sample_rate), "duration": f32,
        "video_id": s, "video_title": s, "playlist": s, "channel": s,
        "local_speaker": i32, "start": f64, "end": f64, "source": s, "music_db": f32, "input_lufs": f32,
        "clip_ratio": f32, "start_cut": s, "end_cut": s, "start_pause": f32, "end_pause": f32,
        "speech_ratio": f32, "max_internal_silence": f32, "weak_cut": b, "edge_start_db": f32,
        "edge_end_db": f32, "other_spk_max": f32, "dnsmos_p808": f32, "dnsmos_sig": f32, "dnsmos_bak": f32,
        "dnsmos_ovrl": f32, "utmos": f32, "window_similarity": f32, "speaker_embedding": List(Value("float32")),
    })


def write_shard(rows: list[dict], features, path: Path) -> None:
    """Parquet shard with the Hugging Face schema (the Hub reads `audio` as an Audio column).

    Written with pyarrow directly: the FLAC bytes are stored as they are (no re-encoding).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows, schema=features.arrow_schema)
    pq.write_table(table, str(path), row_group_size=100)  # small row groups: fast dataset viewer


def dataset_card(repo_id: str, manifest: dict, cfg: Section) -> str:
    f = cfg.filter
    return f"""---
language:
- ar
- arz
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: EgySpeech chunks (untranscribed)
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
---

# EgySpeech — single-speaker clips (before transcription)

{manifest['clips']:,} clips, {manifest['hours']:.1f} h of Egyptian Arabic speech from {len(manifest['videos']):,}
YouTube episodes, prepared by the [EgySpeech pipeline](https://github.com/MohammedAly22/EgySpeech):
Sortformer diarization, clips of {cfg.segmentation.min_sec:g}-{cfg.segmentation.max_sec:g} s cut inside pauses,
24 kHz mono FLAC loudness-normalized to {cfg.segmentation.loudness_lufs:g} LUFS.

Filter applied: DNSMOS OVRL >= {f.min_dnsmos_ovrl}, SIG >= {f.min_dnsmos_sig}, BAK >= {f.min_dnsmos_bak},
UTMOS >= {f.min_utmos}, voice consistency (TitaNet, every window) >= {f.min_window_similarity},
silence at both clip edges (<= {f.max_edge_db} dB relative to the speech).

Transcripts are added in the next stage (`python -m egyspeech.cli pull_chunks` on a GPU machine, then
`run --stage gpu`). `metadata/videos.jsonl` lists the source episodes.
"""


def main(cfg: Section, args):
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    lay = layout(cfg)
    h = cfg.hub
    repo = h.chunks_repo_id
    if not repo or repo.startswith("your-username/"):
        raise SystemExit("set hub.chunks_repo_id in the config (e.g. mohammedaly22/egyspeech-chunks)")
    rows = read_jsonl(lay.filtered)
    if not rows:
        raise SystemExit("no filtered clips: run the filter step first")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=bool(h.private), exist_ok=True)
    manifest_path = lay.hub / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() and not args.force else None
    if args.force:
        existing = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("data/")]
        if existing:
            logger.warning(f"--force: deleting {len(existing)} remote shards of {repo}")
            api.delete_folder("data", repo_id=repo, repo_type="dataset", commit_message="reset chunk shards")
    if manifest is None or manifest.get("repo_id") != repo:
        manifest = {"repo_id": repo, "next_shard": 0, "videos": {}, "clips": 0, "hours": 0.0}

    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)
    todo = [vid for vid in sorted(by_video) if vid not in manifest["videos"]]
    if args.limit:
        todo = todo[: args.limit]
    meta = {v["id"]: v for v in videos(cfg)}
    todo_sec = sum(r["duration"] for vid in todo for r in by_video[vid])
    logger.info(f"{len(manifest['videos'])} videos already on the Hub, {len(todo)} to upload "
                f"({todo_sec / 3600:.1f} h of clips) -> {repo} ({'private' if h.private else 'PUBLIC'})")

    features = chunk_features(cfg.download.sample_rate)
    shard_bytes = int(float(h.shard_mb) * 1e6)
    with tempfile.TemporaryDirectory(prefix="hub_", dir=str(lay.root)) as tmp, \
            StepBar("push_chunks", len(todo), audio_sec=todo_sec) as bar:
        staged: list[tuple[Path, int, list[str], int, float]] = []  # (file, index, videos, clips, seconds)
        buf: list[dict] = []
        buf_vids: list[str] = []
        size = 0

        def commit():
            if not staged:
                return
            gb = sum(p.stat().st_size for p, *_ in staged) / 1e9
            bar.status(f"uploading {len(staged)} shards ({gb:.1f} GB)")
            ops = [CommitOperationAdd(path_in_repo=f"data/train-{i:05d}.parquet", path_or_fileobj=str(p))
                   for p, i, *_ in staged]
            api.create_commit(repo, operations=ops, repo_type="dataset",
                              commit_message=f"add shards {staged[0][1]}-{staged[-1][1]}")
            for p, i, vids, n, sec in staged:
                for vid in vids:
                    manifest["videos"][vid] = i
                manifest["clips"] += n
                manifest["hours"] = round(manifest["hours"] + sec / 3600, 3)
                p.unlink()
            write_json(manifest_path, manifest)
            for _, _, vids, _, sec in staged:
                bar.advance(items=len(vids), audio_sec=sec)
            staged.clear()

        def flush():
            nonlocal buf, buf_vids, size
            if not buf:
                return
            i = manifest["next_shard"]
            manifest["next_shard"] += 1
            path = Path(tmp) / f"train-{i:05d}.parquet"
            bar.status(f"writing shard {i}")
            write_shard(buf, features, path)
            staged.append((path, i, buf_vids, len(buf), sum(r["duration"] for r in buf)))
            buf, buf_vids, size = [], [], 0
            if len(staged) >= int(h.shards_per_commit):
                commit()

        for vid in todo:
            z = np.load(lay.speaker_emb(vid))
            emb = dict(zip(z["ids"].tolist(), z["embeddings"].astype(np.float32), strict=True))
            v = meta.get(vid, {})
            for r in by_video[vid]:
                audio = Path(r["path"]).read_bytes()
                row = {k: r.get(k) for k in features if k not in ("audio", "speaker_embedding", *VIDEO_KEYS)}
                row.update(audio={"bytes": audio, "path": Path(r["path"]).name},
                           speaker_embedding=emb[r["id"]].tolist(), video_title=v.get("title"),
                           playlist=v.get("playlist"), channel=v.get("channel"))
                buf.append(row)
                size += len(audio)
            buf_vids.append(vid)
            if size >= shard_bytes:
                flush()
        flush()
        commit()

    pushed = set(manifest["videos"])
    with tempfile.TemporaryDirectory(prefix="hub_meta_", dir=str(lay.root)) as tmp:
        t = Path(tmp)
        write_jsonl(t / "videos.jsonl", [{k: v for k, v in meta[vid].items() if k not in ("path", "mtime")}
                                         for vid in sorted(pushed) if vid in meta])
        (t / "README.md").write_text(dataset_card(repo, manifest, cfg), encoding="utf-8")
        files = {"metadata/videos.jsonl": t / "videos.jsonl", "README.md": t / "README.md"}
        if lay.filter_report.exists():
            files["metadata/filter_report.json"] = lay.filter_report
        (t / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        files["metadata/manifest.json"] = t / "manifest.json"
        api.create_commit(repo, repo_type="dataset", commit_message="update metadata and card",
                          operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p))
                                      for k, p in files.items()])
    logger.info(f"done: {manifest['clips']:,} clips, {manifest['hours']:.1f} h from {len(pushed)} videos at "
                f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    step_main(main)
