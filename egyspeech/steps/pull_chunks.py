"""pull_chunks — download the chunk dataset from the Hub onto this machine (e.g. a GPU pod).

Writes the clips to chunks/<vid>/<id>.flac and rebuilds what the later steps read:
meta/videos.jsonl, meta/chunks/, meta/quality/, meta/speakers/<vid>.npz and
meta/filtered.jsonl. Then `run --stage gpu` (transcribe -> publish) works as if the
local steps had run here. Shard by shard (download, unpack, delete): resumable, and it
needs little more disk than the clips themselves.
"""

import logging
from pathlib import Path

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, read_json, read_jsonl, write_json, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main
from egyspeech.steps.push_chunks import QUALITY_KEYS, VIDEO_KEYS

logger = logging.getLogger("pull_chunks")


def unpack(shard: Path, cfg: Section) -> tuple[int, float]:
    import pyarrow.parquet as pq

    lay = layout(cfg)
    by_video: dict[str, list[dict]] = {}
    pf = pq.ParquetFile(str(shard))
    for batch in pf.iter_batches(batch_size=256):
        for r in batch.to_pylist():
            vid = r["video_id"]
            path = lay.chunk_dir(vid) / f"{r['id']}.{cfg.segmentation.audio_format}"
            tmp = atomic_path(path).with_suffix(path.suffix)
            tmp.write_bytes(r.pop("audio")["bytes"])
            tmp.replace(path)
            r["path"] = str(path)
            by_video.setdefault(vid, []).append(r)
    n = sec = 0
    for vid, rows in by_video.items():
        rows.sort(key=lambda r: r["start"])
        chunk_rows = [{k: v for k, v in r.items() if k not in (*QUALITY_KEYS, *VIDEO_KEYS, "speaker_embedding",
                                                                "window_similarity")} for r in rows]
        write_jsonl(lay.quality(vid), [{"id": r["id"], **{k: r[k] for k in QUALITY_KEYS}} for r in rows])
        tmp = atomic_path(lay.speaker_emb(vid)).with_suffix(".npz")
        np.savez(tmp, ids=np.array([r["id"] for r in rows]),
                 embeddings=np.array([r["speaker_embedding"] for r in rows], dtype=np.float16),
                 window_similarity=np.array([r["window_similarity"] for r in rows], dtype=np.float32),
                 local_speaker=np.array([r["local_speaker"] for r in rows]))
        tmp.replace(lay.speaker_emb(vid))
        write_jsonl(lay.chunks_meta(vid), chunk_rows)  # written last: the video is complete
        n += len(rows)
        sec += sum(r["duration"] for r in rows)
    return n, sec


def main(cfg: Section, args):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    lay = layout(cfg)
    repo = cfg.hub.chunks_repo_id
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    dl_dir = lay.hub / "download"
    state_path = lay.hub / "pulled.json"
    state = read_json(state_path) if state_path.exists() and not args.force else {"repo_id": repo, "shards": []}

    videos_file = hf_hub_download(repo, "metadata/videos.jsonl", repo_type="dataset", local_dir=str(dl_dir))
    known = {v["id"]: v for v in read_jsonl(lay.videos)}
    known.update({v["id"]: v for v in read_jsonl(videos_file)})
    write_jsonl(lay.videos, sorted(known.values(), key=lambda v: v["id"]))

    todo = [f for f in files if f not in state["shards"]]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"{repo}: {len(files)} shards, {len(files) - len(todo)} already here, {len(todo)} to download")
    clips = 0
    with StepBar("pull_chunks", len(todo), unit="shards") as bar:
        for f in todo:
            bar.status(f"downloading {f}")
            local = Path(hf_hub_download(repo, f, repo_type="dataset", local_dir=str(dl_dir)))
            bar.status(f"unpacking {f}")
            n, _ = unpack(local, cfg)
            clips += n
            local.unlink()
            state["shards"].append(f)
            write_json(state_path, state)
            bar.advance()

    # every pulled clip passed the filter on the machine that pushed it
    from egyspeech.steps.filter import load_joined

    rows = load_joined(cfg)
    write_jsonl(lay.filtered, rows)
    logger.info(f"pulled {clips:,} clips this run; {len(rows):,} clips ({sum(r['duration'] for r in rows) / 3600:.1f} h) "
                f"ready in {lay.filtered}. Next: python -m egyspeech.cli run --stage gpu")


if __name__ == "__main__":
    step_main(main)
