"""Step 2 — index: list the episodes on disk -> meta/videos.jsonl.

Scans download.dir recursively for audio files named "<title> [<video id>].<ext>"
(what the download step and yt-dlp write), reads each file's duration from its
header, and records one row per video:
    {"id", "path", "duration", "sample_rate", "channels", "title", "playlist", "channel", ...}
Files are used where they are (nothing is copied). Files still being written
(modified in the last index.min_age_sec seconds, or yt-dlp temp files) are skipped.
Cheap and safe to re-run: new downloads are picked up, unchanged files are not re-read.
"""

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from egyspeech.config import Section
from egyspeech.io import read_jsonl, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.steps import layout, raw_dir, step_main

logger = logging.getLogger("index")

ID_IN_NAME = re.compile(r"^(?P<title>.*?)\s*\[(?P<id>[A-Za-z0-9_-]{11})\]$")
TEMP_MARKERS = (".temp.", ".part", ".ytdl", ".tmp")


def parse_name(path: Path, root: Path) -> dict:
    """Video ID, title and playlist (parent folder) from "<title> [<id>].<ext>"."""
    m = ID_IN_NAME.match(path.stem)
    if m:
        vid, title, url = m.group("id"), m.group("title"), f"https://www.youtube.com/watch?v={m.group('id')}"
    else:  # not a yt-dlp name: stable ID from the relative path
        vid = "f" + hashlib.sha1(str(path.relative_to(root)).encode()).hexdigest()[:10]
        title, url = path.stem, None
    rel_parent = path.parent.relative_to(root)
    playlist = None if str(rel_parent) in (".", "NA") else str(rel_parent)  # "NA": yt-dlp's "no playlist"
    return {"id": vid, "title": title.replace("⧸", "/").replace("｜", "|").strip(), "playlist": playlist, "url": url}


def probe(path: Path) -> dict | None:
    try:
        i = sf.info(str(path))
        return {"duration": round(float(i.duration), 3), "sample_rate": i.samplerate, "channels": i.channels}
    except Exception:  # noqa: BLE001 - not readable by libsndfile: ask ffprobe
        import subprocess

        from egyspeech.io import ffmpeg_bin

        ffprobe = str(Path(ffmpeg_bin()).with_name("ffprobe"))
        try:
            out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                  "stream=sample_rate,channels:format=duration", "-of", "json", str(path)],
                                 capture_output=True, check=True, text=True).stdout
            j = json.loads(out)
            st = j["streams"][0]
            return {"duration": round(float(j["format"]["duration"]), 3), "sample_rate": int(st["sample_rate"]),
                    "channels": int(st["channels"])}
        except Exception:  # noqa: BLE001
            return None


def main(cfg: Section, args):
    lay = layout(cfg)
    root = raw_dir(cfg)
    if not root.exists():
        raise SystemExit(f"download.dir not found: {root}")
    exts = {"." + e.lower().lstrip(".") for e in cfg.index.extensions}
    now = time.time()
    files, young = [], 0
    for p in root.rglob("*"):
        if p.suffix.lower() not in exts or any(t in p.name for t in TEMP_MARKERS) or p.name.startswith("."):
            continue
        st = p.stat()
        if now - st.st_mtime < cfg.index.min_age_sec:
            young += 1
            continue
        files.append((p, st))
    files.sort(key=lambda x: str(x[0]))
    previous = {} if args.force else {r["path"]: r for r in read_jsonl(lay.videos)}

    def describe(item):
        p, st = item
        old = previous.get(str(p))
        if old and old.get("size") == st.st_size and old.get("mtime") == int(st.st_mtime):
            return old
        info = probe(p)
        if info is None:
            return {"path": str(p), "error": "unreadable"}
        row = {**parse_name(p, root), "path": str(p), **info, "size": st.st_size, "mtime": int(st.st_mtime)}
        meta = lay.video_info(row["id"])
        if meta.exists():  # YouTube metadata saved by the download step
            j = json.loads(meta.read_text(encoding="utf-8"))
            row.update({k: j.get(k) for k in ("title", "channel", "channel_id", "url") if j.get(k)})
            row["playlist"] = j.get("playlist") or row["playlist"]
        return row

    rows, bad, dup, short = [], [], 0, 0
    seen: dict[str, str] = {}
    with StepBar("index", n_items=len(files), unit="files") as bar, ThreadPoolExecutor(16) as pool:
        for row in pool.map(describe, files):
            bar.advance()
            if "error" in row:
                bad.append(row["path"])
                continue
            if row["id"] in seen:
                dup += 1
                logger.warning(f"duplicate video {row['id']}: keeping {seen[row['id']]}, ignoring {row['path']}")
                continue
            if row["duration"] < cfg.download.min_video_sec:
                short += 1
                continue
            seen[row["id"]] = row["path"]
            rows.append(row)
    rows.sort(key=lambda r: r["id"])  # stable, channel-mixed order (what --limit N picks first)
    write_jsonl(lay.videos, rows)
    for p in bad:
        logger.warning(f"unreadable, skipped: {p}")
    rates = sorted({r["sample_rate"] for r in rows})
    logger.info(f"{len(rows)} episodes, {sum(r['duration'] for r in rows) / 3600:.1f} h in {root} | sample rates "
                f"{rates} | skipped: {short} shorter than {cfg.download.min_video_sec}s, {dup} duplicates, "
                f"{len(bad)} unreadable, {young} still being written")


if __name__ == "__main__":
    step_main(main)
