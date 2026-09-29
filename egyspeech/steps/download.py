"""Step 2 — download: best audio stream -> standardized mono 24 kHz MP3 (+ metadata json).

Resumable: a video is done when its audio file exists. Failures are recorded in
meta/download_failures.jsonl and retried on the next run.
"""

import json
import logging
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from egyspeech.config import Section, resolve_repo_path
from egyspeech.io import atomic_path, ffmpeg_bin, read_jsonl, write_jsonl
from egyspeech.steps import layout, step_main

logger = logging.getLogger("download")

INFO_KEYS = ("id", "title", "channel", "channel_id", "uploader", "upload_date", "duration", "language",
             "categories", "tags", "description", "view_count", "webpage_url", "live_status")


def _ydl_opts(cfg: Section, outtmpl: str) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": cfg.download.retries,
        "fragment_retries": cfg.download.retries,
        "ffmpeg_location": str(Path(ffmpeg_bin()).parent),
        "overwrites": True,
    }
    if cfg.download.cookies_file:
        opts["cookiefile"] = str(resolve_repo_path(cfg.download.cookies_file))
    if cfg.download.proxy:
        opts["proxy"] = cfg.download.proxy
    return opts


def standardize(src: Path, dst: Path, cfg: Section):
    """Any audio -> mono, cfg sample rate, cfg format/bitrate (ffmpeg)."""
    d = cfg.download
    tmp = atomic_path(dst).with_suffix(dst.suffix)
    codec = {"mp3": ["-c:a", "libmp3lame", "-b:a", str(d.bitrate)], "flac": ["-c:a", "flac"],
             "wav": ["-c:a", "pcm_s16le"]}[d.format]
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-vn", "-ac", "1",
           "-ar", str(d.sample_rate), *codec, str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.replace(dst)


def download_one(video: dict, cfg: Section) -> tuple[str, str | None]:
    import yt_dlp

    lay = layout(cfg)
    vid = video["id"]
    dst = lay.raw_audio(vid, cfg.download.format)
    with tempfile.TemporaryDirectory(prefix=f"dl_{vid}_", dir=str(dst.parent)) as tmpdir:
        opts = _ydl_opts(cfg, str(Path(tmpdir) / "%(id)s.%(ext)s"))
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video["url"], download=False)
            if info is None:
                return vid, "no info (private / removed / geo-blocked?)"
            duration = info.get("duration") or 0
            if cfg.download.skip_live and info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
                return vid, f"skipped: live_status={info.get('live_status')}"
            if duration and not cfg.download.min_video_sec <= duration <= cfg.download.max_video_sec:
                return vid, f"skipped: duration {duration}s"
            ydl.process_info(info)
        files = [p for p in Path(tmpdir).iterdir() if p.is_file() and not p.name.endswith(".part")]
        if not files:
            return vid, "yt-dlp produced no file"
        standardize(files[0], dst, cfg)
    meta = {k: info.get(k) for k in INFO_KEYS}
    lay.info_json(vid).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    time.sleep(cfg.download.sleep_between_sec)
    return vid, None


def main(cfg: Section, args):
    lay = layout(cfg)
    videos = read_jsonl(lay.videos)
    if not videos:
        raise SystemExit("no videos: run the collect step first")
    lay.raw_audio("x").parent.mkdir(parents=True, exist_ok=True)
    pending = [v for v in videos if args.force or not lay.raw_audio(v["id"], cfg.download.format).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"{len(videos) - len(pending)} already downloaded, {len(pending)} to download")
    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.download.workers) as pool:
        futures = {pool.submit(download_one, v, cfg): v for v in pending}
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                vid, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                vid, err = v["id"], f"{type(exc).__name__}: {exc}"
            if err:
                failures.append({"id": vid, "url": v["url"], "error": err[:500]})
                logger.warning(f"{vid}: {err[:200]}")
            else:
                done += 1
                if done % 10 == 0:
                    logger.info(f"downloaded {done}/{len(pending)}")
    write_jsonl(lay.download_failures, failures)
    have = sum(lay.raw_audio(v["id"], cfg.download.format).exists() for v in videos)
    logger.info(f"done: {done} new downloads, {len(failures)} failures, {have}/{len(videos)} videos on disk")
    if any("confirm you" in f["error"].lower() or "sign in" in f["error"].lower() for f in failures):
        logger.warning("YouTube asked to sign in: export cookies.txt from a logged-in browser and set "
                       "download.cookies_file in the config")


if __name__ == "__main__":
    step_main(main)
