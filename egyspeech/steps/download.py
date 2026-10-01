"""Step 1 — download: links file -> unique videos -> audio as FLAC, 24 kHz, mono.

Uses the settings of the downloader that proved reliable at scale (no cookies, no
rate-limit workarounds needed):
  * discovery: every playlist is expanded flat (no audio), videos are de-duplicated by ID;
  * download: `bestaudio`, converted by ffmpeg to FLAC 24 kHz mono, IPv4, one video at a time;
  * yt-dlp's download archive (<dir>/.download_archive.txt) makes re-runs skip finished videos.

Layout: playlist videos go to <dir>/<playlist title>/, single videos to <dir>/, as
"<title> [<video id>].flac". YouTube metadata goes to meta/video_info/<id>.json.
Run the `index` step afterwards (any folder of "<title> [<id>].<ext>" files works).
"""

import json
import logging
import re
from pathlib import Path

from rich.progress import (BarColumn, DownloadColumn, Progress, ProgressColumn, TextColumn, TimeElapsedColumn,
                           TransferSpeedColumn)
from rich.text import Text

from egyspeech.config import Section, resolve_repo_path
from egyspeech.io import atomic_path, ffmpeg_bin
from egyspeech.progress import console
from egyspeech.steps import layout, raw_dir, step_main, write_failures

logger = logging.getLogger("download")

ID_IN_NAME = re.compile(r"\[([A-Za-z0-9_-]{11})\]\.[A-Za-z0-9]+$")
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class _YdlLogger:
    """yt-dlp messages -> our log (errors only; progress is shown by the bar)."""

    def debug(self, msg: str):
        pass

    def info(self, msg: str):
        pass

    def warning(self, msg: str):
        pass

    def error(self, msg: str):
        logger.warning(msg)


def read_links(path: Path) -> list[str]:
    links, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            links.append(line)
    return links


def discovery_opts() -> dict:
    """Flat extraction: playlists are listed without touching any audio."""
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
        "source_address": "0.0.0.0",  # IPv4 (helps under WSL)
        "logger": _YdlLogger(),
    }


def _entry(e: dict, playlist: str | None) -> dict:
    vid = e["id"]
    return {"id": vid, "url": e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
            "title": e.get("title") or vid, "playlist": playlist,
            "channel": e.get("channel") or e.get("uploader"), "channel_id": e.get("channel_id"),
            "duration": e.get("duration"), "live_status": e.get("live_status")}


def expand(ydl, url: str, depth: int = 0) -> list[dict]:
    """Videos behind a link; playlists (and channel tabs, one level deep) are expanded."""
    info = ydl.extract_info(url, download=False)
    if not info:
        return []
    if info.get("_type") != "playlist":
        return [_entry(info, None)] if info.get("id") else []
    title = info.get("title") or "Unknown Playlist"
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        if e.get("_type") == "url" and e.get("ie_key") == "YoutubeTab" and depth < 1:
            out.extend(expand(ydl, e["url"], depth + 1))  # channel -> its Videos / Live tabs
        elif len(e["id"]) == 11:
            out.append(_entry(e, title))
    return out


def discover(links: list[str]) -> list[dict]:
    import yt_dlp

    found: list[dict] = []
    seen: set[str] = set()
    with yt_dlp.YoutubeDL(discovery_opts()) as ydl:
        for i, url in enumerate(links, 1):
            try:
                videos = expand(ydl, url)
            except Exception as exc:  # noqa: BLE001 - one bad link must not stop the others
                logger.warning(f"({i}/{len(links)}) could not expand {url}: {exc}")
                continue
            new = [v for v in videos if v["id"] not in seen]
            seen.update(v["id"] for v in new)
            found.extend(new)
            logger.info(f"({i}/{len(links)}) {url}: {len(new)} new videos"
                        + (f", {len(videos) - len(new)} duplicates skipped" if len(videos) > len(new) else ""))
    return found


def on_disk_ids(root: Path) -> set[str]:
    """Video IDs already present as "<title> [<id>].<ext>" anywhere under root."""
    if not root.exists():
        return set()
    return {m.group(1) for p in root.rglob("*") if (m := ID_IN_NAME.search(p.name)) and ".temp." not in p.name}


def archived_ids(archive: Path) -> set[str]:
    if not archive.exists():
        return set()
    return {parts[1] for line in archive.read_text(encoding="utf-8").splitlines() if len(parts := line.split()) == 2}


def safe_dirname(name: str, max_bytes: int = 120) -> str:
    name = _UNSAFE.sub("_", name).strip(" .") or "playlist"
    while len(name.encode("utf-8")) > max_bytes:
        name = name[:-1]
    return name


class _FileOnly(ProgressColumn):
    """Show a column only on the per-file bar (bytes, speed), not on the overall one."""

    def __init__(self, column: ProgressColumn):
        super().__init__()
        self.column = column

    def render(self, task):
        return self.column.render(task) if task.fields.get("file") else Text("")


class _Hook:
    """yt-dlp progress -> the current-file bar (bytes, speed), then the FLAC conversion."""

    def __init__(self, progress: Progress, task):
        self.progress, self.task = progress, task

    def download(self, d: dict):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            self.progress.update(self.task, completed=d.get("downloaded_bytes") or 0, total=total)
        elif d.get("status") == "finished":
            self.progress.update(self.task, description="[yellow]converting to FLAC")

    def postprocess(self, d: dict):
        if d.get("status") == "finished":
            self.progress.update(self.task, description="[green]done")


def download_opts(cfg: Section, out_dir: Path, archive: Path, hook: _Hook) -> dict:
    d = cfg.download
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title).180B [%(id)s].%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": d.audio_format}],
        "postprocessor_args": ["-ar", str(d.sample_rate), "-ac", str(d.channels)],
        "progress_hooks": [hook.download],
        "postprocessor_hooks": [hook.postprocess],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": False,  # raise, so failures are counted
        "source_address": "0.0.0.0",
        "download_archive": str(archive),
        "retries": d.retries,
        "fragment_retries": d.retries,
        "logger": _YdlLogger(),
    }
    try:
        opts["ffmpeg_location"] = ffmpeg_bin()
    except RuntimeError:
        pass  # yt-dlp looks on PATH itself
    return opts


def save_info(cfg: Section, v: dict):
    path = layout(cfg).video_info(v["id"])
    tmp = atomic_path(path)
    tmp.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main(cfg: Section, args):
    import yt_dlp

    d = cfg.download
    root = raw_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ".download_archive.txt"
    links_path = resolve_repo_path(cfg.links_file)
    if not links_path.exists():
        raise SystemExit(f"links file not found: {links_path}")
    links = read_links(links_path)
    logger.info(f"{len(links)} links in {links_path} -> {root} ({d.audio_format.upper()}, "
                f"{d.sample_rate / 1000:g} kHz, {'mono' if d.channels == 1 else f'{d.channels} ch'})")

    found = discover(links)
    for v in found:
        save_info(cfg, v)
    have = on_disk_ids(root) | archived_ids(archive)
    skipped, todo = [], []
    for v in found:
        dur, live = v.get("duration"), v.get("live_status")
        if v["id"] in have and not args.force:
            continue
        if d.skip_live and live in ("is_live", "is_upcoming"):
            skipped.append((v, f"live_status={live}"))
        elif dur and not d.min_video_sec <= dur <= d.max_video_sec:
            skipped.append((v, f"duration {dur:.0f}s"))
        else:
            todo.append(v)
    if args.limit:
        todo = todo[: args.limit]
    hours = sum(v.get("duration") or 0 for v in todo) / 3600
    logger.info(f"{len(found)} unique videos: {len(found) - len(todo) - len(skipped)} already downloaded, "
                f"{len(skipped)} skipped (live / duration), {len(todo)} to download (~{hours:.1f} h)")

    failures, ok = [], 0
    columns = [TextColumn("{task.description}"), BarColumn(bar_width=30), TextColumn("{task.percentage:>5.1f}%"),
               _FileOnly(DownloadColumn()), _FileOnly(TransferSpeedColumn()), TimeElapsedColumn()]
    with Progress(*columns, console=console) as progress:
        overall = progress.add_task("[bold cyan]download", total=len(todo))
        for i, v in enumerate(todo, 1):
            out_dir = root / safe_dirname(v["playlist"]) if v.get("playlist") else root
            file_task = progress.add_task(f"[cyan]{v['id']}", total=None, file=True)
            progress.update(overall, description=f"[bold cyan]download {i}/{len(todo)}")
            try:
                with yt_dlp.YoutubeDL(download_opts(cfg, out_dir, archive, _Hook(progress, file_task))) as ydl:
                    if ydl.download([v["url"]]) != 0:
                        raise RuntimeError("yt-dlp returned an error")
                ok += 1
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                failures.append({"id": v["id"], "url": v["url"], "title": v["title"], "error": str(exc)[:500]})
                logger.warning(f"{v['id']} ({v['title'][:60]}): {str(exc)[:200]}")
            finally:
                progress.remove_task(file_task)
                progress.advance(overall)
    write_failures(cfg, "download", failures)
    logger.info(f"done: {ok} downloaded, {len(failures)} failed (meta/failures/download.jsonl; "
                f"re-run to retry). Next: the `index` step.")


if __name__ == "__main__":
    step_main(main)
