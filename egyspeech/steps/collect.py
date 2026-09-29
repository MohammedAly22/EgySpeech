"""Step 1 — collect: expand the links file (playlists, channels, single videos) into unique videos.

Re-running adds videos from new links / newly published playlist items and keeps
the existing list untouched.
"""

import logging
import re

from egyspeech.config import Section, resolve_repo_path
from egyspeech.io import read_jsonl, write_jsonl
from egyspeech.steps import layout, step_main

logger = logging.getLogger("collect")

_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([A-Za-z0-9_-]{11})")


def read_links(path) -> list[str]:
    links, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip() if not line.strip().startswith("http") else line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            links.append(line)
    return links


def expand(url: str, cfg: Section) -> list[dict]:
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }
    if cfg.download.cookies_file:
        opts["cookiefile"] = str(resolve_repo_path(cfg.download.cookies_file))
    if cfg.download.proxy:
        opts["proxy"] = cfg.download.proxy
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        return []
    stack, videos = [info], []
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if item.get("_type") in ("playlist", "multi_video") or item.get("entries") is not None:
            stack.extend(reversed(list(item.get("entries") or [])))
            continue
        vid = item.get("id")
        if not vid or len(vid) != 11:
            m = _ID.search(item.get("url") or item.get("webpage_url") or "")
            vid = m.group(1) if m else None
        if not vid:
            continue
        videos.append(
            {
                "id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": item.get("title"),
                "channel": item.get("channel") or item.get("uploader"),
                "channel_id": item.get("channel_id"),
                "duration": item.get("duration"),
                "live_status": item.get("live_status"),
                "source": url,
                "playlist": info.get("title") if info.get("_type") == "playlist" else None,
            }
        )
    return videos


def main(cfg: Section, args):
    lay = layout(cfg)
    links_path = resolve_repo_path(cfg.links_file)
    if not links_path.exists():
        raise SystemExit(f"links file not found: {links_path}")
    known = {v["id"]: v for v in read_jsonl(lay.videos)}
    before = len(known)
    for link in read_links(links_path):
        try:
            found = expand(link, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad link must not stop the others
            logger.warning(f"could not expand {link}: {exc}")
            continue
        new = 0
        for v in found:
            if v["id"] not in known:
                known[v["id"]] = v
                new += 1
        logger.info(f"{link} -> {len(found)} videos ({new} new)")
    write_jsonl(lay.videos, known.values())
    hours = sum((v.get("duration") or 0) for v in known.values()) / 3600
    logger.info(f"{len(known)} unique videos ({len(known) - before} new), ~{hours:.1f} h listed")


if __name__ == "__main__":
    step_main(main)
