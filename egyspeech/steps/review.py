"""review — a listening page for checking clips by ear: <work_dir>/review/index.html.

Random kept clips plus examples of every rejection reason, each with an audio player,
its scores, the silence measured at both edges, and a link to the moment in the
YouTube video. Open the file in a browser (on Windows: G:\\EgySpeech\\work\\review\\index.html
when work_dir is /mnt/g/EgySpeech/work). Re-run after changing filter thresholds.

    python -m egyspeech.cli review [--n 60] [--per-reason 8] [--seed 0]
"""

import html
import logging
import os
import random
from collections import defaultdict

from egyspeech.config import Section
from egyspeech.steps import layout, step_main, videos
from egyspeech.steps.filter import load_joined, reasons

logger = logging.getLogger("review")

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --ok:#1f7a3a; --bad:#b3261e; --chip:#f2f2f7; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#f2f2f7; --muted:#a1a1a6; --line:#2c2c2e;
  --ok:#5ad27d; --bad:#ff6b5e; --chip:#2c2c2e; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1200px;
  padding:16px; }
h1 { font-size:22px; margin:8px 0 4px; } h2 { font-size:17px; margin:28px 0 8px; }
.muted { color:var(--muted); } .ok { color:var(--ok); } .bad { color:var(--bad); }
.summary span { display:inline-block; background:var(--chip); border-radius:8px; padding:4px 10px; margin:3px 4px 3px 0; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line); padding:6px 8px;
  text-align:left; vertical-align:middle; white-space:nowrap; } th { color:var(--muted); font-weight:600; }
audio { height:32px; width:260px; } a { color:inherit; }
"""

COLS = ["clip", "audio", "dur", "spk", "OVRL", "SIG", "BAK", "UTMOS", "voice sim", "edge dB (start / end)",
        "cuts", "other spk", "video"]


def _row(r: dict, root: str, title: str, why: list[str]) -> str:
    rel = os.path.relpath(r["path"], root).replace(os.sep, "/")
    t = int(r["start"])
    url = f"https://www.youtube.com/watch?v={r['video_id']}&t={t}s" if len(r["video_id"]) == 11 else None
    video = f'<a href="{url}" target="_blank">{html.escape(title[:50])} @ {t // 60}:{t % 60:02d}</a>' if url else \
        html.escape(title[:50])
    edge = f"{r.get('edge_start_db', '–')} / {r.get('edge_end_db', '–')}"
    cls = "bad" if why else "ok"
    cells = [f'<span class="{cls}">{html.escape(r["id"])}</span>' + (f'<br><span class="bad">{", ".join(why)}</span>'
             if why else ""),
             f'<audio controls preload="none" src="{html.escape(rel)}"></audio>',
             f"{r['duration']:.1f}s", str(r["local_speaker"]), f"{r['dnsmos_ovrl']:.2f}", f"{r['dnsmos_sig']:.2f}",
             f"{r['dnsmos_bak']:.2f}", f"{r['utmos']:.2f}", f"{r['window_similarity']:.2f}", edge,
             f"{r['start_cut']} / {r['end_cut']}", f"{r.get('other_spk_max', 0):.2f}", video]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _table(rows: list[tuple[dict, list[str]]], root: str, titles: dict) -> str:
    head = "".join(f"<th>{c}</th>" for c in COLS)
    body = "\n".join(_row(r, root, titles.get(r["video_id"], r["video_id"]), why) for r, why in rows)
    return f'<div class="wrap"><table><tr>{head}</tr>{body}</table></div>'


def main(cfg: Section, args):
    lay = layout(cfg)
    rows = load_joined(cfg)
    if not rows:
        raise SystemExit("nothing to review: run segment, quality, speaker_check (and filter) first")
    judged = [(r, reasons(r, cfg.filter, cfg.segmentation)) for r in rows]
    kept = [x for x in judged if not x[1]]
    by_reason = defaultdict(list)
    for x in judged:
        for why in x[1]:
            by_reason[why].append(x)
    rng = random.Random(args.seed)
    titles = {v["id"]: v.get("title") or v["id"] for v in videos(cfg)}
    root = str(lay.review)
    lay.review.mkdir(parents=True, exist_ok=True)

    total_h = sum(r["duration"] for r in rows) / 3600
    kept_h = sum(r["duration"] for r, _ in kept) / 3600
    chips = [f"<span><b>{len(kept):,}</b> / {len(rows):,} clips kept</span>",
             f"<span><b>{kept_h:.2f} h</b> / {total_h:.2f} h kept</span>",
             f"<span>{len({r['video_id'] for r in rows})} videos</span>"]
    chips += [f'<span class="bad">{k}: {len(v):,}</span>' for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))]
    parts = [f"<h1>EgySpeech · clip review</h1><p class='muted'>Random sample (seed {args.seed}). "
             "Listen for: a second voice, a word cut at the start or end, music, noise. "
             "Edge dB = loudest 10 ms at the clip edge relative to its speech (silence ≈ -35 dB or lower).</p>",
             f'<div class="summary">{"".join(chips)}</div>',
             f"<h2 class='ok'>Kept ({min(args.n, len(kept))} of {len(kept):,})</h2>",
             _table(rng.sample(kept, min(args.n, len(kept))), root, titles)]
    for why, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        k = min(args.per_reason, len(items))
        parts += [f"<h2 class='bad'>Rejected: {why} ({k} of {len(items):,})</h2>",
                  _table(rng.sample(items, k), root, titles)]
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>EgySpeech review</title><style>{CSS}</style>"
            f"</head><body>{''.join(parts)}</body></html>")
    out = lay.review / "index.html"
    out.write_text(page, encoding="utf-8")
    logger.info(f"wrote {out} ({len(kept):,} kept / {len(rows):,} clips; open it in a browser)")


if __name__ == "__main__":
    def _args(p):
        p.add_argument("--n", type=int, default=60, help="kept clips to show")
        p.add_argument("--per-reason", type=int, default=8, help="rejected clips to show per reason")
        p.add_argument("--seed", type=int, default=0)

    step_main(main, _args)
