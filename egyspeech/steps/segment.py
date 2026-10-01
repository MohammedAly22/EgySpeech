"""Step 5 — segment: single-speaker 5-30 s clips cut in silences, loudness normalized.

Per video (CPU, several videos in parallel):
  1. decode the episode at 16 kHz; Silero VAD (run as parallel streams: ~3x faster) and
     frame energy;
  2. Sortformer probabilities + VAD + energy -> the pause-aware planner
     (egyspeech/segmenter.py): only frames where exactly one speaker talks, a guard
     distance from every other voice, boundaries only inside real pauses;
  3. every clip is read straight from the episode file at 24 kHz (seek, no full decode),
     measured (silence at both edges, other-speaker activity, clipping), loudness
     normalized, faded in / out over a few ms, and written as FLAC.

Workers share a RAM budget (segmentation.memory_gb): a 4 h episode needs ~2.5 GB,
so long episodes never run out of memory together.
--watch keeps running while `diarize` works in another terminal and segments every
newly diarized video as soon as it is ready.
"""

import logging
import time
from contextlib import nullcontext

import numpy as np

from egyspeech.config import Section
from egyspeech.io import ClipReader, decode_audio, read_json, read_jsonl, write_audio, write_jsonl
from egyspeech.progress import StepBar
from egyspeech.segmenter import HOP, SegParams, plan_clips, to_grid
from egyspeech.steps import layout, select, speech_audio, step_main, total_sec, videos, write_failures
from egyspeech.steps.separate import music_db

logger = logging.getLogger("segment")

_VAD = None
_METERS: dict = {}
VAD_WIN = 512  # samples at 16 kHz = 32 ms
VAD_CONTEXT = 94  # windows (3 s) of warm-up audio before each parallel stream


def vad_probs(wav16: np.ndarray, streams: int = 16) -> tuple[np.ndarray, float]:
    """Silero VAD speech probability per 32 ms window.

    Long recordings are split into `streams` consecutive parts processed as one batch
    (vectorized: ~3x faster on one CPU core). Each part starts with 3 s of the audio
    before it, so the recurrent state has settled when its own windows begin.
    """
    global _VAD
    import torch
    from silero_vad import load_silero_vad

    if _VAD is None:
        _VAD = load_silero_vad()
    n = len(wav16) // VAD_WIN
    if n == 0:
        return np.zeros(0, dtype=np.float32), VAD_WIN / 16000
    with torch.inference_mode():
        if streams <= 1 or n < streams * VAD_CONTEXT * 4:
            probs = _VAD.audio_forward(torch.from_numpy(np.ascontiguousarray(wav16[: n * VAD_WIN])), 16000)[0]
            return probs.numpy()[:n].astype(np.float32), VAD_WIN / 16000
        per = -(-n // streams)  # windows per stream
        batch = np.zeros((streams, (VAD_CONTEXT + per) * VAD_WIN), dtype=np.float32)
        for b in range(streams):
            first = b * per - VAD_CONTEXT  # first window of this stream, warm-up included
            lo, hi = max(0, first), min(n, (b + 1) * per)
            if lo < hi:
                off = (lo - first) * VAD_WIN
                batch[b, off : off + (hi - lo) * VAD_WIN] = wav16[lo * VAD_WIN : hi * VAD_WIN]
        out = _VAD.audio_forward(torch.from_numpy(batch), 16000).numpy()
    return out[:, VAD_CONTEXT:].reshape(-1)[:n].astype(np.float32), VAD_WIN / 16000


def energy_db(wav: np.ndarray, sr: int) -> np.ndarray:
    """Frame energy (dB) on the HOP grid, smoothed over 3 frames; computed in blocks (bounded RAM)."""
    hop = int(sr * HOP)
    n = len(wav) // hop
    out = np.empty(n, dtype=np.float32)
    block = 60_000  # frames (10 min)
    for a in range(0, n, block):
        b = min(n, a + block)
        fr = wav[a * hop : b * hop].reshape(b - a, hop)
        out[a:b] = 10 * np.log10(np.einsum("ij,ij->i", fr, fr) / hop + 1e-10)
    return np.convolve(out, np.ones(3, dtype=np.float32) / 3, mode="same").astype(np.float32)


def seg_params(cfg: Section) -> SegParams:
    s, d = cfg.segmentation, cfg.diarization
    return SegParams(
        min_sec=s.min_sec, max_sec=s.max_sec, target_sec=s.target_sec,
        speaker_threshold=d.speaker_threshold, other_speaker_threshold=d.other_speaker_threshold,
        vad_threshold=s.vad_threshold, vad_off_threshold=s.vad_off_threshold,
        min_pause_sec=s.min_pause_sec, strong_pause_sec=s.strong_pause_sec,
        allow_weak_cuts=s.allow_weak_cuts, max_internal_silence_sec=s.max_internal_silence_sec,
        guard_sec=s.guard_sec, edge_pad_sec=s.edge_pad_sec,
    )


def loudness_normalize(x: np.ndarray, sr: int, lufs: float, peak_dbfs: float) -> tuple[np.ndarray, float]:
    import pyloudnorm as pyln

    meter = _METERS.get(sr) or _METERS.setdefault(sr, pyln.Meter(sr))
    loud = meter.integrated_loudness(x.astype(np.float64))
    if np.isfinite(loud):
        x = x * (10 ** ((lufs - loud) / 20))
    ceiling = 10 ** (peak_dbfs / 20)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > ceiling:
        x = x * (ceiling / peak)
    return x.astype(np.float32), float(loud)


def edge_levels(x: np.ndarray, sr: int, edge_sec: float = 0.04) -> tuple[float, float]:
    """Loudest 10 ms frame in the first / last `edge_sec`, in dB relative to the clip's speech level.

    A clip that starts and ends in silence gives values far below 0 (typically -35 to
    -60 dB); a clip cut inside a word gives values near 0.
    """
    hop = int(0.01 * sr)
    n = len(x) // hop
    if n < 10:
        return 0.0, 0.0
    fr = x[: n * hop].reshape(n, hop).astype(np.float64)
    db = 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
    level = float(np.percentile(db, 90))
    k = max(1, int(round(edge_sec / 0.01)))
    return round(float(db[:k].max()) - level, 1), round(float(db[-k:].max()) - level, 1)


def fade(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    k = int(sr * ms / 1000)
    if k > 0 and len(x) > 4 * k:
        ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)).astype(np.float32)
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def memory_gb(video: dict) -> float:
    """Peak RAM of one worker for this episode: 16 kHz audio + the VAD batch + the process itself."""
    return 0.5 + 0.5 * (video.get("duration") or 3600) / 3600


def process_video(v: dict, cfg: Section) -> list[dict]:
    lay = layout(cfg)
    vid, sr, seg = v["id"], cfg.download.sample_rate, cfg.segmentation
    src = speech_audio(cfg, v)
    separated = cfg.separation.mode != "never"

    wav16 = decode_audio(src, 16000, v.get("duration"))
    vad, vad_sec = vad_probs(wav16, int(seg.vad_streams))
    db = energy_db(wav16, 16000)
    del wav16
    diar = np.load(lay.diar_probs(vid)).astype(np.float32)
    frame_sec = read_json(lay.diar_json(vid))["frame_sec"]
    n = min(len(db), int(len(vad) * vad_sec / HOP), int(diar.shape[0] * frame_sec / HOP))
    grid_diar = to_grid(diar, frame_sec, n)
    clips = plan_clips(grid_diar, to_grid(vad, vad_sec, n), db[:n], seg_params(cfg))

    out_dir = lay.chunk_dir(vid)
    if out_dir.exists():  # re-segmenting: never leave clips from an older plan behind
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counters: dict[int, int] = {}
    with ClipReader(src, sr) as speech, (ClipReader(v["path"], sr) if separated else nullcontext()) as raw:
        for clip in clips:
            x = speech.read(clip.start, clip.end)
            mdb, source = None, "original"
            if separated:  # keep the untouched original wherever (almost) no music was removed
                r = raw.read(clip.start, clip.end)
                k = min(len(r), len(x))
                mdb = round(music_db(r[:k], x[:k]), 2)
                x, source = (r, "original") if mdb <= cfg.separation.use_original_below_music_db else (x, "vocals")
            if len(x) < int(seg.min_sec * sr * 0.9):
                continue
            clip_ratio = float(np.mean(np.abs(x) >= 0.999))
            edge_start, edge_end = edge_levels(x, sr)
            x, loud = loudness_normalize(x, sr, seg.loudness_lufs, seg.peak_dbfs)
            x = fade(x, sr, seg.fade_ms)
            a, b = int(round(clip.start / HOP)), int(round(clip.end / HOP))
            others = np.delete(grid_diar[a:b], clip.speaker, axis=1)
            k = counters.get(clip.speaker, 0)
            counters[clip.speaker] = k + 1
            cid = f"{vid}_s{clip.speaker}_{k:04d}"
            path = out_dir / f"{cid}.{seg.audio_format}"
            write_audio(path, x, sr)
            rows.append({
                "id": cid, "video_id": vid, "local_speaker": clip.speaker, "path": str(path),
                "start": clip.start, "end": clip.end, "duration": round(len(x) / sr, 3),
                "source": source, "music_db": mdb,
                "input_lufs": round(loud, 2) if np.isfinite(loud) else None, "clip_ratio": clip_ratio,
                "start_cut": clip.start_cut, "end_cut": clip.end_cut,
                "start_pause": clip.start_pause, "end_pause": clip.end_pause,
                "speech_ratio": clip.speech_ratio, "max_internal_silence": clip.max_internal_silence,
                "weak_cut": "weak" in (clip.start_cut, clip.end_cut),
                "edge_start_db": edge_start, "edge_end_db": edge_end,
                "other_spk_max": round(float(others.max()) if others.size else 0.0, 3),
            })
    return rows


def _worker(v: dict, cfg_path: str) -> tuple[int, float]:
    import torch

    from egyspeech.config import load_config

    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    rows = process_video(v, cfg)
    lay = layout(cfg)
    write_jsonl(lay.chunks_meta(v["id"]), rows)  # written last: marks the video as done
    if cfg.storage.delete_vocals_after_segment and cfg.separation.mode != "never":
        lay.vocals(v["id"]).unlink(missing_ok=True)
    return len(rows), sum(r["duration"] for r in rows) / 3600


def run_batch(cfg: Section, pending: list[dict], failures: list[dict]) -> float:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from egyspeech.parallel import run_with_budget

    seg = cfg.segmentation
    workers = max(1, int(seg.workers))
    jobs = [(v, memory_gb(v), (v, cfg["_path"])) for v in pending]
    clip_hours = 0.0
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool, \
            StepBar("segment", len(pending), audio_sec=total_sec(pending)) as bar:
        for v, fut in run_with_budget(pool, _worker, jobs, workers, float(seg.memory_gb)):
            try:
                n, hours = fut.result()
                clip_hours += hours
                share = hours * 3600 / max(v["duration"], 1)
                logger.info(f"{v['id']}: {n} clips, {hours:.2f} h ({share:.0%} of {v['duration'] / 3600:.2f} h)")
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                failures.append({"id": v["id"], "path": v["path"], "error": f"{type(exc).__name__}: {exc}"[:500]})
                logger.warning(f"{v['id']}: {type(exc).__name__}: {str(exc)[:200]}")
            bar.advance(audio_sec=v["duration"])
    return clip_hours


def main(cfg: Section, args):
    lay = layout(cfg)
    failures: list[dict] = []
    clip_hours = 0.0
    while True:
        failed = {f["id"] for f in failures}
        pending, n_done = select(cfg, args, ready=lambda v, failed=failed: lay.diar_json(v["id"]).exists() and v["id"] not in failed,
                                 done=lambda v: lay.chunks_meta(v["id"]).exists())
        if pending:
            logger.info(f"{n_done} done, {len(pending)} to segment ({total_sec(pending) / 3600:.1f} h, up to "
                        f"{cfg.segmentation.workers} workers, {cfg.segmentation.memory_gb} GB RAM budget)")
            clip_hours += run_batch(cfg, pending, failures)
            write_failures(cfg, "segment", failures)
        if not getattr(args, "watch", False):
            break
        args.force, args.limit = False, None  # later rounds: only new videos
        diar_failed = {f["id"] for f in read_jsonl(lay.failures("diarize"))}
        failed = {f["id"] for f in failures}
        left = [v for v in videos(cfg) if not lay.chunks_meta(v["id"]).exists()
                and v["id"] not in diar_failed | failed]
        if not left:
            break
        if not pending:
            logger.info(f"--watch: waiting for diarize ({len(left)} videos left)")
            time.sleep(60)
    if not pending and not clip_hours and not failures:
        logger.info("nothing to segment (run diarize first, or everything is done)")
    logger.info(f"this run: {clip_hours:.2f} h of clips, {len(failures)} videos failed"
                + (" (meta/failures/segment.jsonl)" if failures else ""))


if __name__ == "__main__":
    step_main(main, lambda p: p.add_argument("--watch", action="store_true",
                                             help="keep segmenting newly diarized videos until all are done"))
