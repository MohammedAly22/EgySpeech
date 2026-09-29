"""Step 5 — segment: single-speaker 5-30 s clips cut at pauses, loudness normalized.

Per video: Silero VAD on the vocal stem, Sortformer probabilities, energy -> the
pause-aware planner (egyspeech/segmenter.py) -> clips. For each clip the music level
(original minus vocal stem) decides whether the original audio (untouched, best
fidelity) or the vocal stem (music / effects removed) is exported.
"""

import logging

import numpy as np

from egyspeech.config import Section
from egyspeech.io import decode_audio, read_audio, read_json, resample, write_audio, write_jsonl
from egyspeech.segmenter import HOP, SegParams, plan_clips, to_grid
from egyspeech.steps import layout, step_main, video_ids
from egyspeech.steps.separate import music_db

logger = logging.getLogger("segment")

_VAD = None


def vad_probs(wav16: np.ndarray) -> tuple[np.ndarray, float]:
    """Silero VAD speech probability per 512-sample (32 ms) window at 16 kHz."""
    global _VAD
    import torch
    from silero_vad import load_silero_vad

    if _VAD is None:
        _VAD = load_silero_vad()
    model = _VAD
    model.reset_states()
    win = 512
    n = len(wav16) // win
    x = torch.from_numpy(wav16[: n * win].copy())
    with torch.inference_mode():
        # stateful, batched over the whole recording (one call instead of one per 32 ms window)
        probs = model.audio_forward(x, 16000).squeeze(0).float().numpy()
    return probs[:n].astype(np.float32), win / 16000


def energy_db(wav: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * HOP)
    n = len(wav) // hop
    frames = wav[: n * hop].reshape(n, hop).astype(np.float64)
    db = 10 * np.log10(np.mean(frames * frames, axis=1) + 1e-10)
    return np.convolve(db, np.ones(3) / 3, mode="same").astype(np.float32)


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

    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(x.astype(np.float64))
    if np.isfinite(loud):
        x = x * (10 ** ((lufs - loud) / 20))
    ceiling = 10 ** (peak_dbfs / 20)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > ceiling:
        x = x * (ceiling / peak)
    return x.astype(np.float32), float(loud)


def process_video(vid: str, cfg: Section) -> list[dict]:
    lay = layout(cfg)
    sr = cfg.download.sample_rate
    vocals, vsr = read_audio(lay.vocals(vid))
    vocals = resample(vocals, vsr, sr)
    raw = decode_audio(lay.raw_audio(vid, cfg.download.format), sr)
    n_samples = min(len(vocals), len(raw))
    vocals, raw = vocals[:n_samples], raw[:n_samples]

    diar = np.load(lay.diar_probs(vid)).astype(np.float32)
    diar_info = read_json(lay.diar_json(vid))
    vad, vad_sec = vad_probs(resample(vocals, sr, 16000))
    db = energy_db(vocals, sr)
    n = min(len(db), int(len(vad) * vad_sec / HOP), int(diar.shape[0] * diar_info["frame_sec"] / HOP))
    grid_diar = to_grid(diar, diar_info["frame_sec"], n)
    grid_vad = to_grid(vad, vad_sec, n)
    clips = plan_clips(grid_diar, grid_vad, db[:n], seg_params(cfg))

    seg = cfg.segmentation
    out_dir = lay.chunk_dir(vid)
    if out_dir.exists():  # re-segmenting: never leave clips from an older plan behind
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counters: dict[int, int] = {}
    for clip in clips:
        a, b = int(clip.start * sr), int(clip.end * sr)
        v, r = vocals[a:b], raw[a:b]
        mdb = music_db(r, v)  # gain-invariant level of what separation removed
        use_raw = cfg.separation.mode == "never" or mdb <= cfg.separation.use_original_below_music_db
        x = r if use_raw else v
        clip_ratio = float(np.mean(np.abs(x) >= 0.999))
        x, loud = loudness_normalize(x, sr, seg.loudness_lufs, seg.peak_dbfs)
        k = counters.get(clip.speaker, 0)
        counters[clip.speaker] = k + 1
        cid = f"{vid}_s{clip.speaker}_{k:04d}"
        path = out_dir / f"{cid}.{seg.audio_format}"
        write_audio(path, x, sr)
        rows.append({
            "id": cid, "video_id": vid, "local_speaker": clip.speaker, "path": str(path),
            "start": clip.start, "end": clip.end, "duration": round(clip.duration, 3),
            "source": "original" if use_raw else "vocals", "music_db": round(float(mdb), 2),
            "input_lufs": round(loud, 2) if np.isfinite(loud) else None, "clip_ratio": clip_ratio,
            "start_cut": clip.start_cut, "end_cut": clip.end_cut,
            "start_pause": clip.start_pause, "end_pause": clip.end_pause,
            "speech_ratio": clip.speech_ratio, "max_internal_silence": clip.max_internal_silence,
            "weak_cut": "weak" in (clip.start_cut, clip.end_cut),
        })
    return rows


def _worker(vid: str, cfg_path: str) -> tuple[str, int, float, int]:
    import torch

    from egyspeech.config import load_config

    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    rows = process_video(vid, cfg)
    lay = layout(cfg)
    write_jsonl(lay.chunks_meta(vid), rows)  # written last: marks the video as done
    if cfg.storage.delete_vocals_after_segment:
        lay.vocals(vid).unlink(missing_ok=True)
    return vid, len(rows), sum(r["duration"] for r in rows) / 3600, sum(r["source"] == "original" for r in rows)


def main(cfg: Section, args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    lay = layout(cfg)
    vids = [v for v in video_ids(cfg) if lay.diar_json(v).exists()]
    pending = [v for v in vids if args.force or not lay.chunks_meta(v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    workers = max(1, int(cfg.segmentation.workers))
    logger.info(f"{len(vids) - len(pending)} done, {len(pending)} to segment ({workers} workers)")
    total = 0.0
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_worker, v, cfg["_path"]) for v in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            vid, n, hours, orig = fut.result()
            total += hours
            logger.info(f"[{i}/{len(pending)}] {vid}: {n} clips, {hours:.2f} h ({orig} original / {n - orig} vocal-stem)")
    logger.info(f"segmented {total:.2f} h in this run")


if __name__ == "__main__":
    step_main(main)
