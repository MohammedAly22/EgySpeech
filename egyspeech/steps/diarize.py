"""Step 4 — diarize (env: egyspeech-nemo): NVIDIA Streaming Sortformer on every episode.

Saves per video the frame-level speaker-activity probabilities (80 ms frames, up to
4 speakers) and the predicted segments. The segmenter uses the probabilities to keep
only frames where exactly one speaker talks.

The next episode is decoded (ffmpeg -> 16 kHz) in a background thread while the GPU
works on the current one. A video that fails is recorded in meta/failures/diarize.jsonl
and retried on the next run; it does not stop the step.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, decode_audio, read_jsonl, write_jsonl
from egyspeech.parallel import prefetch
from egyspeech.progress import StepBar
from egyspeech.steps import layout, select, speech_audio, step_main, total_sec

os.environ.setdefault("TQDM_DISABLE", "1")  # NeMo's own bars would break the step's progress bar
logger = logging.getLogger("diarize")
FRAME_SEC = 0.08


def load_model(cfg: Section):
    import nemo.collections.asr.models.sortformer_diar_models as sortformer
    from nemo.collections.asr.models import SortformerEncLabelModel

    sortformer.tqdm = lambda it, **kw: it  # NeMo's per-window bar would break the step's progress bar
    d = cfg.diarization
    model = SortformerEncLabelModel.from_pretrained(d.model)
    model.eval()
    mods = model.sortformer_modules
    mods.chunk_len = d.chunk_len
    mods.chunk_right_context = d.chunk_right_context
    mods.fifo_len = d.fifo_len
    mods.spkcache_update_period = d.spkcache_update_period
    mods.spkcache_len = d.spkcache_len
    if hasattr(mods, "_check_streaming_parameters"):
        mods._check_streaming_parameters()
    return model


def parse_segments(raw) -> list[list]:
    out = []
    for seg in raw:
        if isinstance(seg, str):
            start, end, spk = seg.split()
        else:
            start, end, spk = seg
        out.append([round(float(start), 3), round(float(end), 3), str(spk)])
    return out


def segments_from_probs(p: np.ndarray, threshold: float = 0.5, min_sec: float = 0.2) -> list[list]:
    """[start, end, "speaker_k"] runs where a speaker's probability is above threshold."""
    out = []
    for k in range(p.shape[1]):
        on = np.concatenate([[False], p[:, k] > threshold, [False]])
        d = np.diff(on.astype(np.int8))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            if (b - a) * FRAME_SEC >= min_sec:
                out.append([round(a * FRAME_SEC, 3), round(b * FRAME_SEC, 3), f"speaker_{k}"])
    return sorted(out)


def best_permutation(a: np.ndarray, b: np.ndarray) -> list[int]:
    """Column order of b that best matches a's speakers on their overlap (b[:, perm] ~ a)."""
    from itertools import permutations

    sim = a.T @ b  # [S_a, S_b] co-activity
    return list(max(permutations(range(b.shape[1])), key=lambda perm: sum(sim[i, j] for i, j in enumerate(perm))))


def stitch(windows: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Join per-window probabilities [(start frame, probs)] into one track.

    Speakers are renamed window by window to match the previous window on the overlap;
    each overlap is split in the middle.
    """
    _, full = windows[0]
    for start, p in windows[1:]:
        ov = full.shape[0] - start
        if ov > 0:
            p = p[:, best_permutation(full[start:], p[:ov])]
            half = ov // 2
            full = np.concatenate([full[: start + half], p[half:]])
        else:
            full = np.concatenate([full, p])
    return full


def diarize_audio(model, wav16: np.ndarray, window_sec: float, overlap_sec: float) -> tuple[np.ndarray, list[list]]:
    """(probabilities [frames, speakers], segments) of one recording, in windows if it is long.

    Long recordings are diarized in windows (bounded GPU memory) that overlap by
    overlap_sec; the windows are stitched by matching speakers on the overlaps.
    """
    import torch

    def run(x: np.ndarray):
        with torch.inference_mode():
            segments, probs = model.diarize(audio=[x], sample_rate=16000, batch_size=1, include_tensor_outputs=True,
                                            verbose=False)
        p = probs[0]
        p = p.detach().float().cpu().numpy() if hasattr(p, "detach") else np.asarray(p, dtype=np.float32)
        return (p[0] if p.ndim == 3 else p), segments[0]

    hop = int(FRAME_SEC * 16000)  # windows start on frame boundaries
    win = max(hop, int(window_sec * 16000) // hop * hop)
    if len(wav16) <= win:
        p, segs = run(wav16)
        return p, parse_segments(segs)
    step = max(hop, (win - int(overlap_sec * 16000)) // hop * hop)
    windows, start = [], 0
    while True:
        p, _ = run(wav16[start : start + win])
        windows.append((start // hop, p))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if start + win >= len(wav16):
            break
        start += step
    p = stitch(windows)
    return p, segments_from_probs(p)


def _cuda_broken(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return ("CUDA" in text or "INTERNAL ASSERT" in text or "AcceleratorError" in text) and "out of memory" not in text


RESTART = 75  # exit code: the runner restarts the step in a fresh process (clean GPU state)


def main(cfg: Section, args):
    import torch

    lay = layout(cfg)
    pending, n_done = select(cfg, args, ready=lambda v: speech_audio(cfg, v).exists(),
                             done=lambda v: lay.diar_json(v["id"]).exists())
    logger.info(f"{n_done} done, {len(pending)} to diarize ({total_sec(pending) / 3600:.1f} h)")
    if not pending:
        return
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)
    except ImportError:
        pass
    model = load_model(cfg)

    def load(v: dict):
        try:
            return decode_audio(speech_audio(cfg, v), 16000, v.get("duration"))
        except Exception as exc:  # noqa: BLE001 - reported when its turn comes
            return exc

    d = cfg.diarization
    old = {r["id"]: r for r in read_jsonl(lay.failures("diarize"))}
    failures = dict(old)
    skip = {k for k, r in old.items() if r.get("gpu_crashes", 0) >= 2}
    if skip:
        logger.warning(f"skipping {len(skip)} videos that broke the GPU state twice (meta/failures/diarize.jsonl; "
                       "delete that file to retry them)")
        pending = [v for v in pending if v["id"] not in skip]

    def save_failures():
        write_jsonl(lay.failures("diarize"), list(failures.values()))

    ok = 0
    with ThreadPoolExecutor(1) as pool, StepBar("diarize", len(pending), audio_sec=total_sec(pending)) as bar:
        for v, wav16 in prefetch(pool, load, pending, depth=2):
            vid = v["id"]
            bar.status(vid)
            window = float(d.window_min) * 60
            while True:
                try:
                    if isinstance(wav16, Exception):
                        raise wav16
                    p, segs = diarize_audio(model, wav16, window, float(d.window_overlap_sec))
                    tmp_npy = atomic_path(lay.diar_probs(vid)).with_suffix(".npy")
                    np.save(tmp_npy, p.astype(np.float16))
                    tmp_npy.replace(lay.diar_probs(vid))
                    speakers = sorted({s[2] for s in segs})
                    info = {"frame_sec": FRAME_SEC, "n_frames": int(p.shape[0]), "n_speakers": len(speakers),
                            "duration": len(wav16) / 16000, "window_sec": window, "segments": segs}
                    tmp_json = atomic_path(lay.diar_json(vid))
                    tmp_json.write_text(json.dumps(info), encoding="utf-8")
                    tmp_json.replace(lay.diar_json(vid))  # written last: marks the video as done
                    failures.pop(vid, None)
                    ok += 1
                    logger.info(f"{vid}: {len(speakers)} speakers, {v['duration'] / 3600:.2f} h")
                    break
                except Exception as exc:  # noqa: BLE001 - classify, then keep going (or restart cleanly)
                    err = f"{type(exc).__name__}: {exc}"[:500]
                    if "out of memory" in err and window > 300:
                        torch.cuda.empty_cache()
                        window /= 2
                        logger.warning(f"{vid}: GPU out of memory, retrying with {window / 60:.0f} min windows")
                        continue
                    failures[vid] = {"id": vid, "path": v["path"], "error": err,
                                     "gpu_crashes": old.get(vid, {}).get("gpu_crashes", 0) + int(_cuda_broken(exc))}
                    save_failures()
                    if _cuda_broken(exc):
                        logger.error(f"{vid}: {err[:200]} -- the GPU state is broken: restarting the step in a new "
                                     "process (finished videos are kept)")
                        bar.close()
                        sys.exit(RESTART)
                    logger.warning(f"{vid}: {err[:200]}")
                    break
            del wav16
            bar.advance(audio_sec=v["duration"])
    save_failures()
    logger.info(f"diarized {ok} videos, {len(failures)} failing"
                + (" (meta/failures/diarize.jsonl; re-run to retry)" if failures else ""))


if __name__ == "__main__":
    step_main(main)
