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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, decode_audio
from egyspeech.parallel import prefetch
from egyspeech.progress import StepBar
from egyspeech.steps import layout, select, speech_audio, step_main, total_sec, write_failures

os.environ.setdefault("TQDM_DISABLE", "1")  # NeMo's own bars would break the step's progress bar
logger = logging.getLogger("diarize")
FRAME_SEC = 0.08


def load_model(cfg: Section):
    from nemo.collections.asr.models import SortformerEncLabelModel

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

    failures = []
    with ThreadPoolExecutor(1) as pool, StepBar("diarize", len(pending), audio_sec=total_sec(pending)) as bar:
        for v, wav16 in prefetch(pool, load, pending, depth=2):
            vid = v["id"]
            bar.status(vid)
            try:
                if isinstance(wav16, Exception):
                    raise wav16
                with torch.inference_mode():
                    segments, probs = model.diarize(audio=[wav16], sample_rate=16000, batch_size=1,
                                                    include_tensor_outputs=True, verbose=False)
                p = probs[0]
                p = p.detach().float().cpu().numpy() if hasattr(p, "detach") else np.asarray(p, dtype=np.float32)
                if p.ndim == 3:
                    p = p[0]
                tmp_npy = atomic_path(lay.diar_probs(vid)).with_suffix(".npy")
                np.save(tmp_npy, p.astype(np.float16))
                tmp_npy.replace(lay.diar_probs(vid))
                segs = parse_segments(segments[0])
                speakers = sorted({s[2] for s in segs})
                info = {"frame_sec": FRAME_SEC, "n_frames": int(p.shape[0]), "n_speakers": len(speakers),
                        "duration": len(wav16) / 16000, "segments": segs}
                tmp_json = atomic_path(lay.diar_json(vid))
                tmp_json.write_text(json.dumps(info), encoding="utf-8")
                tmp_json.replace(lay.diar_json(vid))  # written last: marks the video as done
                logger.info(f"{vid}: {len(speakers)} speakers, {len(segs)} segments, {v['duration'] / 3600:.2f} h")
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                failures.append({"id": vid, "path": v["path"], "error": f"{type(exc).__name__}: {exc}"[:500]})
                logger.warning(f"{vid}: {type(exc).__name__}: {str(exc)[:200]}")
            del wav16
            bar.advance(audio_sec=v["duration"])
    write_failures(cfg, "diarize", failures)
    logger.info(f"diarized {len(pending) - len(failures)} videos, {len(failures)} failed"
                + (" (meta/failures/diarize.jsonl; re-run to retry)" if failures else ""))


if __name__ == "__main__":
    step_main(main)
