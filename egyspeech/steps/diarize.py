"""Step 4 — diarize (env: egyspeech-nemo): NVIDIA Streaming Sortformer on the vocal stem.

Saves per video the frame-level speaker-activity probabilities (80 ms frames, up to
4 speakers) and the predicted segments. The segmenter uses the probabilities to keep
only frames where exactly one speaker talks.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from egyspeech.config import Section
from egyspeech.io import atomic_path, read_audio, resample
from egyspeech.steps import layout, step_main, video_ids

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
    vids = [v for v in video_ids(cfg) if lay.vocals(v).exists()]
    pending = [v for v in vids if args.force or not lay.diar_json(v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"{len(vids) - len(pending)} done, {len(pending)} to diarize")
    if not pending:
        return
    model = load_model(cfg)
    for i, vid in enumerate(pending, 1):
        wav, sr = read_audio(lay.vocals(vid))
        wav16 = resample(wav, sr, 16000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{vid}.wav"
            sf.write(str(path), wav16, 16000, subtype="PCM_16")
            with torch.inference_mode():
                segments, probs = model.diarize(audio=[str(path)], batch_size=1, include_tensor_outputs=True)
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
        tmp_json.replace(lay.diar_json(vid))
        logger.info(f"[{i}/{len(pending)}] {vid}: {len(speakers)} speakers, {len(segs)} segments")


if __name__ == "__main__":
    step_main(main)
