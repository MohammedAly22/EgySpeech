"""Step 3 — separate: remove music and sound effects (vocal stem) with audio-separator.

Writes audio/vocals/<vid>.flac (mono, 24 kHz). The segmenter later compares the
vocal stem with the original per clip and keeps the original wherever there was no
music, so separation artifacts only appear where separation was really needed.
With separation.enabled=false the "vocal stem" is simply the original audio.
"""

import logging
import tempfile
from pathlib import Path

from egyspeech.config import Section
from egyspeech.io import decode_audio, read_audio, resample, write_audio
from egyspeech.steps import layout, step_main, video_ids

logger = logging.getLogger("separate")


def main(cfg: Section, args):
    lay = layout(cfg)
    fmt, sr = cfg.download.format, cfg.download.sample_rate
    vids = [v for v in video_ids(cfg) if lay.raw_audio(v, fmt).exists()]
    pending = [v for v in vids if args.force or not lay.vocals(v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"{len(vids) - len(pending)} done, {len(pending)} to process")
    if not pending:
        return
    lay.vocals("x").parent.mkdir(parents=True, exist_ok=True)

    if not cfg.separation.enabled:
        for vid in pending:
            write_audio(lay.vocals(vid), decode_audio(lay.raw_audio(vid, fmt), sr), sr)
        return

    from audio_separator.separator import Separator

    models_dir = Path(cfg.work_dir) / "models" / "audio-separator"
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sep_", dir=str(cfg.work_dir)) as tmp:
        sep = Separator(
            output_dir=tmp,
            output_format="FLAC",
            output_single_stem="Vocals",
            model_file_dir=str(models_dir),
            log_level=logging.WARNING,
        )
        sep.load_model(model_filename=cfg.separation.model)
        for i, vid in enumerate(pending, 1):
            outputs = [Path(tmp) / Path(o).name for o in sep.separate(str(lay.raw_audio(vid, fmt)))]
            vocal = [o for o in outputs if "vocal" in o.name.lower()] or outputs
            wav, wsr = read_audio(vocal[0])
            write_audio(lay.vocals(vid), resample(wav, wsr, sr), sr)
            for o in outputs:
                o.unlink(missing_ok=True)
            logger.info(f"[{i}/{len(pending)}] {vid}: vocal stem written")


if __name__ == "__main__":
    step_main(main)
