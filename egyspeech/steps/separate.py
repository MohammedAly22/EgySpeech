"""Step 3 — separate: remove music and sound effects (vocal stem) with audio-separator.

Writes audio/vocals/<vid>.flac (mono, 24 kHz) and meta/separation/<vid>.json.

separation.mode
  never   (default) no separation: clean podcasts are used as they are (this step does nothing)
  auto    probe a few short windows spread over the episode; separate the
          whole episode only if one of them contains music / effects. Talk podcasts
          without a music bed skip the (expensive) separation entirely.
  always  separate every episode

The segmenter later measures the music level per clip and keeps the original audio
wherever there is (almost) no music, so separation artifacts only appear where
separation was needed.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, decode_audio, read_audio, resample, write_audio
from egyspeech.progress import StepBar
from egyspeech.steps import layout, select, step_main, total_sec, write_failures

logger = logging.getLogger("separate")


def music_db(original: np.ndarray, vocals: np.ndarray) -> float:
    """Level of what the separator removed, relative to the speech (dB).

    Gain-invariant: the vocal stem is least-squares matched to the original first,
    so output normalization of the separator does not bias the estimate.
    """
    n = min(len(original), len(vocals))
    o, v = original[:n].astype(np.float64), vocals[:n].astype(np.float64)
    vv = float(v @ v)
    if vv < 1e-9:
        return 0.0 if float(o @ o) > 1e-9 else -100.0
    g = float(o @ v) / vv
    residual = o - g * v
    return float(10 * np.log10((residual @ residual + 1e-10) / (g * g * vv + 1e-10)))


class Stemmer:
    def __init__(self, cfg: Section, tmp: str):
        from audio_separator.separator import Separator

        models_dir = Path(cfg.work_dir) / "models" / "audio-separator"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tmp)
        self.sep = Separator(
            output_dir=tmp,
            output_format="FLAC",
            output_single_stem="Vocals",
            model_file_dir=str(models_dir),
            normalization_threshold=1.0,
            use_autocast=bool(cfg.separation.use_autocast),
            log_level=logging.WARNING,
        )
        self.sep.load_model(model_filename=cfg.separation.model)

    def vocals(self, path: Path, sr: int) -> np.ndarray:
        outputs = [self.tmp / Path(o).name for o in self.sep.separate(str(path))]
        vocal = [o for o in outputs if "vocal" in o.name.lower()] or outputs
        wav, wsr = read_audio(vocal[0])
        for o in outputs:
            o.unlink(missing_ok=True)
        return resample(wav, wsr, sr)


def probe(raw: np.ndarray, sr: int, stemmer: Stemmer, cfg: Section) -> list[float]:
    """Music level of probe windows spread over the episode (skipping intro / outro)."""
    s = cfg.separation
    win = int(s.probe_window_sec * sr)
    if len(raw) <= win:
        starts = [0]
    else:
        lo, hi = int(0.05 * len(raw)), int(0.95 * len(raw)) - win
        starts = np.linspace(lo, max(lo, hi), int(s.probe_windows)).astype(int).tolist()
    windows = [raw[a : a + win] for a in starts]
    gap = np.zeros(int(0.5 * sr), dtype=np.float32)
    joined = np.concatenate([np.concatenate([w, gap]) for w in windows])
    path = stemmer.tmp / "probe.wav"
    write_audio(path, joined, sr)
    voc = stemmer.vocals(path, sr)
    path.unlink(missing_ok=True)
    levels, pos = [], 0
    for w in windows:
        levels.append(round(music_db(w, voc[pos : pos + len(w)]), 2))
        pos += len(w) + len(gap)
    return levels


def main(cfg: Section, args):
    lay = layout(cfg)
    sr = cfg.download.sample_rate
    mode = cfg.separation.mode
    if mode not in ("auto", "always", "never"):
        raise SystemExit(f"separation.mode must be auto | always | never, got {mode!r}")
    if mode == "never":
        logger.info("separation.mode is never: the speech steps read the episodes as they are")
        return
    pending, n_done = select(cfg, args, ready=lambda v: True, done=lambda v: lay.vocals(v["id"]).exists())
    logger.info(f"{n_done} done, {len(pending)} to process (mode={mode})")
    if not pending:
        return
    lay.vocals("x").parent.mkdir(parents=True, exist_ok=True)
    info_dir = lay.meta / "separation"
    info_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    with tempfile.TemporaryDirectory(prefix="sep_", dir=str(cfg.work_dir)) as tmp,             StepBar("separate", len(pending), audio_sec=total_sec(pending)) as bar:
        stemmer = Stemmer(cfg, tmp)
        n_sep = 0
        for v in pending:
            vid = v["id"]
            bar.status(vid)
            try:
                raw = decode_audio(v["path"], sr, v.get("duration"))
                info: dict = {"mode": mode}
                separate = mode == "always"
                if mode == "auto":
                    levels = probe(raw, sr, stemmer, cfg)
                    info["probe_music_db"] = levels
                    separate = max(levels) > cfg.separation.use_original_below_music_db
                if separate:
                    src = Path(tmp) / f"{vid}.flac"
                    write_audio(src, raw, sr)
                    vocals = stemmer.vocals(src, sr)
                    src.unlink(missing_ok=True)
                    n_sep += 1
                else:
                    vocals = raw
                write_audio(lay.vocals(vid), vocals[: len(raw)], sr)
                info["separated"] = separate
                tmp_json = atomic_path(info_dir / f"{vid}.json")
                tmp_json.write_text(json.dumps(info), encoding="utf-8")
                tmp_json.replace(info_dir / f"{vid}.json")
                what = "separated" if separate else "no music found, original kept"
                extra = f" | probe music dB {info.get('probe_music_db')}" if mode == "auto" else ""
                logger.info(f"{vid}: {what}{extra}")
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                failures.append({"id": vid, "path": v["path"], "error": f"{type(exc).__name__}: {exc}"[:500]})
                logger.warning(f"{vid}: {type(exc).__name__}: {exc}")
            bar.advance(audio_sec=v["duration"])
    write_failures(cfg, "separate", failures)
    logger.info(f"{n_sep}/{len(pending)} episodes needed separation, {len(failures)} failed")


if __name__ == "__main__":
    step_main(main)
