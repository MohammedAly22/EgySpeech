"""Step 7 — speaker_check (env: egyspeech-nemo): TitaNet speaker embeddings per clip.

For every clip:
  * embeddings of sliding windows (3 s, hop 1.5 s) -> `window_similarity` = the lowest
    cosine similarity between any window and the clip's mean voice. A second voice
    anywhere in the clip (a guest's "aha", a laugh from someone else) pulls it down.
  * the clip embedding (mean of windows) -> global speaker clustering (cluster step).
Saved as meta/speakers/<vid>.npz (ids, embeddings float16, window_similarity, local_speaker).

Clips are processed in groups; the next group is read from disk in a background
thread while the GPU embeds the current one.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, read_audio, read_jsonl, resample
from egyspeech.parallel import prefetch
from egyspeech.progress import StepBar
from egyspeech.steps import layout, select, step_main, write_failures

os.environ.setdefault("TQDM_DISABLE", "1")  # NeMo's own bars would break the step's progress bar
logger = logging.getLogger("speaker_check")
GROUP = 64  # clips per group
BATCH = 64  # windows per GPU batch


def windows(wav: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> list[np.ndarray]:
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    if len(wav) <= win:
        return [wav]
    starts = list(range(0, len(wav) - win + 1, hop))
    if starts[-1] + win < len(wav):
        starts.append(len(wav) - win)  # cover the tail too
    return [wav[s : s + win] for s in starts]


class Embedder:
    def __init__(self, cfg: Section):
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncDecSpeakerLabelModel.from_pretrained(cfg.speakers.embedding_model).to(self.device).eval()

    def embed(self, wins: list[np.ndarray]) -> np.ndarray:
        import torch

        embs = np.zeros((len(wins), 192), dtype=np.float32)
        order = np.argsort([len(w) for w in wins])
        for b0 in range(0, len(order), BATCH):
            idx = order[b0 : b0 + BATCH]
            lens = [len(wins[j]) for j in idx]
            x = torch.zeros(len(idx), max(lens))
            for r, j in enumerate(idx):
                x[r, : lens[r]] = torch.from_numpy(wins[j])
            with torch.inference_mode():
                _, e = self.model.forward(input_signal=x.to(self.device),
                                          input_signal_length=torch.tensor(lens, device=self.device))
            embs[idx] = e.float().cpu().numpy()
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)


def main(cfg: Section, args):
    lay = layout(cfg)
    pending, n_done = select(cfg, args, ready=lambda v: lay.chunks_meta(v["id"]).exists(),
                             done=lambda v: lay.speaker_emb(v["id"]).exists())
    chunks = {v["id"]: read_jsonl(lay.chunks_meta(v["id"])) for v in pending}
    logger.info(f"{n_done} done, {len(pending)} videos / {sum(len(c) for c in chunks.values())} clips to embed")
    if not pending:
        return
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)
    except ImportError:
        pass
    model = Embedder(cfg)
    q = cfg.quality

    def load_group(group: list[dict]) -> list[list[np.ndarray]]:
        out = []
        for c in group:
            wav, sr = read_audio(c["path"])
            out.append(windows(resample(wav, sr, 16000), 16000, q.speaker_window_sec, q.speaker_hop_sec))
        return out

    failures = []
    clip_sec = sum(c["duration"] for cs in chunks.values() for c in cs)
    with ThreadPoolExecutor(2) as loader, StepBar("speaker_check", len(pending), audio_sec=clip_sec) as bar:
        for v in pending:
            vid = v["id"]
            cs = chunks[vid]
            bar.status(vid)
            try:
                clip_embs = np.zeros((len(cs), 192), dtype=np.float32)
                sims = np.zeros(len(cs), dtype=np.float32)
                groups = [cs[i : i + GROUP] for i in range(0, len(cs), GROUP)]
                k0 = 0
                for group, wins in prefetch(loader, load_group, groups, depth=2):
                    flat = [w for ws in wins for w in ws]
                    owner = np.repeat(np.arange(len(wins)), [len(ws) for ws in wins])
                    embs = model.embed(flat)
                    for k in range(len(group)):
                        w = embs[owner == k]
                        mean = w.mean(axis=0)
                        mean /= np.linalg.norm(mean) + 1e-9
                        clip_embs[k0 + k] = mean
                        sims[k0 + k] = float((w @ mean).min())
                    k0 += len(group)
                tmp = atomic_path(lay.speaker_emb(vid)).with_suffix(".npz")
                np.savez(tmp, ids=np.array([c["id"] for c in cs]), embeddings=clip_embs.astype(np.float16),
                         window_similarity=sims, local_speaker=np.array([c["local_speaker"] for c in cs]))
                tmp.replace(lay.speaker_emb(vid))
                if len(cs):
                    logger.info(f"{vid}: {len(cs)} clips, window similarity median {np.median(sims):.2f}, "
                                f"min {sims.min():.2f}")
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                failures.append({"id": vid, "error": f"{type(exc).__name__}: {exc}"[:500]})
                logger.warning(f"{vid}: {type(exc).__name__}: {str(exc)[:200]}")
            bar.advance(audio_sec=sum(c["duration"] for c in cs))
    write_failures(cfg, "speaker_check", failures)
    logger.info(f"embedded {len(pending) - len(failures)} videos, {len(failures)} failed")


if __name__ == "__main__":
    step_main(main)
