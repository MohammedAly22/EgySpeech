"""Step 7 — speaker_check (env: egyspeech-nemo): TitaNet speaker embeddings per clip.

For every clip:
  * embeddings of sliding windows (3 s, hop 1.5 s) -> `window_similarity` = the lowest
    cosine similarity between any window and the clip's mean voice. A second voice
    anywhere in the clip (a guest's "aha", a laugh from someone else) pulls it down.
  * the clip embedding (mean of windows) -> global speaker clustering (cluster step).
Saved as meta/speakers/<vid>.npz (ids, embeddings float16, window_similarity).
"""

import logging

import numpy as np

from egyspeech.config import Section
from egyspeech.io import atomic_path, read_audio, read_jsonl, resample
from egyspeech.steps import layout, step_main, video_ids

logger = logging.getLogger("speaker_check")


def windows(wav: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> list[np.ndarray]:
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    if len(wav) <= win:
        return [wav]
    starts = list(range(0, len(wav) - win + 1, hop))
    if starts[-1] + win < len(wav):
        starts.append(len(wav) - win)  # cover the tail too
    return [wav[s : s + win] for s in starts]


def main(cfg: Section, args):
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    lay = layout(cfg)
    vids = [v for v in video_ids(cfg) if lay.chunks_meta(v).exists()]
    pending = [v for v in vids if args.force or not lay.speaker_emb(v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"{len(vids) - len(pending)} done, {len(pending)} to embed")
    if not pending:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncDecSpeakerLabelModel.from_pretrained(cfg.speakers.embedding_model).to(device).eval()
    q = cfg.quality
    batch_size = 64
    for i, vid in enumerate(pending, 1):
        chunks = read_jsonl(lay.chunks_meta(vid))
        all_windows, owner = [], []
        for k, c in enumerate(chunks):
            wav, sr = read_audio(c["path"])
            for w in windows(resample(wav, sr, 16000), 16000, q.speaker_window_sec, q.speaker_hop_sec):
                all_windows.append(w)
                owner.append(k)
        embs = np.zeros((len(all_windows), 192), dtype=np.float32)
        order = np.argsort([len(w) for w in all_windows])
        for b0 in range(0, len(order), batch_size):
            idx = order[b0 : b0 + batch_size]
            lens = [len(all_windows[j]) for j in idx]
            x = torch.zeros(len(idx), max(lens))
            for r, j in enumerate(idx):
                x[r, : lens[r]] = torch.from_numpy(all_windows[j])
            with torch.inference_mode():
                _, e = model.forward(input_signal=x.to(device), input_signal_length=torch.tensor(lens, device=device))
            embs[idx] = e.float().cpu().numpy()
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        owner_arr = np.array(owner)
        clip_embs = np.zeros((len(chunks), embs.shape[1]), dtype=np.float32)
        sims = np.zeros(len(chunks), dtype=np.float32)
        for k in range(len(chunks)):
            w = embs[owner_arr == k]
            mean = w.mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-9
            clip_embs[k] = mean
            sims[k] = float((w @ mean).min())
        tmp = atomic_path(lay.speaker_emb(vid)).with_suffix(".npz")
        np.savez(tmp, ids=np.array([c["id"] for c in chunks]), embeddings=clip_embs.astype(np.float16),
                 window_similarity=sims, local_speaker=np.array([c["local_speaker"] for c in chunks]))
        tmp.replace(lay.speaker_emb(vid))
        logger.info(f"[{i}/{len(pending)}] {vid}: {len(chunks)} clips, window similarity "
                    f"median {np.median(sims) if len(sims) else 0:.2f}, min {sims.min() if len(sims) else 0:.2f}")


if __name__ == "__main__":
    step_main(main)
