"""Step 6 — quality: DNSMOS (P.835 SIG/BAK/OVRL + P.808) and UTMOS per clip.

Runs on every clip (cheap); the filter step applies thresholds later, so they can be
tuned without recomputing anything.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_audio, read_jsonl, resample, write_jsonl
from egyspeech.steps import layout, step_main, video_ids

logger = logging.getLogger("quality")


class Scorers:
    def __init__(self, cfg: Section):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        self.utmos.to(self.device).eval()
        self.batch = cfg.quality.batch_size
        self.workers = cfg.quality.workers

    def dnsmos(self, wav16: np.ndarray) -> list[float]:
        import torch
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

        # [p808_mos, mos_sig, mos_bak, mos_ovr]
        out = deep_noise_suppression_mean_opinion_score(torch.from_numpy(wav16), 16000, personalized=False)
        return [float(v) for v in out.tolist()]

    def utmos_batch(self, wavs16: list[np.ndarray]) -> list[float]:
        import torch

        scores = []
        order = np.argsort([len(w) for w in wavs16])
        result = [0.0] * len(wavs16)
        for b0 in range(0, len(order), self.batch):
            idx = order[b0 : b0 + self.batch]
            # score each clip at its own length: pad-free by grouping similar lengths
            length = min(len(wavs16[i]) for i in idx)
            x = torch.stack([torch.from_numpy(wavs16[i][:length]) for i in idx]).to(self.device)
            with torch.inference_mode():
                s = self.utmos(x, 16000).float().cpu().numpy()
            for i, v in zip(idx, s, strict=True):
                result[i] = float(v)
        return result or scores


def main(cfg: Section, args):
    lay = layout(cfg)
    vids = [v for v in video_ids(cfg) if lay.chunks_meta(v).exists()]
    pending = [v for v in vids if args.force or not lay.quality(v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"{len(vids) - len(pending)} done, {len(pending)} to score")
    if not pending:
        return
    scorers = Scorers(cfg)
    pool = ThreadPoolExecutor(max_workers=scorers.workers)
    for i, vid in enumerate(pending, 1):
        chunks = read_jsonl(lay.chunks_meta(vid))
        wavs = []
        for c in chunks:
            wav, sr = read_audio(c["path"])
            wavs.append(resample(wav, sr, 16000))
        dns = list(pool.map(scorers.dnsmos, wavs))
        ut = scorers.utmos_batch(wavs) if wavs else []
        rows = [
            {"id": c["id"], "dnsmos_p808": round(d[0], 3), "dnsmos_sig": round(d[1], 3),
             "dnsmos_bak": round(d[2], 3), "dnsmos_ovrl": round(d[3], 3), "utmos": round(u, 3)}
            for c, d, u in zip(chunks, dns, ut, strict=True)
        ]
        write_jsonl(lay.quality(vid), rows)
        if rows:
            logger.info(f"[{i}/{len(pending)}] {vid}: {len(rows)} clips | OVRL "
                        f"{np.mean([r['dnsmos_ovrl'] for r in rows]):.2f} | UTMOS "
                        f"{np.mean([r['utmos'] for r in rows]):.2f}")


if __name__ == "__main__":
    step_main(main)
