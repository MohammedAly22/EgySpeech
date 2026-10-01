"""Step 6 — quality: DNSMOS (P.835 SIG/BAK/OVRL + P.808) and UTMOS per clip.

DNSMOS runs on the CPU in a pool of single-threaded worker processes while UTMOS runs
on the GPU in this process, both at the same time: the pool works one video ahead.
Clips are loaded in small sorted batches (bounded RAM). Thresholds are applied by the
filter step, so they can be tuned without recomputing anything.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_audio, read_jsonl, resample, write_jsonl
from egyspeech.parallel import prefetch
from egyspeech.progress import StepBar
from egyspeech.steps import layout, select, step_main, write_failures

logger = logging.getLogger("quality")


def _init_worker():
    import torch

    torch.set_num_threads(1)


def dnsmos_file(path: str) -> list[float]:
    """[p808_mos, sig, bak, ovrl] of one clip (CPU, one thread)."""
    import torch
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    wav, sr = read_audio(path)
    out = deep_noise_suppression_mean_opinion_score(torch.from_numpy(resample(wav, sr, 16000)), 16000,
                                                    personalized=False, device="cpu", num_threads=1)
    return [float(v) for v in out.tolist()]


def load16(path: str) -> np.ndarray:
    wav, sr = read_audio(path)
    return resample(wav, sr, 16000)


class Utmos:
    def __init__(self, batch_size: int):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        self.model.to(self.device).eval()
        self.batch = batch_size
        self.loader = ThreadPoolExecutor(4)

    def score(self, chunks: list[dict]) -> list[float]:
        """UTMOS per clip; clips of similar length are batched (each batch cut to its shortest clip)."""
        import torch

        order = np.argsort([c["duration"] for c in chunks])
        batches = [order[i : i + self.batch] for i in range(0, len(order), self.batch)]
        result = [0.0] * len(chunks)

        def load(idx):
            return [load16(chunks[i]["path"]) for i in idx]

        for idx, wavs in prefetch(self.loader, load, batches, depth=2):
            length = min(len(w) for w in wavs)
            x = torch.from_numpy(np.stack([w[:length] for w in wavs])).to(self.device)
            with torch.inference_mode():
                s = self.model(x, 16000).float().cpu().numpy()
            for i, v in zip(idx, s, strict=True):
                result[int(i)] = float(v)
        return result


def main(cfg: Section, args):
    lay = layout(cfg)
    pending, n_done = select(cfg, args, ready=lambda v: lay.chunks_meta(v["id"]).exists(),
                             done=lambda v: lay.quality(v["id"]).exists())
    chunks = {v["id"]: read_jsonl(lay.chunks_meta(v["id"])) for v in pending}
    n_clips = sum(len(c) for c in chunks.values())
    logger.info(f"{n_done} done, {len(pending)} videos / {n_clips} clips to score "
                f"({cfg.quality.workers} DNSMOS processes + UTMOS on GPU)")
    if not pending:
        return
    import torch
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    # fetch the DNSMOS ONNX models once here, not concurrently from every worker
    deep_noise_suppression_mean_opinion_score(torch.zeros(16000), 16000, personalized=False, device="cpu")
    utmos = Utmos(cfg.quality.batch_size)
    failures = []
    clip_sec = sum(c["duration"] for cs in chunks.values() for c in cs)
    with ProcessPoolExecutor(max_workers=int(cfg.quality.workers), mp_context=get_context("spawn"),
                             initializer=_init_worker) as pool, \
            StepBar("quality", len(pending), audio_sec=clip_sec) as bar:
        dns_futs: dict[str, list] = {}

        def submit(k: int):
            if k < len(pending):
                vid = pending[k]["id"]
                dns_futs[vid] = [pool.submit(dnsmos_file, c["path"]) for c in chunks[vid]]

        submit(0)
        for k, v in enumerate(pending):
            vid = v["id"]
            submit(k + 1)  # keep the CPU pool one video ahead of the GPU
            bar.status(vid)
            cs = chunks[vid]
            try:
                ut = utmos.score(cs) if cs else []
                dns = [f.result() for f in dns_futs.pop(vid)]
                rows = [{"id": c["id"], "dnsmos_p808": round(d[0], 3), "dnsmos_sig": round(d[1], 3),
                         "dnsmos_bak": round(d[2], 3), "dnsmos_ovrl": round(d[3], 3), "utmos": round(u, 3)}
                        for c, d, u in zip(cs, dns, ut, strict=True)]
                write_jsonl(lay.quality(vid), rows)
                if rows:
                    logger.info(f"{vid}: {len(rows)} clips | OVRL {np.mean([r['dnsmos_ovrl'] for r in rows]):.2f} "
                                f"| UTMOS {np.mean([r['utmos'] for r in rows]):.2f}")
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                for f in dns_futs.pop(vid, []):
                    f.cancel()
                failures.append({"id": vid, "error": f"{type(exc).__name__}: {exc}"[:500]})
                logger.warning(f"{vid}: {type(exc).__name__}: {str(exc)[:200]}")
            bar.advance(audio_sec=sum(c["duration"] for c in cs))
    utmos.loader.shutdown()
    write_failures(cfg, "quality", failures)
    logger.info(f"scored {len(pending) - len(failures)} videos, {len(failures)} failed")


if __name__ == "__main__":
    step_main(main)
