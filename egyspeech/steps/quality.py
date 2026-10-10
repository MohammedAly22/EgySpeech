"""Step 6 — quality: DNSMOS (P.835 SIG / BAK / OVRL) per clip, optionally UTMOS and DNSMOS P.808.

quality.fast: true (default)
  DNSMOS P.835 runs on the GPU (onnxruntime CUDA) in large batches, with Microsoft's
  reference algorithm (9.01 s windows, 1 s hop, short clips repeated, polynomial mapping);
  clips are read and resampled by a pool of threads. The P.808 model is skipped.
quality.fast: false
  torchmetrics DNSMOS (SIG / BAK / OVRL + P.808) in single-threaded CPU worker processes.
quality.utmos: true adds UTMOS (GPU). Neither P.808 nor UTMOS is used by the default filter.
Thresholds are applied by the filter step, so they can be tuned without recomputing anything.
"""

import logging
from pathlib import Path
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


SR16 = 16000
WIN = int(9.01 * SR16)  # DNSMOS P.835 input: 9.01 s at 16 kHz
POLY = {"sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
        "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
        "ovr": np.poly1d([-0.06766283, 1.11546468, 0.04602535])}


def dnsmos_windows(wav16: np.ndarray) -> np.ndarray:
    """[n, 9.01 s] windows of one clip, as in Microsoft's dnsmos_local.py (1 s hop, short clips repeated)."""
    x = wav16.astype(np.float32)
    while len(x) < WIN:
        x = np.concatenate([x, x])
    n = int(np.floor(len(x) / SR16) - 9.01) + 1
    # the reference slices x[int(i * 16000) : int((i + 9.01) * 16000)]; float rounding makes some of these
    # one sample short and the reference skips them: reproduced here so the scores match exactly
    starts = [int(i * SR16) for i in range(n) if int((i + 9.01) * SR16) - int(i * SR16) >= WIN] or [0]
    return np.stack([x[s : s + WIN] for s in starts])


class FastDnsmos:
    """DNSMOS P.835 model converted from Microsoft's ONNX file to PyTorch (onnx2torch), run in GPU batches."""

    def __init__(self, batch_windows: int = 64):  # ~70 MB of activations per window
        import torch
        from onnx2torch import convert

        path = Path.home() / ".torchmetrics" / "DNSMOS" / "DNSMOS" / "sig_bak_ovr.onnx"
        if not path.exists():  # torchmetrics downloads the official model files
            from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

            deep_noise_suppression_mean_opinion_score(torch.zeros(SR16), SR16, personalized=False, device="cpu")
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = convert(str(path)).eval().to(self.torch_device)
        self.device = "GPU" if self.torch_device == "cuda" else "CPU"
        if self.device == "CPU":
            logger.warning("no CUDA GPU: DNSMOS runs on the CPU (slow)")
        self.batch = batch_windows

    def _run(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            return self.model(torch.from_numpy(x).to(self.torch_device)).float().cpu().numpy()

    def score(self, wavs: list[np.ndarray]) -> list[tuple[float, float, float]]:
        """(sig, bak, ovrl) per clip."""
        wins = [dnsmos_windows(w) for w in wavs]
        owner = np.repeat(np.arange(len(wins)), [len(w) for w in wins])
        flat = np.concatenate(wins) if wins else np.zeros((0, WIN), np.float32)
        raw = np.zeros((len(flat), 3), dtype=np.float32)
        for b0 in range(0, len(flat), self.batch):
            raw[b0 : b0 + self.batch] = self._run(flat[b0 : b0 + self.batch])
        sig, bak, ovr = POLY["sig"](raw[:, 0]), POLY["bak"](raw[:, 1]), POLY["ovr"](raw[:, 2])
        out = []
        for k in range(len(wins)):
            m = owner == k
            out.append((float(sig[m].mean()), float(bak[m].mean()), float(ovr[m].mean())))
        return out


def main_fast(cfg: Section, args, pending: list[dict], chunks: dict) -> list[dict]:
    lay = layout(cfg)
    dns = FastDnsmos(int(cfg.quality.get("dnsmos_batch", 64)))
    utmos = Utmos(cfg.quality.batch_size) if cfg.quality.get("utmos") else None
    logger.info(f"DNSMOS P.835 on {dns.device}" + (", UTMOS on GPU" if utmos else " (UTMOS off)"))
    failures = []
    clip_sec = sum(c["duration"] for cs in chunks.values() for c in cs)
    with ThreadPoolExecutor(int(cfg.quality.get("load_threads", 12))) as io, ThreadPoolExecutor(1) as ahead, \
            StepBar("quality", len(pending), audio_sec=clip_sec) as bar:
        def load_video(v):
            return list(io.map(lambda c: load16(c["path"]), chunks[v["id"]]))

        for v, wavs in prefetch(ahead, load_video, pending, depth=2):
            vid, cs = v["id"], chunks[v["id"]]
            bar.status(vid)
            try:
                scores = dns.score(wavs)
                ut = utmos.score(cs) if utmos and cs else [None] * len(cs)
                rows = [{"id": c["id"], "dnsmos_p808": None, "dnsmos_sig": round(sg, 3), "dnsmos_bak": round(bk, 3),
                         "dnsmos_ovrl": round(ov, 3), "utmos": None if u is None else round(u, 3)}
                        for c, (sg, bk, ov), u in zip(cs, scores, ut, strict=True)]
                write_jsonl(lay.quality(vid), rows)
                if rows:
                    logger.info(f"{vid}: {len(rows)} clips | BAK {np.mean([r['dnsmos_bak'] for r in rows]):.2f} "
                                f"| OVRL {np.mean([r['dnsmos_ovrl'] for r in rows]):.2f}")
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                failures.append({"id": vid, "error": f"{type(exc).__name__}: {exc}"[:500]})
                logger.warning(f"{vid}: {type(exc).__name__}: {str(exc)[:200]}")
            bar.advance(audio_sec=sum(c["duration"] for c in cs))
    return failures


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
    if cfg.quality.get("fast", True):
        failures = main_fast(cfg, args, pending, chunks)
        write_failures(cfg, "quality", failures)
        logger.info(f"scored {len(pending) - len(failures)} videos, {len(failures)} failed")
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
