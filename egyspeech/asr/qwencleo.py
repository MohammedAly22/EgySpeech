"""QwenCleo-ASR (mohammedaly22/QwenCleo-ASR): Egyptian Arabic + Arabic/English code-switching.

The model is loaded with qwen_asr's Qwen3ASRModel and clips are transcribed in real GPU
batches (QwenCleoASR.transcribe() itself loops over clips one at a time). Clips are sorted
by length so each batch has little padding; the next batch's audio is read and resampled
by a thread pool while the GPU works on the current one. Results come back in input order.
"""

from concurrent.futures import ThreadPoolExecutor

from egyspeech.config import Section
from egyspeech.io import read_audio, resample
from egyspeech.parallel import prefetch


def _load16(path: str):
    wav, sr = read_audio(path)
    return resample(wav, sr, 16000), 16000


class QwenCleoBackend:
    name = "qwencleo"

    def __init__(self, cfg: Section):
        import logging

        import torch
        from qwen_asr import Qwen3ASRModel

        for name in ("transformers", "qwen_asr"):
            logging.getLogger(name).setLevel(logging.ERROR)
        c = cfg.transcription
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        self.batch_size = int(c.batch_size)
        self.model = Qwen3ASRModel.from_pretrained(
            c.qwencleo.model,
            dtype=dtype if cuda else torch.float32,
            device_map="cuda:0" if cuda else "cpu",
            max_new_tokens=448,  # a fast 30 s Egyptian clip needs ~200-300 tokens
            max_inference_batch_size=self.batch_size,
        )
        self.language = None if c.language in (None, "None") else c.language
        self.io = ThreadPoolExecutor(int(c.get("load_threads", 16)))
        self.ahead = ThreadPoolExecutor(1)

    def transcribe(self, paths: list[str], durations: list[float]) -> list[dict]:
        order = sorted(range(len(paths)), key=lambda i: -durations[i])  # longest first: OOM shows up early
        batches = [order[b : b + self.batch_size] for b in range(0, len(order), self.batch_size)]
        out: list[dict] = [{}] * len(paths)

        def load(idx):
            return list(self.io.map(_load16, [paths[i] for i in idx]))

        for idx, audio in prefetch(self.ahead, load, batches, depth=2):
            res = self.model.transcribe(audio=audio, language=self.language)
            for i, r in zip(idx, res, strict=True):
                text = (getattr(r, "text", None) or "").strip()
                out[i] = {"text": text, "raw": text}
        return out
