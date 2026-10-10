"""QwenCleo-ASR (mohammedaly22/QwenCleo-ASR): Egyptian Arabic + Arabic/English code-switching.

The model is loaded with qwen_asr's Qwen3ASRModel and clips are transcribed in real GPU
batches (QwenCleoASR.transcribe() itself loops over clips one at a time). Clips are sorted
by length so each batch has little padding; results come back in the input order.
"""

from egyspeech.config import Section


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

    def transcribe(self, paths: list[str], durations: list[float]) -> list[dict]:
        order = sorted(range(len(paths)), key=lambda i: -durations[i])  # longest first: OOM shows up early
        out: list[dict] = [{}] * len(paths)
        for b0 in range(0, len(order), self.batch_size):
            idx = order[b0 : b0 + self.batch_size]
            res = self.model.transcribe(audio=[paths[i] for i in idx], language=self.language)
            for i, r in zip(idx, res, strict=True):
                text = (getattr(r, "text", None) or "").strip()
                out[i] = {"text": text, "raw": text}
        return out
