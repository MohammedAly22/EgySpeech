"""QwenCleo-ASR (mohammedaly22/QwenCleo-ASR): Egyptian Arabic + Arabic/English code-switching."""

from egyspeech.config import Section


class QwenCleoBackend:
    name = "qwencleo"

    def __init__(self, cfg: Section):
        import torch
        from qwencleo_asr import QwenCleoASR

        c = cfg.transcription
        cuda = torch.cuda.is_available()
        dtype = "bfloat16" if cuda and torch.cuda.is_bf16_supported(including_emulation=False) else "float16"
        self.asr = QwenCleoASR(
            c.qwencleo.model,
            device="cuda:0" if cuda else "cpu",
            dtype=dtype if cuda else "float32",
            max_new_tokens=448,  # a fast 30 s Egyptian clip needs ~200-300 tokens
            default_language=c.language,
        )
        self.language = c.language
        self.batch_size = c.batch_size

    def transcribe(self, paths: list[str], durations: list[float]) -> list[dict]:
        out = []
        for b0 in range(0, len(paths), self.batch_size):
            batch = paths[b0 : b0 + self.batch_size]
            res = self.asr.transcribe(batch, language=self.language, normalize=False)
            if not isinstance(res, list):
                res = [res]
            for r in res:
                text = getattr(r, "text", r if isinstance(r, str) else str(r))
                out.append({"text": text.strip(), "raw": text})
        return out
