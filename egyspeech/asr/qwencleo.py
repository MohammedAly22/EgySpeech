"""QwenCleo-ASR (mohammedaly22/QwenCleo-ASR): Egyptian Arabic + Arabic/English code-switching."""

import inspect

from egyspeech.config import Section


class QwenCleoBackend:
    name = "qwencleo"

    def __init__(self, cfg: Section):
        from qwencleo_asr import QwenCleoASR

        c = cfg.transcription
        kwargs = {}
        params = inspect.signature(QwenCleoASR).parameters
        for key in ("model", "model_id", "model_name", "repo_id"):
            if key in params:
                kwargs[key] = c.qwencleo.model
                break
        self.asr = QwenCleoASR(**kwargs)
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
