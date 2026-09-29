"""NVIDIA NeMo ASR (Parakeet / FastConformer checkpoints), e.g.
nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 (Arabic, MSA-oriented)."""

from egyspeech.config import Section


def _text(h) -> str:
    if isinstance(h, str):
        return h
    return getattr(h, "text", str(h))


class ParakeetBackend:
    name = "parakeet"

    def __init__(self, cfg: Section):
        import torch
        from nemo.collections.asr.models import ASRModel

        self.model = ASRModel.from_pretrained(cfg.transcription.parakeet.model)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.batch_size = cfg.transcription.batch_size

    def transcribe(self, paths: list[str], durations: list[float]) -> list[dict]:
        import torch

        with torch.inference_mode():
            out = self.model.transcribe(paths, batch_size=self.batch_size, verbose=False)
        if isinstance(out, tuple):  # hybrid / RNNT models may return (best, all)
            out = out[0]
        return [{"text": _text(h).strip(), "raw": _text(h)} for h in out]
