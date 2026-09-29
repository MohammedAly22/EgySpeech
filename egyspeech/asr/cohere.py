"""Cohere Transcribe Arabic (CohereLabs/cohere-transcribe-arabic-07-2026), Transformers >= 5.4."""

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_audio, resample


class CohereBackend:
    name = "cohere"

    def __init__(self, cfg: Section):
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_id = cfg.transcription.cohere.model
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", dtype=dtype if torch.cuda.is_available() else torch.float32
        ).eval()
        self.batch_size = cfg.transcription.batch_size

    def transcribe(self, paths: list[str], durations: list[float]) -> list[dict]:
        import torch

        audios = []
        for p in paths:
            wav, sr = read_audio(p)
            audios.append(resample(wav, sr, 16000))
        results: list[dict] = [{}] * len(paths)
        order = np.argsort(durations)
        for b0 in range(0, len(order), self.batch_size):
            idx = list(order[b0 : b0 + self.batch_size])
            inputs = self.processor([audios[i] for i in idx], sampling_rate=16000, return_tensors="pt",
                                    language="ar")
            inputs.to(self.model.device, dtype=self.model.dtype)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=448)
            texts = self.processor.batch_decode(out, skip_special_tokens=True)
            for i, t in zip(idx, texts, strict=True):
                results[i] = {"text": t.strip(), "raw": t}
        return results
