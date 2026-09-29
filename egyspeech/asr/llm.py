"""Audio LLM backend over an OpenAI-compatible API (vLLM serving Qwen3-Omni by default).

The system prompt (configurable) asks for a verbatim Egyptian-Arabic transcript with
English in Latin script; with use_tags it also asks for paralinguistic tags inline
(laughs, sighs, breath ... and spans like [whispering] ... [/whispering]). Tags are
validated against the configured tag set: unknown tags are dropped, unclosed spans
are closed.
"""

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

from egyspeech.config import Section
from egyspeech.io import read_audio, resample
from egyspeech.text import TagSet, clean_llm_output, clean_tags, strip_tags

logger = logging.getLogger("asr.llm")


def audio_data_uri(path: str) -> str:
    wav, sr = read_audio(path)
    wav = resample(wav, sr, 16000)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


class LLMBackend:
    name = "llm"

    def __init__(self, cfg: Section):
        from openai import OpenAI

        c = cfg.transcription.llm
        self.cfg = c
        self.client = OpenAI(base_url=c.url, api_key=c.api_key or "EMPTY", timeout=300)
        self.model = c.model
        served = [m.id for m in self.client.models.list().data]
        if self.model not in served:
            if len(served) == 1:
                logger.warning(f"model {self.model!r} not served, using {served[0]!r}")
                self.model = served[0]
            else:
                raise SystemExit(f"model {self.model!r} is not served at {c.url} (served: {served})")
        t = cfg.transcription.tags
        self.tags = TagSet(list(t.events), list(t.spans), list(t.styles))
        self.system = c.system_prompt.strip()
        if c.use_tags:
            self.system += "\n\n" + c.tags_prompt.strip().format(tags=self.tags.prompt_list())
        self.use_tags = bool(c.use_tags)

    def _one(self, path: str) -> dict:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": audio_data_uri(path)}},
                {"type": "text", "text": "Transcribe this audio."},
            ]},
        ]
        last_exc = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                raw = resp.choices[0].message.content or ""
                text = clean_llm_output(raw)
                if self.use_tags:
                    tagged, found = clean_tags(text, self.tags)
                    return {"text": strip_tags(tagged), "text_tagged": tagged, "tags": found, "raw": raw}
                return {"text": strip_tags(text), "raw": raw}
            except Exception as exc:  # noqa: BLE001 - network / server errors: retry
                last_exc = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM request failed for {path}: {last_exc}")

    def transcribe(self, paths: list[str], durations: list[float]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(self._one, paths))
