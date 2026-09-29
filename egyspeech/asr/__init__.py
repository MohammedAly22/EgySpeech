"""ASR backends. Each returns, per clip, {"text": str, "raw": str} (+ "text_tagged", "tags" for llm).

    qwencleo  QwenCleo-ASR (Qwen3-ASR fine-tuned for Egyptian + code-switching)   env: egyspeech-qwen
    cohere    Cohere Transcribe Arabic (Transformers)                            env: egyspeech
    parakeet  NVIDIA NeMo FastConformer / Parakeet checkpoints                   env: egyspeech-nemo
    llm       any OpenAI-compatible audio LLM (vLLM + Qwen3-Omni by default)      env: egyspeech
"""

from egyspeech.config import Section


def get_backend(name: str, cfg: Section):
    if name == "qwencleo":
        from egyspeech.asr.qwencleo import QwenCleoBackend

        return QwenCleoBackend(cfg)
    if name == "cohere":
        from egyspeech.asr.cohere import CohereBackend

        return CohereBackend(cfg)
    if name == "parakeet":
        from egyspeech.asr.parakeet import ParakeetBackend

        return ParakeetBackend(cfg)
    if name == "llm":
        from egyspeech.asr.llm import LLMBackend

        return LLMBackend(cfg)
    raise ValueError(f"unknown ASR backend {name!r} (qwencleo | cohere | parakeet | llm)")
