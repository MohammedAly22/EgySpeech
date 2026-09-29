"""Step 10 (optional) — verify: a second ASR backend for agreement filtering.

Transcribes the same clips with transcription.verify.backend. The balance step then
drops clips whose two transcripts differ by more than verify.max_cer (after
orthography-insensitive normalization). Skipped when verify.backend is null.
"""

import logging

from egyspeech.config import Section
from egyspeech.steps import step_main
from egyspeech.steps.transcribe import run

logger = logging.getLogger("verify")


def main(cfg: Section, args):
    backend = cfg.transcription.verify.backend
    if not backend:
        logger.info("transcription.verify.backend is null: nothing to do")
        return
    if backend == cfg.transcription.backend:
        raise SystemExit("verify.backend must differ from transcription.backend")
    run(cfg, args, backend)


if __name__ == "__main__":
    step_main(main)
