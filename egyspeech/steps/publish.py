"""Step 15 — publish: push the final dataset to the Hugging Face Hub (private by default).

Runs only when publish.enabled is true. Uploads train/validation/test with an
Audio column (24 kHz), all metadata columns, the generated dataset card and the
analysis figures. Authentication: `huggingface-cli login` or HF_TOKEN.
"""

import json
import logging

from egyspeech.config import Section
from egyspeech.io import read_json, read_jsonl
from egyspeech.steps import layout, step_main

logger = logging.getLogger("publish")

COLUMNS = ["id", "audio", "text", "text_tagged", "tags", "duration", "speaker_id", "gender", "code_switched",
           "words", "channel", "video_id", "video_url", "start", "end", "source", "dnsmos_ovrl", "dnsmos_sig",
           "dnsmos_bak", "utmos", "align_score", "asr_backend"]


def rows_for(split: list[dict]) -> list[dict]:
    out = []
    for r in split:
        row = {k: r.get(k) for k in COLUMNS if k != "audio"}
        row["audio"] = r["path"]
        row["text_tagged"] = r.get("text_tagged") or r["text"]
        row["tags"] = r.get("tags") or []
        row["words"] = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in r["words"]]
        out.append(row)
    return out


def dataset_card(cfg: Section, stats: dict, repo_id: str, graphs: list[str]) -> str:
    s = stats
    gh = ", ".join(f"{k}: {v} h" for k, v in s.get("gender_hours", {}).items())
    gs = ", ".join(f"{k}: {v}" for k, v in s.get("gender_speakers", {}).items())
    tags = s.get("tags") or {}
    tag_line = ", ".join(f"`{k}` {v}" for k, v in sorted(tags.items(), key=lambda kv: -kv[1])) or "none"
    imgs = "\n".join(f"![{g}](graphs/{g}.png)" for g in graphs)
    split_rows = "\n".join(f"| {k} | {v['clips']:,} | {v['hours']} |" for k, v in s["splits"].items())
    return f"""---
language:
- ar
- arz
- en
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- egyptian-arabic
- code-switching
- speech
- tts
pretty_name: EgySpeech
---

# EgySpeech — Egyptian Arabic speech for TTS

Single-speaker, studio-cleaned, transcribed and word-aligned clips of natural Egyptian Arabic
(with Arabic–English code-switching), built with the [EgySpeech pipeline](https://github.com/MohammedAly22/EgySpeech).

| | |
|---|---|
| **Total** | **{s['hours']} h**, {s['clips']:,} clips |
| **Speakers** | {s['speakers']} (estimated by voice clustering, from {s['speakers_before_balancing']} before balancing) |
| **Gender (hours)** | {gh} |
| **Gender (speakers)** | {gs} |
| **Code-switched** | {s['code_switched_hours']} h (pure Arabic {s['pure_arabic_hours']} h) |
| **Clip length** | {s['duration_sec']['min']}–{s['duration_sec']['max']} s (median {s['duration_sec']['median']} s) |
| **Speaker balance** | largest speaker = {s['max_speaker_share']:.1%} of the data (Gini {s['gini_before']} → {s['gini_final']}) |
| **Audio** | 24 kHz mono, loudness-normalized to {cfg.segmentation.loudness_lufs} LUFS |
| **Paralinguistic tags** | {tag_line} |

| split | clips | hours |
|---|---|---|
{split_rows}

The `test` split contains only speakers that never appear in `train` / `validation`.

## How it was built
1. YouTube episodes -> mono 24 kHz; music and sound effects removed (vocal stem) only where present.
2. NVIDIA Streaming Sortformer diarization; clips keep a guard distance from every other speaker.
3. Clips of {cfg.segmentation.min_sec}-{cfg.segmentation.max_sec} s cut at real pauses (never inside a word).
4. Quality filter: {', '.join(f'{k} = {v}' for k, v in cfg.filter.items() if v is not None and k != 'drop_weak_cuts')} (TitaNet voice consistency = every 3 s window must match the clip's voice).
5. Transcription with `{cfg.transcription.backend}`; transcripts checked for hallucination / truncation.
6. MMS forced alignment (Arabic + English words) with word timestamps; clips whose transcript does not match the audio are removed.
7. Speakers clustered across episodes and capped so no voice dominates.

## Columns
`audio`, `text`, `text_tagged` (with paralinguistic tags when available), `tags`, `words` (word-level
timestamps), `speaker_id`, `gender`, `code_switched`, `duration`, quality scores (`dnsmos_*`, `utmos`),
`align_score`, provenance (`channel`, `video_id`, `video_url`, `start`, `end`, `source`).

## Figures
{imgs}

## Responsible use
The audio comes from publicly available YouTube content; rights remain with the original creators and
the voices belong to real people. Do not use this dataset to impersonate or clone identifiable
speakers without their consent, and label synthetic audio as synthetic.
"""


def main(cfg: Section, args):
    if not cfg.publish.enabled:
        logger.info("publish.enabled is false: nothing to push (set it to true in the config)")
        return
    from datasets import Audio, Dataset, DatasetDict
    from huggingface_hub import HfApi

    lay = layout(cfg)
    p = cfg.publish
    splits = {}
    for name in ("train", "validation", "test"):
        rows = read_jsonl(lay.split(name))
        if rows:
            splits[name] = Dataset.from_list(rows_for(rows)).cast_column(
                "audio", Audio(sampling_rate=cfg.download.sample_rate))
    if not splits:
        raise SystemExit("no final splits: run the balance step first")
    ds = DatasetDict(splits)
    logger.info(f"pushing {', '.join(f'{k}={len(v)}' for k, v in ds.items())} to {p.repo_id} "
                f"({'private' if p.private else 'PUBLIC'})")
    ds.push_to_hub(p.repo_id, private=bool(p.private), max_shard_size=p.max_shard_size)

    api = HfApi()
    stats = read_json(lay.stats) if lay.stats.exists() else None
    graphs = sorted(g.stem for g in lay.graphs.glob("*.png")) if lay.graphs.exists() else []
    if stats:
        lay.hf.mkdir(parents=True, exist_ok=True)
        card = lay.hf / "README.md"
        card.write_text(dataset_card(cfg, stats, p.repo_id, graphs), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=p.repo_id, repo_type="dataset")
        (lay.hf / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(lay.hf / "stats.json"), path_in_repo="stats.json", repo_id=p.repo_id,
                        repo_type="dataset")
    if graphs:
        api.upload_folder(folder_path=str(lay.graphs), path_in_repo="graphs", repo_id=p.repo_id,
                          repo_type="dataset", allow_patterns=["*.png", "*.html"])
    logger.info(f"done: https://huggingface.co/datasets/{p.repo_id}")


if __name__ == "__main__":
    step_main(main)
