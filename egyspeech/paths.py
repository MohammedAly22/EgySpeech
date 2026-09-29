"""On-disk layout under work_dir.

    meta/videos.jsonl                    collected unique videos
    audio/raw/<vid>.mp3 (+ .info.json)   downloaded episodes (24 kHz mono)
    audio/vocals/<vid>.flac              vocal stem (music / effects removed)
    diar/<vid>.json, diar/<vid>.npy      Sortformer segments + frame probabilities
    chunks/<vid>/<chunk_id>.flac         single-speaker clips (5-30 s, loudness normalized)
    meta/chunks/<vid>.jsonl              clip boundaries and segmentation info
    meta/quality/<vid>.jsonl             DNSMOS / UTMOS / clipping
    meta/speakers/<vid>.npz              TitaNet embeddings + single-speaker consistency
    meta/filtered.jsonl                  clips passing the quality filter
    meta/transcripts/<backend>/<vid>.jsonl
    meta/aligned/<vid>.jsonl             word timestamps + alignment score
    meta/speakers.json                   global speakers (clusters), gender
    final/<split>.jsonl, final/stats.json
    graphs/                              analysis figures (html / png)
    hf/                                  dataset card for publishing
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    # meta
    @property
    def meta(self) -> Path:
        return self.root / "meta"

    @property
    def videos(self) -> Path:
        return self.meta / "videos.jsonl"

    @property
    def download_failures(self) -> Path:
        return self.meta / "download_failures.jsonl"

    def raw_audio(self, vid: str, fmt: str = "mp3") -> Path:
        return self.root / "audio" / "raw" / f"{vid}.{fmt}"

    def info_json(self, vid: str) -> Path:
        return self.root / "audio" / "raw" / f"{vid}.info.json"

    def vocals(self, vid: str) -> Path:
        return self.root / "audio" / "vocals" / f"{vid}.flac"

    def diar_json(self, vid: str) -> Path:
        return self.root / "diar" / f"{vid}.json"

    def diar_probs(self, vid: str) -> Path:
        return self.root / "diar" / f"{vid}.npy"

    def chunk_dir(self, vid: str) -> Path:
        return self.root / "chunks" / vid

    def chunks_meta(self, vid: str) -> Path:
        return self.meta / "chunks" / f"{vid}.jsonl"

    def quality(self, vid: str) -> Path:
        return self.meta / "quality" / f"{vid}.jsonl"

    def speaker_emb(self, vid: str) -> Path:
        return self.meta / "speakers" / f"{vid}.npz"

    @property
    def filtered(self) -> Path:
        return self.meta / "filtered.jsonl"

    @property
    def filter_report(self) -> Path:
        return self.meta / "filter_report.json"

    def transcripts(self, backend: str, vid: str) -> Path:
        return self.meta / "transcripts" / backend / f"{vid}.jsonl"

    def aligned(self, vid: str) -> Path:
        return self.meta / "aligned" / f"{vid}.jsonl"

    @property
    def speakers(self) -> Path:
        return self.meta / "speakers.json"

    @property
    def final(self) -> Path:
        return self.root / "final"

    def split(self, name: str) -> Path:
        return self.final / f"{name}.jsonl"

    @property
    def stats(self) -> Path:
        return self.final / "stats.json"

    @property
    def graphs(self) -> Path:
        return self.root / "graphs"

    @property
    def hf(self) -> Path:
        return self.root / "hf"
