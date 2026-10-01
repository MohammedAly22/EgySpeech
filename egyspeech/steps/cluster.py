"""Step 12 — cluster: global speaker identities across videos + gender.

The diarizer labels speakers per video (s0, s1, ...). A podcast host appears in
hundreds of episodes, so per-video speakers are merged into global speakers:
mean TitaNet embedding per (video, local speaker) -> agglomerative clustering with
cosine similarity >= speakers.merge_similarity. Gender is classified per global
speaker from several of its clips (majority of probabilities).
"""

import logging
import random
from collections import defaultdict

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_audio, read_json, read_jsonl, resample, write_json
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main

logger = logging.getLogger("cluster")


def cluster_embeddings(embs: np.ndarray, merge_similarity: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(embs) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=1.0 - merge_similarity
    )
    return model.fit_predict(embs)


class GenderClassifier:
    def __init__(self, name: str):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModelForAudioClassification.from_pretrained(name).to(self.device).eval()
        self.labels = {i: l.lower() for i, l in self.model.config.id2label.items()}

    def female_prob(self, wav16: np.ndarray) -> float:
        import torch

        x = self.fe(wav16[: 16000 * 12], sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            p = self.model(**x).logits.softmax(-1)[0].cpu().numpy()
        return float(sum(p[i] for i, l in self.labels.items() if l.startswith("f")))


def main(cfg: Section, args):
    lay = layout(cfg)
    clips = {r["id"]: r for r in read_jsonl(lay.filtered)}
    if not clips:
        raise SystemExit("no filtered clips: run the filter step first")
    groups: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    group_clips: dict[tuple[str, int], list[str]] = defaultdict(list)
    vids = sorted({r["video_id"] for r in clips.values()})
    with StepBar("cluster (embeddings)", len(vids)) as bar:
        for vid in vids:
            z = np.load(lay.speaker_emb(vid))
            for cid, emb, spk in zip(z["ids"].tolist(), z["embeddings"], z["local_speaker"].tolist(), strict=True):
                if cid in clips:
                    groups[(vid, int(spk))].append(emb.astype(np.float32))
                    group_clips[(vid, int(spk))].append(cid)
            bar.advance()
    keys = list(groups)
    means = np.stack([np.mean(groups[k], axis=0) for k in keys])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
    labels = cluster_embeddings(means, cfg.speakers.merge_similarity)

    # stable global ids ordered by amount of speech
    hours: dict[int, float] = defaultdict(float)
    for k, lab in zip(keys, labels, strict=True):
        hours[int(lab)] += sum(clips[c]["duration"] for c in group_clips[k]) / 3600
    order = sorted(hours, key=lambda lab: -hours[lab])
    gid = {lab: f"spk_{i:05d}" for i, lab in enumerate(order)}

    videos_meta = {v["id"]: v for v in read_jsonl(lay.videos)}
    speakers: dict[str, dict] = {}
    assignment: dict[str, str] = {}
    for k, lab in zip(keys, labels, strict=True):
        g = gid[int(lab)]
        sp = speakers.setdefault(g, {"id": g, "clips": [], "videos": set(), "channels": set(), "hours": 0.0})
        sp["clips"].extend(group_clips[k])
        sp["videos"].add(k[0])
        ch = (videos_meta.get(k[0]) or {}).get("channel")
        if ch:
            sp["channels"].add(ch)
        for c in group_clips[k]:
            assignment[c] = g

    classifier = GenderClassifier(cfg.speakers.gender_model)
    rng = random.Random(0)
    bar = StepBar("cluster (gender)", len(speakers), unit="speakers").start()
    for sp in speakers.values():
        bar.advance()
        sample = rng.sample(sp["clips"], min(cfg.speakers.gender_samples, len(sp["clips"])))
        probs = []
        for cid in sample:
            wav, sr = read_audio(clips[cid]["path"])
            probs.append(classifier.female_prob(resample(wav, sr, 16000)))
        pf = float(np.mean(probs))
        sp["gender"] = "female" if pf >= 0.5 else "male"
        sp["gender_confidence"] = round(abs(pf - 0.5) * 2, 3)
        sp["hours"] = round(sum(clips[c]["duration"] for c in sp["clips"]) / 3600, 4)
        sp["n_clips"] = len(sp["clips"])
        sp["videos"] = sorted(sp["videos"])
        sp["channels"] = sorted(sp["channels"])
        del sp["clips"]
    bar.close()
    write_json(lay.speakers, {"speakers": sorted(speakers.values(), key=lambda s: -s["hours"]),
                              "assignment": assignment})
    n_f = sum(s["gender"] == "female" for s in speakers.values())
    logger.info(f"{len(keys)} per-video speakers -> {len(speakers)} global speakers "
                f"({n_f} female / {len(speakers) - n_f} male)")


def load_speakers(cfg: Section) -> dict:
    return read_json(layout(cfg).speakers)


if __name__ == "__main__":
    step_main(main)
