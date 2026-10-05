"""cluster: one global ID per speaker across all episodes (MALE_00001, FEMALE_00002, ...).

The diarizer labels speakers per episode (s0, s1, ...); a podcast host appears in many
episodes. Using the TitaNet embedding of every chunk (speaker_check):
  1. voice print per (episode, local speaker): duration-weighted mean of its chunk
     embeddings, recomputed without chunks that do not match it (cosine < outlier_similarity);
  2. reliable voice prints (>= min_group_sec of speech) are clustered across episodes
     (agglomerative, average linkage, cosine >= merge_similarity = same person);
  3. one refinement pass moves each voice print to the closest speaker centroid;
  4. small voice prints join the closest speaker if similar enough, else stay their own speaker;
  5. gender per speaker from several of its chunks (wav2vec2 classifier, mean probability).
IDs are numbered by total hours (largest speaker first), with the gender as prefix.
Runs on every chunk (no quality filter needed); writes meta/speakers.json
{"speakers": [...], "assignment": {chunk id: speaker id}} and prints the statistics.
"""

import logging
import random
from collections import defaultdict

import numpy as np

from egyspeech.config import Section
from egyspeech.io import read_audio, read_json, read_jsonl, resample, write_json
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main, videos

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


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def voice_prints(cfg: Section) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray, dict, dict]:
    """(group keys, voice prints [G, 192], seconds per group, chunks per group, chunk rows by id)."""
    lay = layout(cfg)
    sp = cfg.speakers
    vids = [v["id"] for v in videos(cfg) if lay.chunks_meta(v["id"]).exists() and lay.speaker_emb(v["id"]).exists()]
    keys, prints, secs, members, rows = [], [], [], {}, {}
    with StepBar("cluster (voice prints)", len(vids)) as bar:
        for vid in vids:
            meta = {r["id"]: r for r in read_jsonl(lay.chunks_meta(vid))}
            z = np.load(lay.speaker_emb(vid))
            embs = unit(z["embeddings"].astype(np.float32))
            ids = z["ids"].tolist()
            by_spk: dict[int, list[int]] = defaultdict(list)
            for i, (cid, spk) in enumerate(zip(ids, z["local_speaker"].tolist(), strict=True)):
                if cid in meta:
                    by_spk[int(spk)].append(i)
                    rows[cid] = meta[cid]
            for spk, idx in by_spk.items():
                e = embs[idx]
                w = np.array([meta[ids[i]]["duration"] for i in idx])
                c = unit((e * w[:, None]).sum(0))
                good = e @ c >= sp.outlier_similarity
                if good.sum() >= max(1, len(idx) // 4):  # drop chunks that do not match the voice print
                    c = unit((e[good] * w[good, None]).sum(0))
                keys.append((vid, spk))
                prints.append(c)
                secs.append(float(w.sum()))
                members[(vid, spk)] = [ids[i] for i in idx]
            bar.advance()
    return keys, np.array(prints, dtype=np.float32), np.array(secs), members, rows


def assign_speakers(prints: np.ndarray, secs: np.ndarray, merge_similarity: float, min_group_sec: float) -> np.ndarray:
    """Cluster label per voice print (see module doc, steps 2-4)."""
    n = len(prints)
    labels = np.full(n, -1)
    big = np.flatnonzero(secs >= min_group_sec)
    if len(big) == 0:
        big = np.arange(n)
    labels[big] = cluster_embeddings(prints[big], merge_similarity) if len(big) > 1 else 0

    def centroids():
        ids = np.unique(labels[labels >= 0])
        cents = np.stack([unit((prints[labels == k] * secs[labels == k, None]).sum(0)) for k in ids])
        return ids, cents

    ids, cents = centroids()  # refinement: move each voice print to its closest speaker
    labels[big] = ids[np.argmax(prints[big] @ cents.T, axis=1)]
    ids, cents = centroids()
    nxt = labels.max() + 1
    for i in np.flatnonzero(labels < 0):  # small voice prints: closest speaker, or a new one
        sim = prints[i] @ cents.T
        if sim.max() >= merge_similarity:
            labels[i] = ids[int(np.argmax(sim))]
        else:
            labels[i] = nxt
            nxt += 1
    return labels


def main(cfg: Section, args):
    lay = layout(cfg)
    sp = cfg.speakers
    keys, prints, secs, members, rows = voice_prints(cfg)
    if not keys:
        raise SystemExit("no chunk embeddings: run segment and speaker_check first")
    labels = assign_speakers(prints, secs, sp.merge_similarity, sp.min_group_sec)

    hours: dict[int, float] = defaultdict(float)
    for k, lab in zip(keys, labels, strict=True):
        hours[int(lab)] += sum(rows[c]["duration"] for c in members[k]) / 3600
    videos_meta = {v["id"]: v for v in videos(cfg)}
    spk: dict[int, dict] = {}
    for k, lab in zip(keys, labels, strict=True):
        s = spk.setdefault(int(lab), {"clips": [], "videos": set(), "channels": set()})
        s["clips"].extend(members[k])
        s["videos"].add(k[0])
        ch = (videos_meta.get(k[0]) or {}).get("channel")
        if ch:
            s["channels"].add(ch)

    classifier = GenderClassifier(sp.gender_model)
    rng = random.Random(0)
    with StepBar("cluster (gender)", len(spk), unit="speakers") as bar:
        for s in spk.values():
            longest = sorted(s["clips"], key=lambda c: -rows[c]["duration"])[: 4 * sp.gender_samples]
            probs = []
            for cid in rng.sample(longest, min(sp.gender_samples, len(longest))):
                wav, sr = read_audio(rows[cid]["path"])
                probs.append(classifier.female_prob(resample(wav, sr, 16000)))
            pf = float(np.mean(probs))
            s["gender"] = "female" if pf >= 0.5 else "male"
            s["gender_confidence"] = round(abs(pf - 0.5) * 2, 3)
            bar.advance()

    order = sorted(spk, key=lambda lab: -hours[lab])
    speakers, assignment = [], {}
    for i, lab in enumerate(order, 1):
        s = spk[lab]
        sid = f"{s['gender'].upper()}_{i:05d}"
        for c in s["clips"]:
            assignment[c] = sid
        speakers.append({"id": sid, "gender": s["gender"], "gender_confidence": s["gender_confidence"],
                         "hours": round(hours[lab], 4), "n_clips": len(s["clips"]), "n_videos": len(s["videos"]),
                         "videos": sorted(s["videos"]), "channels": sorted(s["channels"])})
    write_json(lay.speakers, {"speakers": speakers, "assignment": assignment})
    report(speakers)


def report(speakers: list[dict]) -> None:
    h = np.array([s["hours"] for s in speakers])
    by_g = defaultdict(list)
    for s in speakers:
        by_g[s["gender"]].append(s["hours"])
    logger.info(f"{len(speakers)} speakers, {h.sum():.1f} h | per speaker: mean {h.mean():.2f} h, median "
                f"{np.median(h):.2f} h, min {h.min() * 60:.1f} min, max {h.max():.1f} h | speakers with >= 5 min: "
                f"{(h >= 5 / 60).sum()}, >= 1 h: {(h >= 1).sum()}, >= 10 h: {(h >= 10).sum()}")
    for g, hs in sorted(by_g.items()):
        logger.info(f"  {g}: {len(hs)} speakers, {sum(hs):.1f} h ({sum(hs) / max(h.sum(), 1e-9):.0%})")
    top = ", ".join(f"{s['id']} {s['hours']:.1f} h ({s['n_videos']} videos)" for s in speakers[:10])
    logger.info(f"  largest: {top}")


def load_speakers(cfg: Section) -> dict:
    return read_json(layout(cfg).speakers)


if __name__ == "__main__":
    step_main(main)
