"""Step 11 — align: word timestamps for every transcript (Arabic and English words).

MMS CTC forced alignment on romanized text (uroman): one acoustic model covers the
Arabic words and the code-switched English words of the same clip. Per clip:
  * words: [{"word", "start", "end"}]         (what TTS trainers need for cutting / prompts)
  * align_score: mean log-probability of the aligned characters (low = transcript and
    audio disagree -> dropped at balancing)
  * edge_ok: the first word starts / last word ends at least min_edge_sec inside the clip
    (no word cut in the middle)
"""

import logging
import re
from collections import defaultdict

import numpy as np
import torch

from egyspeech.config import Section
from egyspeech.io import read_audio, read_jsonl, resample, write_jsonl
from egyspeech.steps import layout, step_main
from egyspeech.text import strip_tags

logger = logging.getLogger("align")


def batched_viterbi(emissions: torch.Tensor, T: torch.Tensor, token_lists: list[list[int]], blank: int):
    """Viterbi CTC alignment (one token per emitting frame), batched. Adapted from
    pocket-tts training/scripts/align_data.py (MIT, Kyutai).
    Returns per item (frame of each token, log-prob of each token) or None."""
    device = emissions.device
    B, Tmax, _ = emissions.shape
    N = torch.tensor([len(t) for t in token_lists], device=device)
    Nmax = max(1, int(N.max()))
    tok = torch.zeros(B, Nmax, dtype=torch.long, device=device)
    for b, t in enumerate(token_lists):
        if t:
            tok[b, : len(t)] = torch.tensor(t, device=device)
    neg = float("-inf")
    trellis = torch.full((Tmax + 1, B, Nmax + 1), neg, device=device)
    trellis[0, :, 0] = 0.0
    blank_em = emissions[:, :, blank]
    tok_em = emissions.gather(2, tok.unsqueeze(1).expand(B, Tmax, Nmax))
    for t in range(Tmax):
        prev = trellis[t]
        stay = prev + blank_em[:, t : t + 1]
        move = torch.cat([torch.full((B, 1), neg, device=device), prev[:, :-1] + tok_em[:, t]], 1)
        trellis[t + 1] = torch.where((t < T).view(B, 1), torch.maximum(stay, move), prev)
    tr_all = trellis.permute(1, 0, 2).cpu().numpy()
    blank_all = blank_em.float().cpu().numpy()
    tok_all = tok_em.float().cpu().numpy()
    results = []
    for b in range(B):
        n, t_end = int(N[b]), int(T[b])
        tr = tr_all[b]
        if n == 0 or t_end < n or not np.isfinite(tr[t_end, n]):
            results.append(None)
            continue
        frames = [0] * n
        j = n
        for t in range(t_end, 0, -1):
            if j == 0:
                break
            if tr[t - 1, j - 1] + tok_all[b, t - 1, j - 1] >= tr[t - 1, j] + blank_all[b, t - 1]:
                j -= 1
                frames[j] = t - 1
        if j != 0:
            results.append(None)
            continue
        results.append((frames, [float(tok_all[b, f, k]) for k, f in enumerate(frames)]))
    return results


class Aligner:
    def __init__(self, cfg: Section):
        import uroman
        from transformers import AutoTokenizer, Wav2Vec2ForCTC

        name = cfg.alignment.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(name).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()
        tok = AutoTokenizer.from_pretrained(name)
        self.vocab: dict[str, int] = tok.get_vocab()
        self.blank = tok.pad_token_id if tok.pad_token_id is not None else self.vocab.get("<blank>", 0)
        self.delim = self.vocab.get("|")
        self.uroman = uroman.Uroman()
        self.batch_size = cfg.alignment.batch_size

    def romanize(self, word: str) -> list[int]:
        w = re.sub(r"[^\w']", "", word)
        if not w:
            return []
        lang = "ara" if re.search(r"[؀-ۿ]", w) else "eng"
        roman = self.uroman.romanize_string(w, lcode=lang).lower()
        return [self.vocab[c] for c in roman if c in self.vocab and c not in ("|",)]

    def align(self, items: list[tuple[np.ndarray, list[str]]]) -> list[dict | None]:
        out: list[dict | None] = []
        for b0 in range(0, len(items), self.batch_size):
            chunk = items[b0 : b0 + self.batch_size]
            token_lists, word_of = [], []
            for _, words in chunk:
                toks, owners = [], []
                for w_idx, w in enumerate(words):
                    ids = self.romanize(w)
                    if not ids:
                        continue
                    if toks and self.delim is not None:
                        toks.append(self.delim)
                        owners.append(-1)
                    toks.extend(ids)
                    owners.extend([w_idx] * len(ids))
                token_lists.append(toks)
                word_of.append(owners)
            lens = [len(w) for w, _ in chunk]
            x = torch.zeros(len(chunk), max(lens))
            for r, (w, _) in enumerate(chunk):
                x[r, : len(w)] = torch.from_numpy(w)
            # per-utterance normalization, as the MMS feature extractor does
            mask = torch.arange(max(lens))[None, :] < torch.tensor(lens)[:, None]
            mean = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
            var = (((x - mean) * mask) ** 2).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
            x = ((x - mean) / torch.sqrt(var + 1e-7)) * mask
            with torch.inference_mode():
                logits = self.model(x.to(self.device, self.model.dtype), attention_mask=mask.long().to(self.device)).logits
            em = logits.float().log_softmax(-1)
            T = torch.tensor([int(self.model._get_feat_extract_output_lengths(n)) for n in lens], device=self.device)
            paths = batched_viterbi(em, T, token_lists, self.blank)
            for (wav, words), owners, path, t_frames in zip(chunk, word_of, paths, T.tolist(), strict=True):
                if path is None:
                    out.append(None)
                    continue
                frames, scores = path
                spf = (len(wav) / 16000) / t_frames
                spans: dict[int, list[int]] = {}
                word_scores = []
                for f, o, sc in zip(frames, owners, scores, strict=True):
                    if o < 0:
                        continue
                    spans.setdefault(o, [f, f])[1] = f
                    word_scores.append(sc)
                timed = [
                    {"word": w, "start": round(spans[k][0] * spf, 3), "end": round((spans[k][1] + 1) * spf, 3)}
                    if k in spans else {"word": w, "start": None, "end": None}
                    for k, w in enumerate(words)
                ]
                out.append({"words": timed, "align_score": round(float(np.mean(word_scores)), 4) if word_scores else -99.0})
        return out


def main(cfg: Section, args):
    lay = layout(cfg)
    backend = cfg.transcription.backend
    clips = {r["id"]: r for r in read_jsonl(lay.filtered)}
    by_video = defaultdict(list)
    for vid in sorted({r["video_id"] for r in clips.values()}):
        tpath = lay.transcripts(backend, vid)
        if tpath.exists():
            by_video[vid] = [t for t in read_jsonl(tpath) if t["ok"] and t["id"] in clips]
    pending = [v for v in by_video if args.force or not lay.aligned(v).exists()]
    if args.limit:
        pending = pending[: args.limit]
    logger.info(f"{len(by_video) - len(pending)} videos done, {len(pending)} to align")
    if not pending:
        return
    aligner = Aligner(cfg)
    a = cfg.alignment
    for i, vid in enumerate(pending, 1):
        items, ids = [], []
        for t in by_video[vid]:
            wav, sr = read_audio(clips[t["id"]]["path"])
            items.append((resample(wav, sr, 16000), strip_tags(t["text"]).split()))
            ids.append(t["id"])
        order = np.argsort([len(w) for w, _ in items])
        results: list[dict | None] = [None] * len(items)
        sorted_res = aligner.align([items[k] for k in order])
        for k, r in zip(order, sorted_res, strict=True):
            results[k] = r
        rows = []
        for cid, (wav, words), res in zip(ids, items, results, strict=True):
            dur = len(wav) / 16000
            if res is None:
                rows.append({"id": cid, "words": [], "align_score": -99.0, "timed_ratio": 0.0, "edge_ok": False,
                             "align_ok": False})
                continue
            timed = [w for w in res["words"] if w["start"] is not None]
            ratio = len(timed) / max(1, len(words))
            edge_ok = bool(timed) and timed[0]["start"] >= a.min_edge_sec and timed[-1]["end"] <= dur - a.min_edge_sec
            rows.append({"id": cid, "words": res["words"], "align_score": res["align_score"],
                         "timed_ratio": round(ratio, 3), "edge_ok": edge_ok,
                         "align_ok": edge_ok and res["align_score"] >= a.min_score and ratio >= 0.9})
        write_jsonl(lay.aligned(vid), rows)
        ok = sum(r["align_ok"] for r in rows)
        logger.info(f"[{i}/{len(pending)}] {vid}: {ok}/{len(rows)} aligned OK")


if __name__ == "__main__":
    step_main(main)
