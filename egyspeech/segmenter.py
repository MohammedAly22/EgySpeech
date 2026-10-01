"""Single-speaker, pause-aware segmentation (pure numpy; unit-tested).

Inputs are frame-level signals on a common 10 ms grid:
    diar[t, s]  speaker activity probability (Sortformer)
    vad[t]      speech probability (Silero VAD)
    db[t]       energy in dB (vocal stem)

1. Turns. For each speaker, keep frames where that speaker (and nobody else) is
   active, or where nobody speaks at all, and stay `guard_sec` away from any other
   speaker's activity and from speech the diarizer did not attribute. Turns are
   split at silences longer than `max_internal_silence_sec`.
2. Cut points. Pauses (VAD below `vad_off_threshold` for >= `min_pause_sec`) are
   candidate cuts, scored by their length. If a turn has no usable pause, the
   deepest energy dips between words are added as weak candidates.
3. Plan. A dynamic program chooses cuts so that every kept clip is between
   `min_sec` and `max_sec`, preferring clips near `target_sec`, strong pauses, and
   keeping as much speech as possible; stretches that cannot form a valid clip are
   discarded rather than cut badly.
4. Trim. Each clip is trimmed to its speech plus `edge_pad_sec` of silence.

A turn edge only counts as a boundary when there is silence at it; if the speaker is
still talking where another voice's guard zone begins, the clip ends at the last
pause before that instead, so no clip starts or ends inside a word.
"""

from dataclasses import dataclass, field

import numpy as np

HOP = 0.01  # seconds per frame of the common grid


@dataclass
class SegParams:
    min_sec: float = 5.0
    max_sec: float = 30.0
    target_sec: float = 14.0
    speaker_threshold: float = 0.5
    other_speaker_threshold: float = 0.2
    vad_threshold: float = 0.5
    vad_off_threshold: float = 0.35
    min_pause_sec: float = 0.12
    strong_pause_sec: float = 0.35
    allow_weak_cuts: bool = True
    max_internal_silence_sec: float = 1.5
    guard_sec: float = 0.3
    edge_pad_sec: float = 0.15
    edge_silence_sec: float = 0.08  # a turn edge is a valid cut only with this much silence at it
    silence_db: float = 25.0  # frames this far below the median speech energy are silence, whatever VAD says
    length_weight: float = 2.0
    pause_weight: float = 1.0
    weak_cut_penalty: float = 1.5


@dataclass
class Clip:
    speaker: int
    start: float
    end: float
    start_cut: str  # "turn" | "pause" | "weak"
    end_cut: str
    start_pause: float  # length of the pause (s) at the start boundary
    end_pause: float
    speech_ratio: float
    max_internal_silence: float
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def to_grid(values: np.ndarray, frame_sec: float, n: int) -> np.ndarray:
    """Nearest-frame resampling of a [T, ...] signal onto n frames of HOP seconds."""
    idx = np.minimum((np.arange(n) * HOP / frame_sec).astype(np.int64), len(values) - 1)
    return values[idx]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.int32)
    return np.convolve(mask.astype(np.int32), kernel, mode="same") > 0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of True runs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def silence_masks(vad: np.ndarray, db: np.ndarray, p: SegParams) -> tuple[np.ndarray, np.ndarray]:
    """(speech, pause) frame masks from VAD and energy.

    Silero VAD smooths over short gaps (its probability stays high for ~0.2 s after a
    word), so the quick hand-overs of a conversation look like continuous speech. A
    frame whose energy is `silence_db` below the recording's median speech level is
    silence whatever VAD says.
    """
    voiced = vad > p.vad_threshold
    level = float(np.median(db[voiced])) if voiced.any() else float(db.max(initial=0.0))
    quiet = db < level - p.silence_db
    return voiced & ~quiet, (vad < p.vad_off_threshold) | quiet


def speaker_turns(diar: np.ndarray, speech: np.ndarray, p: SegParams) -> list[tuple[int, int, int]]:
    """(speaker, start frame, end frame) turns with only that speaker (or silence)."""
    n, S = diar.shape
    active = diar > p.speaker_threshold
    present = diar > p.other_speaker_threshold
    nobody = ~present.any(axis=1)
    unattributed_speech = speech & nobody
    guard = int(round(p.guard_sec / HOP))
    turns = []
    for s in range(S):
        others_present = np.delete(present, s, axis=1).any(axis=1)
        own = active[:, s] & ~others_present
        if not own.any():
            continue
        # Continuity only needs this speaker to be the one present: the activity often dips
        # between words (0.2 < p < 0.5) without anybody else talking.
        own_present = present[:, s] & ~others_present
        blocked = _dilate(others_present | unattributed_speech, guard)
        allowed = (own_present | (nobody & ~speech)) & ~blocked
        for a, b in _runs(allowed):
            speech_s = own[a:b] & speech[a:b]
            if not speech_s.any():
                continue
            # split at long silences inside the run
            idx = np.flatnonzero(speech_s) + a
            gaps = np.flatnonzero(np.diff(idx) * HOP > p.max_internal_silence_sec)
            starts = np.concatenate([[idx[0]], idx[gaps + 1]])
            ends = np.concatenate([idx[gaps], [idx[-1]]]) + 1
            for st, en in zip(starts, ends, strict=True):
                # keep the surrounding silence available for padding
                lo = max(a, st - int(round(p.edge_pad_sec / HOP)) * 2)
                hi = min(b, en + int(round(p.edge_pad_sec / HOP)) * 2)
                turns.append((s, int(lo), int(hi)))
    return turns


def _pause_candidates(pause_mask: np.ndarray, db: np.ndarray, a: int, b: int, p: SegParams):
    """(frame, quality, pause_len_sec, kind) cut candidates strictly inside (a, b)."""
    pause = pause_mask[a:b]
    cands = []
    for s, e in _runs(pause):
        if s == 0 or e == b - a:
            continue  # touches the turn edge: that is the turn boundary, not an interior cut
        length = (e - s) * HOP
        if length < p.min_pause_sec:
            continue
        seg = db[a + s : a + e]
        center = a + s + int(np.argmin(seg)) if length < 2 * p.strong_pause_sec else a + (s + e) // 2
        quality = min(length, p.strong_pause_sec) / p.strong_pause_sec
        cands.append((center, quality, length, "pause"))
    return cands


def _weak_candidates(vad: np.ndarray, db: np.ndarray, a: int, b: int, taken: list[int]):
    """Deepest energy dips between words (local minima over +-150 ms, low VAD)."""
    radius = 15
    seg_db = db[a:b]
    if len(seg_db) < 2 * radius + 1:
        return []
    padded = np.pad(seg_db, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    is_min = seg_db <= windows.min(axis=1) + 1e-6
    low = seg_db <= np.percentile(seg_db, 25)
    lowvad = vad[a:b] <= np.percentile(vad[a:b], 30)
    idx = np.flatnonzero(is_min & low & lowvad)
    cands = []
    last = -10**9
    for i in idx:
        f = a + int(i)
        if f - last < 100 or any(abs(f - t) < 100 for t in taken):
            continue  # keep weak cuts >= 1 s apart and away from real pauses
        cands.append((f, -1.0, 0.0, "weak"))
        last = f
    return cands


def _node_bonus(node: tuple[int, float, float, str], p: SegParams) -> float:
    kind = node[3]
    if kind == "pause":
        return p.pause_weight * node[1]
    if kind == "weak":
        return -p.weak_cut_penalty
    return 0.0  # turn edge: silence / speaker change, always a clean boundary


def _plan(nodes: list[tuple[int, float, float, str]], p: SegParams) -> list[tuple[int, int]]:
    """Dynamic program over sorted boundary nodes; returns the kept (i, j) node pairs.

    best[j] = best score of a plan for [node 0, node j] with a boundary at j.
      keep i->j   : min_sec <= length <= max_sec (+ padding that trimming removes);
                    gain = kept seconds - length penalty + boundary quality at i and j
      discard i->j: gain 0 (the stretch is dropped), always allowed
    """
    n = len(nodes)
    pos = np.array([nd[0] for nd in nodes], dtype=np.float64) * HOP
    bonus = np.array([_node_bonus(nd, p) for nd in nodes])
    max_len = p.max_sec + 2 * p.edge_pad_sec
    best = np.full(n, -np.inf)
    back = [(-1, False)] * n
    best[0] = 0.0
    prefix_arg = 0  # argmax of best[0..j-1] (for discard transitions)
    for j in range(1, n):
        if best[j - 1] > best[prefix_arg]:
            prefix_arg = j - 1
        best[j], back[j] = best[prefix_arg], (prefix_arg, False)
        i = j - 1
        while i >= 0 and pos[j] - pos[i] <= max_len:
            d = pos[j] - pos[i]
            if d >= p.min_sec and best[i] > -np.inf:
                gain = d - p.length_weight * ((d - p.target_sec) / p.target_sec) ** 2 + bonus[i] + bonus[j]
                if best[i] + gain > best[j]:
                    best[j], back[j] = best[i] + gain, (i, True)
            i -= 1
    kept = []
    j = n - 1
    while j > 0:
        i, keep = back[j]
        if keep:
            kept.append((i, j))
        j = i
    return list(reversed(kept))


def plan_clips(diar: np.ndarray, vad: np.ndarray, db: np.ndarray, p: SegParams) -> list[Clip]:
    """Clips (in seconds) for one recording; all inputs on the HOP grid, same length."""
    speech, pause_mask = silence_masks(vad, db, p)
    pad = int(round(p.edge_pad_sec / HOP))
    quiet = max(1, int(round(p.edge_silence_sec / HOP)))
    clips: list[Clip] = []
    for spk, a, b in speaker_turns(diar, speech, p):
        if (b - a) * HOP < p.min_sec:
            continue
        # A turn edge where the speaker is still talking (cut short by another voice's guard
        # zone) is not a clean boundary: the clip must then start / end at a pause inside.
        head = [(a, 1.0, 0.0, "turn")] if pause_mask[a : a + quiet].all() else []
        tail = [(b, 1.0, 0.0, "turn")] if pause_mask[b - quiet : b].all() else []
        cands = _pause_candidates(pause_mask, db, a, b, p)
        nodes = head + cands + tail
        if len(nodes) < 2:
            continue
        whole = head and tail and (b - a) * HOP <= p.max_sec
        kept = [(0, len(nodes) - 1)] if whole else _plan(nodes, p)
        if p.allow_weak_cuts and not whole:
            covered = sum(nodes[j][0] - nodes[i][0] for i, j in kept) * HOP
            if covered < 0.8 * (b - a) * HOP:
                weak = _weak_candidates(vad, db, a, b, [c[0] for c in cands])
                if weak:
                    nodes = head + sorted(cands + weak, key=lambda x: x[0]) + tail
                    kept = _plan(nodes, p)
        for i, j in kept:
            s_node, e_node = nodes[i], nodes[j]
            lo, hi = s_node[0], e_node[0]
            sp = np.flatnonzero(speech[lo:hi])
            if len(sp) == 0:
                continue
            first, last = lo + sp[0], lo + sp[-1] + 1
            start = max(lo, first - pad)
            end = min(hi, last + pad)
            dur = (end - start) * HOP
            if not p.min_sec <= dur <= p.max_sec:
                continue
            inner = ~speech[first:last]
            max_sil = max(((e - s) * HOP for s, e in _runs(inner)), default=0.0)
            clips.append(
                Clip(
                    speaker=int(spk),
                    start=round(start * HOP, 3),
                    end=round(end * HOP, 3),
                    start_cut=s_node[3],
                    end_cut=e_node[3],
                    start_pause=round(s_node[2], 3),
                    end_pause=round(e_node[2], 3),
                    speech_ratio=round(float(speech[start:end].mean()), 3),
                    max_internal_silence=round(float(max_sil), 3),
                )
            )
    clips.sort(key=lambda c: c.start)
    return clips
