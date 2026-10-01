"""Cut planner: synthetic frame signals with known speakers and pauses."""

import numpy as np

from egyspeech.segmenter import HOP, SegParams, plan_clips


def frames(sec: float) -> int:
    return int(round(sec / HOP))


def build(timeline: list[tuple[float, str]], n_spk: int = 2):
    """timeline: (duration, what) with what in {"s0", "s1", "pause", "overlap"}."""
    n = frames(sum(d for d, _ in timeline))
    diar = np.zeros((n, n_spk), dtype=np.float32)
    vad = np.zeros(n, dtype=np.float32)
    db = np.full(n, -60.0, dtype=np.float32)
    t = 0
    for dur, what in timeline:
        k = frames(dur)
        if what.startswith("s"):
            diar[t : t + k, int(what[1])] = 0.95
            vad[t : t + k] = 0.95
            db[t : t + k] = -20.0
        elif what == "overlap":
            diar[t : t + k, :] = 0.9
            vad[t : t + k] = 0.95
            db[t : t + k] = -18.0
        t += k
    return diar, vad, db


def speech_with_pauses(total: float, every: float, pause: float, spk: str = "s0"):
    out, acc = [], 0.0
    while acc < total:
        seg = min(every, total - acc)
        out.append((seg, spk))
        acc += seg
        if acc < total:
            out.append((pause, "pause"))
            acc += pause
    return out


def test_short_turn_is_one_clip():
    diar, vad, db = build([(1.0, "pause"), (12.0, "s0"), (1.0, "pause")])
    clips = plan_clips(diar, vad, db, SegParams())
    assert len(clips) == 1
    c = clips[0]
    assert c.speaker == 0
    assert 12.0 <= c.duration <= 12.0 + 2 * 0.15 + 1e-6


def test_long_monologue_cut_at_pauses_only():
    timeline = [(1.0, "pause")] + speech_with_pauses(120, every=4.0, pause=0.5) + [(1.0, "pause")]
    diar, vad, db = build(timeline)
    p = SegParams()
    clips = plan_clips(diar, vad, db, p)
    assert clips
    speech = vad > p.vad_threshold
    for c in clips:
        assert p.min_sec <= c.duration <= p.max_sec
        assert c.start_cut in ("pause", "turn") and c.end_cut in ("pause", "turn")
        # boundaries are silent: never inside a word
        assert not speech[frames(c.start)]
        assert not speech[min(frames(c.end), len(speech) - 1)]
    covered = sum(c.duration for c in clips)
    assert covered > 0.85 * 120  # almost everything is kept


def test_no_clip_crosses_speakers_or_overlap():
    timeline = [(0.5, "pause"), (10, "s0"), (0.4, "pause"), (8, "s1"), (0.5, "pause"), (2, "overlap"), (9, "s0"),
                (0.5, "pause")]
    diar, vad, db = build(timeline)
    p = SegParams()
    clips = plan_clips(diar, vad, db, p)
    assert {c.speaker for c in clips} == {0, 1}
    for c in clips:
        a, b = frames(c.start), frames(c.end)
        other = 1 - c.speaker
        assert (diar[a:b, other] < p.other_speaker_threshold).all(), "clip contains the other speaker"


def test_guard_distance_from_other_speaker():
    timeline = [(0.5, "pause"), (10, "s0"), (0.1, "pause"), (10, "s1"), (0.5, "pause")]
    diar, vad, db = build(timeline)
    p = SegParams(guard_sec=0.3)
    for c in plan_clips(diar, vad, db, p):
        other = 1 - c.speaker
        idx = np.flatnonzero(diar[:, other] > p.other_speaker_threshold)
        gap = min(abs(frames(c.start) - idx).min(), abs(frames(c.end) - idx).min()) * HOP
        assert gap >= p.guard_sec - 0.011


def test_long_silence_splits_turn():
    timeline = [(0.5, "pause"), (8, "s0"), (3.0, "pause"), (8, "s0"), (0.5, "pause")]
    diar, vad, db = build(timeline)
    clips = plan_clips(diar, vad, db, SegParams(max_internal_silence_sec=1.5))
    assert len(clips) == 2
    assert all(c.max_internal_silence < 1.5 for c in clips)


def test_turn_without_pauses_uses_weak_cuts_or_is_dropped():
    diar, vad, db = build([(0.5, "pause"), (70, "s0"), (0.5, "pause")])
    # energy dips between "words" but VAD never drops: no real pause
    db[frames(0.5) : frames(70.5) : 40] = -40.0
    clips = plan_clips(diar, vad, db, SegParams(allow_weak_cuts=True))
    for c in clips:
        assert 5.0 <= c.duration <= 30.0
    assert any(c.start_cut == "weak" or c.end_cut == "weak" for c in clips)
    assert plan_clips(diar, vad, db, SegParams(allow_weak_cuts=False)) == [] or all(
        c.start_cut != "weak" and c.end_cut != "weak" for c in plan_clips(diar, vad, db, SegParams(allow_weak_cuts=False))
    )


def test_activity_dips_between_words_do_not_fragment_turns():
    timeline = [(0.5, "pause")] + speech_with_pauses(25, every=3.0, pause=0.3) + [(0.5, "pause")]
    diar, vad, db = build(timeline)
    diar[vad < 0.5, 0] = 0.35  # between words the diarizer is less sure, nobody else talks
    clips = plan_clips(diar, vad, db, SegParams())
    assert len(clips) == 1 and clips[0].duration > 24


def test_too_short_turns_dropped():
    diar, vad, db = build([(0.5, "pause"), (3.0, "s0"), (0.5, "pause")])
    assert plan_clips(diar, vad, db, SegParams()) == []


def test_interrupted_turn_never_ends_inside_speech():
    # s0 speaks with pauses, then s1 cuts in with no pause: the tail of s0's turn is cut by
    # s1's guard zone while s0 is still talking -> s0's clips must end at an earlier pause
    timeline = [(0.5, "pause")] + speech_with_pauses(20, every=3.0, pause=0.4) + [(6, "s1"), (0.5, "pause")]
    diar, vad, db = build(timeline)
    p = SegParams()
    clips = plan_clips(diar, vad, db, p)
    speech = vad > p.vad_threshold
    s0 = [c for c in clips if c.speaker == 0]
    assert s0, "the clean part of the turn is kept"
    for c in clips:
        assert not speech[frames(c.start)], "starts inside speech"
        assert not speech[min(frames(c.end), len(speech) - 1)], "ends inside speech"
    assert max(c.end for c in s0) <= 20.5 - 0.3, "s0 clip reaches into the interruption"
