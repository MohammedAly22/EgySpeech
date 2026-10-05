"""v1 flow: diarization chunks, global speaker IDs, speaker cap."""

import numpy as np

from egyspeech.steps.cluster import assign_speakers, unit
from egyspeech.steps.filter import cap_speaker, speaker_stats
from egyspeech.steps.segment import diar_turns, split_points

F = 0.08  # Sortformer frame


def track(spec: list[tuple[float, int | None]], n_spk: int = 2) -> np.ndarray:
    """(seconds, speaker or None for silence, -1 for overlap) -> probabilities."""
    rows = []
    for sec, who in spec:
        p = np.zeros(n_spk, dtype=np.float32)
        if who == -1:
            p[:] = 0.9
        elif who is not None:
            p[who] = 0.9
        rows += [p] * int(round(sec / F))
    return np.array(rows)


def test_turns_follow_diarization_and_skip_overlap():
    p = track([(1, None), (10, 0), (0.4, None), (5, 0), (0.2, None), (8, 1), (2, -1), (6, 0), (1, None)])
    turns = diar_turns(p, F, min_sec=2.0, max_gap_sec=0.6, pad_sec=0.1)
    spk = [t[0] for t in turns]
    assert spk == [0, 1, 0]  # 0.4 s pause bridged; the overlap is left out
    s0, s1, s0b = turns
    assert abs(s0[1] - 0.88) < 0.01 and abs(s0[2] - 16.4) < 0.15  # 1 s silence - 1 frame of padding; 10 + 0.4 + 5 s
    assert s1[1] >= s0[2] - 1e-6  # padding never runs into the other speaker
    overlap_end = (np.flatnonzero((p > 0.5).sum(1) == 2).max() + 1) * F
    assert s0b[1] >= overlap_end - 1e-6  # starts after the overlap


def test_long_turn_split_at_quiet_moment():
    sr = 24000
    x = np.random.default_rng(0).standard_normal(sr * 50).astype(np.float32) * 0.1
    x[int(26.0 * sr) : int(26.3 * sr)] *= 0.001  # quiet spot near the middle
    cuts = split_points(x, sr, max_sec=30.0)
    assert len(cuts) == 1 and 26.0 <= cuts[0] / sr <= 26.3
    assert split_points(x[: sr * 20], sr, max_sec=30.0) == []


def test_speakers_matched_across_episodes():
    rng = np.random.default_rng(1)
    people = unit(rng.standard_normal((3, 192)))
    prints, secs, truth = [], [], []
    for ep in range(12):  # each episode: 2 of the 3 people
        for who in (ep % 3, (ep + 1) % 3):
            prints.append(unit(people[who] + 0.35 * unit(rng.standard_normal(192))))
            secs.append(300.0)
            truth.append(who)
    prints.append(unit(people[2] + 0.35 * unit(rng.standard_normal(192))))  # short voice print
    secs.append(5.0)
    truth.append(2)
    labels = assign_speakers(np.array(prints, dtype=np.float32), np.array(secs), 0.65, 30.0)
    assert len(set(labels)) == 3
    for t in range(3):  # every voice print of a person got the same label
        assert len({lab for lab, x in zip(labels, truth) if x == t}) == 1


def test_cap_keeps_every_speaker_and_spreads_over_episodes():
    rows = [{"id": f"a{i}", "video_id": f"v{i % 4}", "duration": 600.0, "speaker_id": "MALE_00001", "gender": "male",
             "dnsmos_bak": 3.0 + i / 100, "dnsmos_sig": 3.5, "window_similarity": 0.9} for i in range(40)]
    kept = cap_speaker(rows, 2 * 3600)
    assert abs(sum(r["duration"] for r in kept) - 7200) < 1
    assert {r["video_id"] for r in kept} == {"v0", "v1", "v2", "v3"}
    small = [{"id": "b", "video_id": "v9", "duration": 100.0, "speaker_id": "FEMALE_00002", "gender": "female"}]
    assert cap_speaker(small, 2 * 3600) == small
    st = speaker_stats(kept + small)
    assert st["speakers"] == 2 and st["gender"]["female"]["speakers"] == 1
