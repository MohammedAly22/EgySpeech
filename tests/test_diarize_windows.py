"""Long-episode diarization: per-window speaker labels are matched across overlaps."""

import numpy as np

from egyspeech.steps.diarize import best_permutation, segments_from_probs, stitch


def test_stitch_restores_speaker_identity_across_windows():
    rng = np.random.default_rng(0)
    T, S = 3000, 4
    truth = np.zeros((T, S), dtype=np.float32)
    turn = rng.integers(0, 3, size=T // 50).repeat(50)  # 3 speakers taking turns of 4 s
    truth[np.arange(T), turn] = 0.95
    win, step = 1000, 800  # frames; 200-frame overlaps
    windows, start = [], 0
    while True:
        part = truth[start : start + win]
        perm = rng.permutation(S)  # every window names the speakers differently
        windows.append((start, part[:, perm]))
        if start + win >= T:
            break
        start += step
    first_perm = best_permutation(truth[:win], windows[0][1])
    joined = stitch(windows)[:, first_perm]
    assert joined.shape == truth.shape
    assert np.array_equal(joined.argmax(1), truth.argmax(1))


def test_segments_from_probs():
    p = np.zeros((100, 2), dtype=np.float32)
    p[10:40, 0] = 0.9
    p[50:52, 1] = 0.9  # 0.16 s: too short
    assert segments_from_probs(p) == [[0.8, 3.2, "speaker_0"]]
