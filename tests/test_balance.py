from egyspeech.steps.balance import cap_speaker


def clip(i: int, video: str, dur: float, utmos: float) -> dict:
    return {"id": f"c{i}", "video_id": video, "duration": dur, "utmos": utmos, "dnsmos_ovrl": 3.2,
            "align_score": -1.0, "weak_cut": False}


def test_cap_respects_budget_and_spreads_over_videos():
    rows = [clip(i, "v1", 10, 4.0) for i in range(50)] + [clip(100 + i, "v2", 10, 3.0) for i in range(5)]
    chosen = cap_speaker(rows, cap_sec=100)
    assert 100 <= sum(r["duration"] for r in chosen) < 110
    assert {r["video_id"] for r in chosen} == {"v1", "v2"}  # round-robin keeps diversity


def test_cap_prefers_best_quality_within_video():
    rows = [clip(i, "v1", 10, float(i)) for i in range(10)]
    chosen = cap_speaker(rows, cap_sec=30)
    assert [r["id"] for r in chosen] == ["c9", "c8", "c7"]


def test_small_speaker_kept_entirely():
    rows = [clip(i, "v1", 10, 3.0) for i in range(3)]
    assert len(cap_speaker(rows, cap_sec=1000)) == 3
