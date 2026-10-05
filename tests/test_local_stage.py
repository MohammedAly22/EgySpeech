"""Local stage helpers on synthetic audio: decoding, clip reading, index, edge silence, scheduling, Hub shards."""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 24000


def tone(sec: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_decode_audio_grows_past_wrong_duration_hint(tmp_path):
    from egyspeech.io import decode_audio

    x = tone(7.0)
    sf.write(tmp_path / "a.flac", x, SR)
    y = decode_audio(tmp_path / "a.flac", 16000, expected_sec=1.0)  # hint far too small
    assert abs(len(y) - 7 * 16000) < 50
    z = decode_audio(tmp_path / "a.flac", 16000, expected_sec=7.0)
    assert len(z) == len(y)


def test_clip_reader_matches_full_read(tmp_path):
    from egyspeech.io import ClipReader

    x = np.random.default_rng(0).standard_normal(SR * 10).astype(np.float32) * 0.1
    sf.write(tmp_path / "a.flac", x, SR, subtype="PCM_16")
    full, _ = sf.read(tmp_path / "a.flac", dtype="float32")
    with ClipReader(tmp_path / "a.flac", SR) as r:
        seg = r.read(2.5, 4.0)
    assert len(seg) == int(1.5 * SR)
    np.testing.assert_allclose(seg, full[int(2.5 * SR) : int(4.0 * SR)], atol=1e-6)


def test_index_parses_names_and_skips_temp_and_duplicates(tmp_path):
    from egyspeech.config import Section
    from egyspeech.io import read_jsonl
    from egyspeech.steps import index

    root = tmp_path / "raw"
    (root / "NA").mkdir(parents=True)
    (root / "My Playlist").mkdir()
    sf.write(root / "NA" / "Talk ｜ part 1 [-yCJX20_B50].flac", tone(90), SR)
    sf.write(root / "My Playlist" / "Episode 2 [B1KrNGry4vU].flac", tone(120), SR)
    sf.write(root / "My Playlist" / "Episode 2 copy [B1KrNGry4vU].flac", tone(100), SR)  # duplicate id
    sf.write(root / "Episode 3 [abcdefghijk].temp.flac", tone(90), SR)  # still converting
    sf.write(root / "short [zzzzzzzzzzz].flac", tone(5), SR)  # under min_video_sec
    cfg = Section({"work_dir": str(tmp_path / "work"), "download": {"dir": str(root), "min_video_sec": 60},
                   "index": {"extensions": ["flac"], "min_age_sec": 0}})
    index.main(cfg, argparse.Namespace(force=False, limit=None))
    rows = {r["id"]: r for r in read_jsonl(tmp_path / "work" / "meta" / "videos.jsonl")}
    assert set(rows) == {"-yCJX20_B50", "B1KrNGry4vU"}
    assert rows["-yCJX20_B50"]["playlist"] is None and rows["-yCJX20_B50"]["title"] == "Talk | part 1"
    assert rows["B1KrNGry4vU"]["playlist"] == "My Playlist"
    assert abs(rows["-yCJX20_B50"]["duration"] - 90) < 0.01
    assert rows["B1KrNGry4vU"]["url"].endswith("v=B1KrNGry4vU")


def test_edge_levels_detect_cut_inside_speech():
    from egyspeech.steps.segment import edge_levels

    quiet = np.zeros(int(0.15 * SR), dtype=np.float32) + 1e-4
    clean = np.concatenate([quiet, tone(5), quiet])
    cut = tone(5)
    s, e = edge_levels(clean, SR)
    assert s < -40 and e < -40
    s, e = edge_levels(cut, SR)
    assert s > -3 and e > -3


def test_run_with_budget_respects_memory():
    from egyspeech.parallel import run_with_budget

    running, peak, lock = [0.0], [0.0], threading.Lock()

    def job(cost):
        with lock:
            running[0] += cost
            peak[0] = max(peak[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= cost
        return cost

    costs = [1.0, 3.0, 0.5, 2.5, 1.0, 6.0, 0.5]  # 6.0 alone exceeds the budget: still runs, alone
    jobs = [(i, c, (c,)) for i, c in enumerate(costs)]
    with ThreadPoolExecutor(8) as pool:
        done = sorted(k for k, _ in run_with_budget(pool, job, jobs, max_jobs=8, budget=4.0))
    assert done == list(range(len(costs)))
    assert peak[0] <= 6.0 + 1e-9


def test_hub_shard_roundtrip(tmp_path):
    pytest.importorskip("datasets")
    from egyspeech.config import Section
    from egyspeech.io import read_jsonl
    from egyspeech.steps.pull_chunks import unpack
    from egyspeech.steps.push_chunks import chunk_features, write_shard

    features = chunk_features(SR)
    rows = []
    for k in range(3):
        p = tmp_path / f"c{k}.flac"
        sf.write(p, tone(6 + k), SR, subtype="PCM_16")
        row = {name: None for name in features}
        row.update(id=f"vidABCDEFGH_s0_{k:04d}", video_id="vidABCDEFGH", local_speaker=0, speaker_id="MALE_00001",
                   gender="male", start=10.0 * k,
                   end=10.0 * k + 6 + k, duration=6.0 + k, source="original", start_cut="pause", end_cut="turn",
                   weak_cut=False, edge_start_db=-50.0, edge_end_db=-48.0, dnsmos_p808=3.5, dnsmos_sig=3.6,
                   dnsmos_bak=4.0, dnsmos_ovrl=3.3, utmos=3.4, window_similarity=0.8, video_title="t",
                   speaker_embedding=list(np.linspace(-1, 1, 192, dtype=np.float32)),
                   audio={"bytes": Path(p).read_bytes(), "path": p.name})
        rows.append(row)
    shard = tmp_path / "train-00000.parquet"
    write_shard(rows, features, shard)
    from datasets import Dataset  # the Hub / datasets see the declared features (Audio, List, ...)

    ds = Dataset.from_parquet(str(shard))
    assert type(ds.features["audio"]).__name__ == "Audio" and len(ds) == 3

    cfg = Section({"work_dir": str(tmp_path / "work"), "segmentation": {"audio_format": "flac"}})
    n, sec, spk = unpack(shard, cfg)
    assert n == 3 and abs(sec - 21.0) < 1e-6 and spk[rows[0]["id"]] == ("MALE_00001", "male")
    work = tmp_path / "work"
    meta = read_jsonl(work / "meta" / "chunks" / "vidABCDEFGH.jsonl")
    assert [m["id"] for m in meta] == [r["id"] for r in rows]
    assert "dnsmos_ovrl" not in meta[0] and "speaker_embedding" not in meta[0] and "speaker_id" not in meta[0]
    wav, sr = sf.read(meta[1]["path"], dtype="float32")
    assert sr == SR and abs(len(wav) - 7 * SR) <= 1
    z = np.load(work / "meta" / "speakers" / "vidABCDEFGH.npz")
    assert z["embeddings"].shape == (3, 192) and z["window_similarity"][0] == pytest.approx(0.8)
    assert read_jsonl(work / "meta" / "quality" / "vidABCDEFGH.jsonl")[2]["utmos"] == pytest.approx(3.4)
