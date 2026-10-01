"""Two-stage filter (hard limits + weighted risk) on clips labeled by listening in the pilot."""

from pathlib import Path

import pytest

from egyspeech.config import load_config
from egyspeech.steps.filter import reasons, risk

CFG = load_config(Path(__file__).resolve().parent.parent / "configs" / "config.yaml")

BASE = {"duration": 10.0, "clip_ratio": 0.0, "speech_ratio": 0.85, "weak_cut": False}
# (ovrl, sig, bak, utmos, voice similarity, edge start dB, edge end dB, listener's verdict)
LABELED = [
    (3.118, 3.532, 3.748, 2.825, 0.941, -22.2, -20.3, "keep"),     # loud-ish edges, otherwise clean
    (3.296, 3.523, 4.173, 2.645, 0.807, -8.2, -76.0, "keep"),      # word starts right at the edge
    (3.229, 3.533, 3.949, 3.094, 0.841, -0.8, -18.3, "keep"),
    (2.452, 3.209, 2.945, 3.113, 0.739, -62.4, -67.3, "keep"),     # low OVRL, slight background
    (2.695, 3.039, 3.741, 2.959, 0.936, -50.0, -48.8, "keep"),
    (2.225, 3.103, 2.734, 2.262, 0.876, -39.9, -22.2, "reject"),   # interrupted inside a word
    (2.687, 3.625, 2.704, 2.066, 0.871, -27.3, -24.9, "reject"),   # sound effects
    (2.454, 3.415, 2.495, 1.505, 0.822, -53.0, -33.9, "reject"),   # background music
]


@pytest.mark.parametrize("ovrl,sig,bak,utmos,sim,es,ee,verdict", LABELED)
def test_labeled_clips(ovrl, sig, bak, utmos, sim, es, ee, verdict):
    r = {**BASE, "dnsmos_ovrl": ovrl, "dnsmos_sig": sig, "dnsmos_bak": bak, "utmos": utmos,
         "window_similarity": sim, "edge_start_db": es, "edge_end_db": ee}
    why = reasons(r, CFG.filter, CFG.segmentation)
    assert (not why) == (verdict == "keep"), (why, risk(r, CFG.filter))


def test_one_borderline_score_passes_several_reject():
    clean = {**BASE, "dnsmos_ovrl": 3.2, "dnsmos_sig": 3.5, "dnsmos_bak": 3.9, "utmos": 2.9,
             "window_similarity": 0.9, "edge_start_db": -50.0, "edge_end_db": -50.0}
    assert not reasons({**clean, "edge_end_db": -5.0}, CFG.filter, CFG.segmentation)
    assert reasons({**clean, "edge_end_db": -5.0, "dnsmos_bak": 3.0, "utmos": 2.2}, CFG.filter,
                   CFG.segmentation) == ["combined_risk"]
    assert "multi_speaker" in reasons({**clean, "window_similarity": 0.6}, CFG.filter, CFG.segmentation)
