"""Step 14 — analysis: statistics + figures into graphs/ (html, and png when possible)."""

import logging

from egyspeech import analysis
from egyspeech.config import Section
from egyspeech.io import write_json
from egyspeech.progress import StepBar
from egyspeech.steps import layout, step_main

logger = logging.getLogger("analysis")


def main(cfg: Section, args):
    lay = layout(cfg)
    with StepBar("analysis", 2, unit="phases") as bar:
        bar.status("computing statistics and figures")
        figs, stats = analysis.build(cfg)
        bar.advance()
        bar.status(f"saving {len(figs)} figures")
        warnings = analysis.save(figs, lay.graphs, png=cfg.analysis.save_png)
        bar.advance()
    for w in warnings:
        logger.warning(f"png export skipped ({w}); html figures are saved")
    write_json(lay.stats, stats)
    logger.info(f"{len(figs)} figures -> {lay.graphs} | {stats['hours']} h, {stats['speakers']} speakers, "
                f"{stats['clips']} clips")


if __name__ == "__main__":
    step_main(main)
