"""Terminal output of every step: rich logging plus one progress bar per step.

The bar counts videos and, when durations are known, hours of audio, so the
speed (x real time) and the ETA stay meaningful even when episode lengths vary a lot.
Log lines are printed above the bar.

    with StepBar("diarize", n_items=len(pending), audio_sec=total_sec) as bar:
        for vid in pending:
            bar.status(vid)
            ...
            bar.advance(audio_sec=duration)
"""

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (BarColumn, Progress, ProgressColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.text import Text

console = Console(highlight=False)

QUIET_LOGGERS = ("urllib3", "httpx", "httpcore", "huggingface_hub", "filelock", "numba", "matplotlib", "datasets",
                 "fsspec", "asyncio", "PIL")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s", datefmt="%H:%M:%S", force=True,
                        handlers=[RichHandler(console=console, show_path=False, markup=False, rich_tracebacks=True,
                                              log_time_format="%H:%M:%S")])
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)  # NeMo telemetry notices


class _RealTime(ProgressColumn):
    """Hours of audio processed per hour of wall-clock time."""

    def render(self, task) -> Text:
        if not task.fields.get("audio") or not task.elapsed or task.elapsed < 1 or not task.completed:
            return Text("")
        return Text(f"{task.completed / task.elapsed:,.0f}x RT", style="magenta")


class StepBar:
    def __init__(self, name: str, n_items: int, audio_sec: float | None = None, unit: str = "videos"):
        self.audio = bool(audio_sec)
        self.done = 0
        self.hours = 0.0
        self.total_hours = (audio_sec or 0.0) / 3600
        columns = [TextColumn(f"[bold cyan]{name}"), BarColumn(bar_width=30), TaskProgressColumn(),
                   TextColumn("{task.fields[done]}/{task.fields[n]} " + unit)]
        if self.audio:
            columns += [TextColumn("• {task.fields[hours]:.1f}/{task.fields[total_hours]:.1f} h audio"), _RealTime()]
        columns += [TextColumn("•"), TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(),
                    TextColumn("[dim]{task.fields[status]}")]
        self.progress = Progress(*columns, console=console, speed_estimate_period=3600)
        self.task = self.progress.add_task(
            name, total=max(float(audio_sec) if self.audio else float(n_items), 1e-9), done=0, n=n_items,
            hours=0.0, total_hours=self.total_hours, audio=self.audio, status="")

    def start(self) -> "StepBar":
        self.progress.start()
        return self

    def close(self) -> None:
        self.status("")
        self.progress.stop()

    def __enter__(self) -> "StepBar":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def advance(self, items: int = 1, audio_sec: float = 0.0) -> None:
        self.done += items
        self.hours += audio_sec / 3600
        self.progress.update(self.task, advance=audio_sec if self.audio else items, done=self.done,
                             hours=self.hours)

    def status(self, text: str) -> None:
        self.progress.update(self.task, status=text[:60])
