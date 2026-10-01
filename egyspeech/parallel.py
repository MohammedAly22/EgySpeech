"""Parallel helpers: bounded prefetching and a RAM-aware process pool scheduler."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from typing import Any


def prefetch(executor: Executor, fn: Callable, items: Iterable, depth: int = 2) -> Iterator[tuple[Any, Any]]:
    """Yield (item, fn(item)) in order, computing at most `depth` items ahead (bounded memory)."""
    queue: deque[tuple[Any, Future]] = deque()
    it = iter(items)
    for item in it:
        queue.append((item, executor.submit(fn, item)))
        if len(queue) >= depth:
            break
    while queue:
        item, fut = queue.popleft()
        nxt = next(it, None)
        if nxt is not None:
            queue.append((nxt, executor.submit(fn, nxt)))
        yield item, fut.result()


def run_with_budget(executor: Executor, fn: Callable, jobs: list[tuple[Any, float, tuple]], max_jobs: int,
                    budget: float) -> Iterator[tuple[Any, Future]]:
    """Run fn(*args) for (key, cost, args) jobs, keeping the summed cost of running jobs <= budget.

    A job larger than the budget still runs, alone. Yields (key, finished future) as jobs complete.
    """
    running: dict[Future, tuple[Any, float]] = {}
    used = 0.0
    queue = deque(jobs)
    while queue or running:
        while queue and len(running) < max_jobs and (not running or used + queue[0][1] <= budget):
            key, cost, args = queue.popleft()
            running[executor.submit(fn, *args)] = (key, cost)
            used += cost
        done, _ = wait(list(running), return_when=FIRST_COMPLETED)
        for fut in done:
            key, cost = running.pop(fut)
            used -= cost
            yield key, fut
