"""Running independent pieces of work on threads, or not.

Most of what the iteration loop does outside the FFT is a sweep over a large
array: zeroing the volume outside the support, gathering the measured points,
scattering the constraint back, clipping to positivity, copying the candidate
iterate.  All of them are single-threaded in NumPy and none of them saturates
memory bandwidth on one core -- measured at 1024**3, splitting them across
threads is worth 4.1x to 4.9x.

Every split here is over a disjoint range, so *which* thread does *what* does
not change the answer.  The one exception is a reduction: adding partial sums
in a different order moves the last bits.  That is why the error metric fixes
its partition count independently of the thread count -- see
``ERROR_PARTITIONS`` in :mod:`.reconstruct` -- so a serial run and a threaded
run agree exactly, and only the comparison with pre-partition results shifts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Sequence

import numpy as np

__all__ = ["Parallel", "split"]


def split(length: int, parts: int) -> List[slice]:
    """``length`` cut into at most ``parts`` contiguous, non-empty slices."""
    parts = max(1, min(int(parts), int(length)))
    edges = np.linspace(0, length, parts + 1).astype(np.int64)
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class Parallel:
    """A thread pool that is also happy to be one thread.

    ``threads=1`` runs the jobs inline -- no pool, no dispatch overhead -- so
    the serial path stays available and costs nothing.
    """

    def __init__(self, threads: int = 1) -> None:
        self.threads = max(1, int(threads))
        self._pool = (ThreadPoolExecutor(max_workers=self.threads)
                      if self.threads > 1 else None)

    @property
    def enabled(self) -> bool:
        return self._pool is not None

    def run(self, jobs: Iterable[Callable[[], None]]) -> None:
        """Run every job; return once all have finished."""
        if self._pool is None:
            for job in jobs:
                job()
            return
        # list() forces the map, which is what waits for the results
        list(self._pool.map(_call, jobs))

    def map_slices(self, length: int, fn: Callable[[slice], None],
                   parts: int = 0) -> None:
        """Apply ``fn`` to each of ``parts`` slices of ``range(length)``."""
        pieces = split(length, parts or self.threads)
        self.run([(lambda s=s: fn(s)) for s in pieces])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "Parallel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return "Parallel(threads=%d)" % self.threads


def _call(fn):
    return fn()
