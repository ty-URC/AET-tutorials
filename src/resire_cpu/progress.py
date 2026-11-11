"""Progress that survives the shell.

A 512**3 run takes over an hour, and the first one was launched with its
stdout piped through ``tail`` -- which buffers until EOF, so the iteration
counter was invisible for the whole run and the only way to guess how far it
had got was to read CPU times out of the process table.  Printing is not a
reliable channel for something you may need to inspect from elsewhere.

So progress also goes to a file, one JSON object per line, flushed every time.
``tail -1 run.jsonl`` then answers "where is it" no matter how the process was
launched, and the file is a record of the run afterwards.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional, TextIO

__all__ = ["Progress"]


class Progress:
    """Reports iteration progress to a stream, a JSON-lines file, or both.

    Parameters
    ----------
    total : number of iterations expected, used for the ETA.
    path : file to append JSON lines to.  Truncated at construction.
    every : report every this many iterations.  The first and last always
        report, so a run is never silent.
    stream : where to print; ``None`` to write only the file.
    label : prefix for the printed lines.
    """

    def __init__(self, total: int, path: Optional[str] = None, every: int = 25,
                 stream: Optional[TextIO] = sys.stdout, label: str = "") -> None:
        self.total = int(total)
        self.every = max(0, int(every))
        self.stream = stream
        self.label = label
        self.start = time.perf_counter()
        self._fh: Optional[TextIO] = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                        exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def _due(self, it: int) -> bool:
        return (it == 1 or it == self.total
                or (self.every and it % self.every == 0))

    def update(self, it: int, **fields: Any) -> None:
        """Record iteration ``it`` (1-based) with arbitrary extra fields."""
        if not self._due(it):
            return
        elapsed = time.perf_counter() - self.start
        rec: Dict[str, Any] = {
            "iter": it, "total": self.total,
            "elapsed_s": round(elapsed, 2),
            # rate over the whole run so far, not the last interval: it is
            # what the remaining time should be extrapolated from
            "eta_s": round(elapsed * (self.total - it) / it, 1) if it else None,
        }
        rec.update({k: (round(float(v), 8) if isinstance(v, float) else v)
                    for k, v in fields.items()})
        if self._fh is not None:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        if self.stream is not None:
            extra = "  ".join("%s %.5f" % (k, v) if isinstance(v, float)
                              else "%s %s" % (k, v) for k, v in fields.items())
            self.stream.write(
                "%s  iter %5d/%d  %s  elapsed %s  eta %s\n"
                % (self.label, it, self.total, extra,
                   _hms(elapsed), _hms(rec["eta_s"] or 0)))
            self.stream.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)
