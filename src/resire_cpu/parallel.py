"""Column-block parallelism for the projection operator, over processes.

Threads cannot help here.  The work is a SciPy sparse matrix-matrix product,
and SciPy holds the GIL through it, so a thread pool would serialise on the
one thing worth parallelising.  Processes can -- the single-axis geometry
makes every y column of the volume an independent problem sharing one matrix.

The matrix is the reason this needs shared memory rather than plain argument
passing.  At 512**3 it is 1.5 GB; handing a copy to each of eight workers
would cost 12 GB and a pickle round trip per call.  Instead the operator's
buffers are allocated in shared memory once, each worker maps them and wraps
them in a CSR view without copying, and a call passes only a pair of column
indices.  ``shares_memory`` is asserted at start-up rather than assumed: if
SciPy ever decides to copy on construction the cost reappears silently, as
memory rather than as a wrong answer.

The volume and projection buffers live in the same shared memory, so the
caller can keep its iterate *in* the buffer and never copy it.  That matters
more than it sounds: a first version passed arrays in and out per call, and
the resulting serial copying was measured at 54% of a 512**3 run's wall clock
-- an Amdahl ceiling of 1.8x no matter how many workers were added.  The
gradient step and the positivity clip are done by the workers for the same
reason.
"""

from __future__ import annotations

import atexit
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = ["ParallelMatmul", "split"]

_STATE: dict = {}


def split(length: int, parts: int) -> List[Tuple[int, int]]:
    """Contiguous, near-equal ranges covering ``range(length)``."""
    parts = max(1, min(int(parts), length))
    edges = np.linspace(0, length, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _attach(spec: dict) -> None:
    """Worker start-up: map every shared block and rebuild the CSR views."""
    blocks, arrays = {}, {}
    for name, (shm_name, shape, dtype) in spec["blocks"].items():
        shm = SharedMemory(name=shm_name)
        blocks[name] = shm                       # keep alive for the process
        arrays[name] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)

    def view(prefix, shape):
        m = sp.csr_matrix((arrays[prefix + "_data"], arrays[prefix + "_indices"],
                           arrays[prefix + "_indptr"]), shape=shape, copy=False)
        if not np.shares_memory(m.data, arrays[prefix + "_data"]):
            raise RuntimeError(
                "SciPy copied %s into the worker; the shared matrix is not "
                "being shared and memory use will scale with worker count"
                % prefix)
        return m

    _STATE.clear()
    _STATE.update(
        blocks=blocks, arrays=arrays,
        A=view("a", spec["a_shape"]), AT=view("at", spec["at_shape"]),
        vol=arrays["vol"], proj=arrays["proj"])


def _forward(rng: Tuple[int, int]) -> None:
    lo, hi = rng
    _STATE["proj"][:, lo:hi] = _STATE["A"] @ _STATE["vol"][:, lo:hi]


def _backward(rng: Tuple[int, int]) -> None:
    lo, hi = rng
    _STATE["vol"][:, lo:hi] = _STATE["AT"] @ _STATE["proj"][:, lo:hi]


def _descend(args) -> None:
    """One whole gradient step on this worker's columns.

    Fused on purpose.  Done in the parent this is a back-projection, a scaling
    temporary, a subtraction and a clip over the full volume -- four passes
    over 537 MB at 512**3, none of them parallel.
    """
    lo, hi, dt, positivity, thresh = args
    block = _STATE["vol"][:, lo:hi]
    block -= dt * (_STATE["AT"] @ _STATE["proj"][:, lo:hi])
    if thresh:
        block -= thresh          # with the clip below this is max(x - t, 0),
    if positivity:               # the proximal operator of an L1 penalty on
        np.maximum(block, 0, out=block)   # the non-negative orthant


class ParallelMatmul:
    """Holds ``A``, ``A.T`` and the volume/projection buffers.

    ``workers <= 1`` keeps everything in this process and allocates plain
    arrays instead of shared ones, so the serial path stays available for
    comparison and for environments where spawning is unwanted.  Both paths
    expose the same buffers, so callers do not branch.
    """

    def __init__(self, A: sp.csr_matrix, AT: sp.csr_matrix, n_cols: int,
                 workers: int = 1) -> None:
        self.A, self.AT, self.n_cols = A, AT, n_cols
        self.workers = max(1, int(workers))
        self._shm: List[SharedMemory] = []
        self._pool = None
        self.vol_shape = (A.shape[1], n_cols)
        self.proj_shape = (A.shape[0], n_cols)
        if self.workers > 1:
            self._start()
            atexit.register(self.close)
        else:
            self.vol = np.zeros(self.vol_shape, dtype=A.dtype)
            self.proj = np.zeros(self.proj_shape, dtype=A.dtype)
            self._ranges = [(0, n_cols)]

    # -- setup ------------------------------------------------------------
    def _share(self, arr: np.ndarray) -> Tuple[str, tuple, str]:
        shm = SharedMemory(create=True, size=max(arr.nbytes, 1))
        self._shm.append(shm)
        out = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        out[...] = arr
        return shm.name, arr.shape, arr.dtype.str

    def _start(self) -> None:
        blocks = {}
        for prefix, m in (("a", self.A), ("at", self.AT)):
            blocks[prefix + "_data"] = self._share(m.data)
            blocks[prefix + "_indices"] = self._share(m.indices)
            blocks[prefix + "_indptr"] = self._share(m.indptr)
        blocks["vol"] = self._share(np.zeros(self.vol_shape, dtype=self.A.dtype))
        blocks["proj"] = self._share(np.zeros(self.proj_shape, dtype=self.A.dtype))
        self.vol = np.ndarray(self.vol_shape, dtype=self.A.dtype,
                              buffer=self._shm[-2].buf)
        self.proj = np.ndarray(self.proj_shape, dtype=self.A.dtype,
                               buffer=self._shm[-1].buf)
        spec = {"blocks": blocks, "a_shape": self.A.shape,
                "at_shape": self.AT.shape}
        self._ranges = split(self.n_cols, self.workers)
        self._pool = get_context("spawn").Pool(
            processes=len(self._ranges), initializer=_attach, initargs=(spec,))

    # -- in-place operations on the shared buffers ------------------------
    def forward_inplace(self) -> np.ndarray:
        """``proj <- A @ vol``, both already in the buffers."""
        if self._pool is None:
            self.proj[...] = self.A @ self.vol
        else:
            self._pool.map(_forward, self._ranges)
        return self.proj

    def backward_inplace(self) -> np.ndarray:
        """``vol <- A.T @ proj``, both already in the buffers."""
        if self._pool is None:
            self.vol[...] = self.AT @ self.proj
        else:
            self._pool.map(_backward, self._ranges)
        return self.vol

    def descend(self, dt: float, positivity: bool = True,
                threshold: float = 0.0) -> np.ndarray:
        """``vol -= dt * (A.T @ proj)``, soft-threshold, clip.

        ``proj`` is the residual.  ``threshold`` subtracts a constant before
        the clip; together the two are the proximal operator of an L1 penalty
        restricted to the non-negative orthant, which is what makes it one
        subtraction rather than a separate pass.
        """
        t = float(threshold)
        if self._pool is None:
            self.vol -= dt * (self.AT @ self.proj)
            if t:
                self.vol -= t
            if positivity:
                np.maximum(self.vol, 0, out=self.vol)
        else:
            self._pool.map(_descend, [(lo, hi, dt, positivity, t)
                                      for lo, hi in self._ranges])
        return self.vol

    # -- copying interface, for tests and the adjoint check ---------------
    def forward(self, vol_flat: np.ndarray) -> np.ndarray:
        self.vol[...] = vol_flat
        return self.forward_inplace()

    def backward(self, proj_flat: np.ndarray) -> np.ndarray:
        self.proj[...] = proj_flat
        return self.backward_inplace()

    # -- teardown ---------------------------------------------------------
    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self.vol = self.proj = None
        for shm in self._shm:
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, BufferError):
                pass
        self._shm = []

    def __enter__(self) -> "ParallelMatmul":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
