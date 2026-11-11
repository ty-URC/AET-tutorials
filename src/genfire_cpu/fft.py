"""Planned, multi-threaded, single-precision 3D FFTs for GENFIRE.

Upstream GENFIRE calls ``pyfftw.interfaces.numpy_fft.rfftn`` (or ``numpy.fft``)
once per iteration.  That path re-plans the transform on every call, allocates
a fresh output array every call, and -- for ``numpy.fft`` -- silently promotes
float32 input to float64.  At 1024**3 that is the difference between a
reconstruction that runs and one that does not fit in memory.

This module builds the plan once and reuses a single buffer for every
iteration:

* in-place real-to-complex, so the volume and its half-spectrum share one
  4 GB allocation at 1024**3 instead of two,
* ``FFTW_MEASURE`` plans persisted as wisdom, so the ~150 s planning cost at
  1024**3 is paid once per machine and shape,
* an explicit thread count, so the caller decides how many cores to use,
* optional **pruning**: when the volume is known to be zero outside a
  bounding box -- which is exactly what a support guarantees -- the 3D
  transform is split into its three 1D passes and the passes that would
  operate on all-zero data are simply not run.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly by the backend selection tests
    import pyfftw

    HAVE_PYFFTW = True
except ImportError:  # pragma: no cover
    pyfftw = None
    HAVE_PYFFTW = False

__all__ = [
    "HAVE_PYFFTW",
    "inverse_norm",
    "RealFFT3D",
    "default_threads",
    "wisdom_dir",
    "load_wisdom",
    "save_wisdom",
]

def _volume(shape) -> float:
    """Number of elements in ``shape``, as a float.

    Not ``np.prod``: on Windows it infers int32 from a tuple of python ints and
    silently wraps.  1536**3 comes back as -671088640, which as an inverse-FFT
    normalisation is the wrong sign and 5.4x the wrong magnitude -- a
    reconstruction that diverges from the third iteration.  The threshold is
    n = 1291, so it never shows up below a 1290**3 grid.
    """
    total = 1.0
    for s_ in shape:
        total *= float(s_)
    return total


_REAL_TO_COMPLEX = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
}


def default_threads() -> int:
    """Threads to use when the caller does not say.

    ``GENFIRE_NUM_THREADS`` wins, then the CPU count.  On a hybrid CPU the
    logical count includes E-cores; they are slower per-thread but a
    memory-bound transform still profits from them, so we take them all.
    """
    env = os.environ.get("GENFIRE_NUM_THREADS")
    if env:
        return max(1, int(env))
    return os.cpu_count() or 1


def wisdom_dir() -> Path:
    """Directory holding the persisted FFTW wisdom."""
    override = os.environ.get("GENFIRE_WISDOM_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "genfire_cpu" / "wisdom"
    return Path.home() / ".cache" / "genfire_cpu" / "wisdom"


def _wisdom_path() -> Path:
    return wisdom_dir() / "fftw_wisdom.pkl"


def load_wisdom() -> bool:
    """Import previously saved FFTW wisdom.  True if anything loaded."""
    if not HAVE_PYFFTW:
        return False
    path = _wisdom_path()
    if not path.is_file():
        return False
    try:
        with path.open("rb") as fh:
            pyfftw.import_wisdom(pickle.load(fh))
        return True
    except Exception:  # corrupt, or from another FFTW build - just re-plan
        return False


def save_wisdom() -> None:
    if not HAVE_PYFFTW:
        return
    path = _wisdom_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(pyfftw.export_wisdom(), fh)
    tmp.replace(path)


def inverse_norm(shape, dtype=np.float32):
    """The 1/N an unnormalised inverse transform of ``shape`` needs."""
    return np.dtype(dtype).type(1.0 / _volume(shape))


class RealFFT3D:
    """A reusable in-place 3D real FFT of a fixed shape.

    The object owns one buffer.  ``real`` is a view of it holding the volume,
    ``spectrum`` is a view of the same memory holding the half-spectrum
    (``shape[-1] // 2 + 1`` along the last axis).  ``forward()`` overwrites
    ``real`` with ``spectrum``; ``backward()`` does the reverse.  Because they
    alias, never assume one is intact after transforming to the other.

    Parameters
    ----------
    shape : (nx, ny, nz); nz must be even to use the in-place layout.
    dtype : real dtype, float32 (default) or float64.
    threads : worker threads, ``default_threads()`` when None.
    planner : "FFTW_MEASURE" (default), "FFTW_ESTIMATE" or "FFTW_PATIENT".
        MEASURE costs ~150 s once at 1024**3 and then runs ~2x faster than
        ESTIMATE; the cost is amortised through the wisdom cache.
    in_place : share one buffer between volume and spectrum.  Halves peak
        memory; the real view is strided, which costs a little on the
        elementwise constraint steps.
    bounds : ((x0, x1), (y0, y1), (z0, z1)) or None.
        Promise that the volume is zero outside this box, and that the inverse
        output is only needed inside it.  With it the transform is pruned; see
        :meth:`_plan_pruned`.  The caller must keep the promise -- call
        :meth:`clear_outside` before every :meth:`forward`.

    Notes
    -----
    After a pruned :meth:`backward` the volume is meaningful **only inside**
    ``bounds``; everywhere else the buffer holds partial transforms.  That is
    exactly the region a support would keep anyway.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        dtype=np.float32,
        threads: Optional[int] = None,
        planner: str = "FFTW_MEASURE",
        in_place: bool = True,
        use_wisdom: bool = True,
        bounds: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        shape = tuple(int(s) for s in shape)
        if len(shape) != 3:
            raise ValueError("RealFFT3D is 3D only, got shape %r" % (shape,))
        self.shape = shape
        self.dtype = np.dtype(dtype)
        if self.dtype not in _REAL_TO_COMPLEX:
            raise ValueError("dtype must be float32 or float64, got %s" % self.dtype)
        self.cdtype = _REAL_TO_COMPLEX[self.dtype]
        self.threads = int(threads) if threads else default_threads()
        self.planner = planner
        self.in_place = bool(in_place) and HAVE_PYFFTW and shape[2] % 2 == 0
        self.spectrum_shape = (shape[0], shape[1], shape[2] // 2 + 1)
        self.plan_seconds = 0.0
        self.backend = "pyfftw" if HAVE_PYFFTW else "scipy"

        if bounds is None:
            bounds = tuple((0, s) for s in shape)
        self.bounds = tuple((int(a), int(b)) for a, b in bounds)
        for (a, b), s_ in zip(self.bounds, shape):
            if not (0 <= a < b <= s_):
                raise ValueError("bad bounds %r for shape %r"
                                 % (self.bounds, shape))
        # pruning only buys anything if whole (x, y) columns can be skipped,
        # and it needs the shared in-place buffer to stay consistent
        (x0, x1), (y0, y1), _ = self.bounds
        self.pruned = (HAVE_PYFFTW and self.in_place
                       and (x1 - x0 < shape[0] or y1 - y0 < shape[1]))

        if HAVE_PYFFTW:
            self._init_pyfftw(use_wisdom)
        else:
            self._init_scipy()

    # -- construction -----------------------------------------------------
    def _init_pyfftw(self, use_wisdom: bool) -> None:
        nz = self.shape[2]
        if self.in_place:
            self._buf = pyfftw.empty_aligned(self.spectrum_shape, dtype=self.cdtype)
            self.spectrum = self._buf
            # the real view spans 2*(nz//2+1) reals per row; the volume is the
            # first nz of them, the remaining are FFTW's in-place padding
            self.real = self._buf.view(self.dtype)[:, :, :nz]
        else:
            self.real = pyfftw.empty_aligned(self.shape, dtype=self.dtype)
            self.spectrum = pyfftw.empty_aligned(self.spectrum_shape, dtype=self.cdtype)

        if use_wisdom:
            load_wisdom()

        flags = (self.planner,)
        t0 = time.perf_counter()
        if self.pruned:
            self._plan_pruned(flags)
        else:
            self._fwd = pyfftw.FFTW(
                self.real, self.spectrum, axes=(0, 1, 2),
                direction="FFTW_FORWARD", flags=flags, threads=self.threads,
            )
            # the c2r transform may clobber its input: after backward() the
            # spectrum is garbage anyway, because in-place it aliases the volume
            self._bwd = pyfftw.FFTW(
                self.spectrum, self.real, axes=(0, 1, 2),
                direction="FFTW_BACKWARD",
                flags=flags + ("FFTW_DESTROY_INPUT",),
                threads=self.threads,
            )
        self.plan_seconds = time.perf_counter() - t0
        if use_wisdom:
            try:
                save_wisdom()
            except OSError:
                pass

    def _plan_pruned(self, flags) -> None:
        """Split the 3D transform into 1D passes and skip the all-zero ones.

        A rank-3 real FFT is three batched 1D passes: r2c along z for every
        (x, y) column, then c2c along y, then c2c along x.  If the volume is
        zero outside a bounding box then

        * the z pass only has data for (x, y) inside the box -- at 2x
          oversampling that is a quarter of the columns,
        * after it, the y pass only has data for x inside the box -- half the
          columns,
        * by the x pass everything is populated, so it runs in full.

        Cost goes from 3 full passes to 1 + 1/2 + 1/4, about 42% less work.
        The inverse is the mirror image: the x pass runs in full, then only
        the columns whose output is wanted are carried through, saving the
        same fraction.

        Each pass is planned over a strided view of the one shared buffer, so
        nothing is copied between passes.
        """
        (x0, x1), (y0, y1), _ = self.bounds
        xs, ys = slice(x0, x1), slice(y0, y1)

        self._fwd_passes = (
            pyfftw.FFTW(self.real[xs, ys, :], self.spectrum[xs, ys, :],
                        axes=(2,), direction="FFTW_FORWARD",
                        flags=flags, threads=self.threads),
            pyfftw.FFTW(self.spectrum[xs], self.spectrum[xs], axes=(1,),
                        direction="FFTW_FORWARD", flags=flags,
                        threads=self.threads),
            pyfftw.FFTW(self.spectrum, self.spectrum, axes=(0,),
                        direction="FFTW_FORWARD", flags=flags,
                        threads=self.threads),
        )
        # no FFTW_DESTROY_INPUT here: FFTW defaults rank-1 c2r to
        # PRESERVE_INPUT, but for an in-place transform that is moot and
        # measuring the two side by side at 1024**3 showed no difference
        # (0.919 s vs 0.902 s median, inside the run-to-run spread)
        self._bwd_passes = (
            pyfftw.FFTW(self.spectrum, self.spectrum, axes=(0,),
                        direction="FFTW_BACKWARD", flags=flags,
                        threads=self.threads),
            pyfftw.FFTW(self.spectrum[xs], self.spectrum[xs], axes=(1,),
                        direction="FFTW_BACKWARD", flags=flags,
                        threads=self.threads),
            pyfftw.FFTW(self.spectrum[xs, ys, :], self.real[xs, ys, :],
                        axes=(2,), direction="FFTW_BACKWARD", flags=flags,
                        threads=self.threads),
        )
        self._norm = inverse_norm(self.shape, self.dtype)

    def clear_outside(self, bounds=None, parallel=None) -> None:
        """Zero the volume outside ``bounds``, FFTW padding included.

        The in-place layout keeps a little padding at the end of every row.
        Zeroing only the real view would leave that padding stale, and a
        pruned forward -- which never writes the skipped rows -- would then
        read it back as spectrum.  Zeroing whole rows through the *complex*
        view covers data and padding in one write.

        This is 3.5 GiB of stores per iteration at 1024**3 and NumPy does them
        on one core; handing ``parallel`` a :class:`~.parallel.Parallel` cuts
        each slab along its free axis and is worth about 4x.  The slabs are
        disjoint, so the result does not depend on the split.
        """
        (x0, x1), (y0, y1), (z0, z1) = bounds or self.bounds
        nx, ny, nz = self.shape
        rows = self.spectrum if self.in_place else self.real
        jobs = []

        def zero(target, prefix, lo, hi, parts):
            """Queue ``target[prefix + (lo:hi,)] = 0``, cut into ``parts``."""
            if hi <= lo:
                return
            from .parallel import split
            for piece in split(hi - lo, parts):
                sl = slice(lo + piece.start, lo + piece.stop)
                jobs.append(lambda t=target, p=prefix, s=sl: t.__setitem__(p + (s,), 0))

        parts = parallel.threads if parallel is not None else 1
        zero(rows, (), 0, x0, parts)
        zero(rows, (), x1, nx, parts)
        # the y and z slabs are cut along x, which is free once x is fixed
        from .parallel import split
        for piece in (split(x1 - x0, parts) if x1 > x0 else []):
            xs = slice(x0 + piece.start, x0 + piece.stop)
            if y0:
                jobs.append(lambda s=xs: rows.__setitem__((s, slice(0, y0)), 0))
            if y1 < ny:
                jobs.append(lambda s=xs: rows.__setitem__((s, slice(y1, ny)), 0))
            if z0:
                jobs.append(lambda s=xs: self.real.__setitem__(
                    (s, slice(y0, y1), slice(0, z0)), 0))
            if z1 < nz:
                jobs.append(lambda s=xs: self.real.__setitem__(
                    (s, slice(y0, y1), slice(z1, nz)), 0))

        if parallel is None:
            for job in jobs:
                job()
        else:
            parallel.run(jobs)

    @property
    def valid_region(self):
        """Where :meth:`backward` leaves meaningful data."""
        if not self.pruned:
            return (slice(None),) * 3
        return tuple(slice(a, b) for a, b in self.bounds)

    @property
    def work_fraction(self) -> float:
        """Share of the unpruned transform work the plan actually does."""
        if not self.pruned:
            return 1.0
        nx, ny = self.shape[0], self.shape[1]
        (x0, x1), (y0, y1), _ = self.bounds
        fx, fy = (x1 - x0) / nx, (y1 - y0) / ny
        return (fx * fy + fx + 1.0) / 3.0

    def _init_scipy(self) -> None:
        """Fallback when pyfftw is unavailable.

        scipy.fft honours float32 (unlike numpy.fft) and takes a ``workers``
        argument, so it is a usable if slower stand-in.  It cannot work
        in-place, so both arrays are allocated.
        """
        self.in_place = False
        self.real = np.zeros(self.shape, dtype=self.dtype)
        self.spectrum = np.zeros(self.spectrum_shape, dtype=self.cdtype)

    # -- transforms -------------------------------------------------------
    def forward(self) -> np.ndarray:
        """volume (``real``) -> half-spectrum (``spectrum``), unnormalised."""
        if self.pruned:
            # execute(), not __call__(): pyfftw's __call__ corrupts a forward
            # transform when normalise_idft is turned off, and the pruned
            # inverse does its own normalisation anyway
            for p in self._fwd_passes:
                p.execute()
        elif HAVE_PYFFTW:
            self._fwd()
        else:
            import scipy.fft as sfft

            self.spectrum[...] = sfft.rfftn(self.real, workers=self.threads)
        return self.spectrum

    def backward(self) -> np.ndarray:
        """half-spectrum -> volume, normalised by 1/N so it round-trips.

        When pruned, only ``valid_region`` is transformed and normalised.
        """
        if self.pruned:
            for p in self._bwd_passes:
                p.execute()
            self.real[self.valid_region] *= self._norm
        elif HAVE_PYFFTW:
            self._bwd()  # pyfftw.FFTW.__call__ normalises the inverse
        else:
            import scipy.fft as sfft

            self.real[...] = sfft.irfftn(self.spectrum, self.shape,
                                         workers=self.threads)
        return self.real

    # -- convenience ------------------------------------------------------
    def set_real(self, volume: np.ndarray) -> np.ndarray:
        """Copy ``volume`` into the buffer, casting if needed."""
        if tuple(volume.shape) != self.shape:
            raise ValueError("expected %r, got %r" % (self.shape, volume.shape))
        self.real[...] = volume
        return self.real

    @property
    def nbytes(self) -> int:
        if self.in_place:
            return self._buf.nbytes
        return self.real.nbytes + self.spectrum.nbytes

    def __repr__(self) -> str:
        pruning = "off"
        if self.pruned:
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            pruning = "%dx%dx%d box, %.0f%% of the work" % (
                x1 - x0, y1 - y0, z1 - z0, 100 * self.work_fraction)
        return ("RealFFT3D(shape=%r, dtype=%s, threads=%d, planner=%s, "
                "in_place=%s, backend=%s, %.2f GB, pruning=%s)"
                % (self.shape, self.dtype.name, self.threads, self.planner,
                   self.in_place, self.backend, self.nbytes / 2 ** 30, pruning))
