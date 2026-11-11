"""Reconstruction configuration.

Upstream GENFIRE keeps its parameters in a mutable ``ReconstructionParameters``
object with getters and setters and no notion of a physical pixel size.  For
atomic-resolution work the pixel size is the parameter that decides whether a
reconstruction means anything, so it is required here and every resolution the
code reports is in angstroms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .fft import default_threads

__all__ = ["ReconstructionConfig", "RESOLUTION_MODES"]

# constraintEnforcementDelayIndicators in upstream terms: cutoffs on the
# normalised spatial frequency, spread evenly over the iterations.  A grid
# point is enforced once its resolution indicator exceeds the current cutoff.
RESOLUTION_MODES = {
    # extend outwards then suppress back inwards
    "extend_suppress": np.concatenate(
        [np.arange(0.95, -0.25, -0.15), np.arange(-0.15, 0.95, 0.10)]
    ).astype(np.float32),
    # enforce every measured point from the first iteration
    "none": np.array([-999.0, -999.0, -999.0, -999.0], dtype=np.float32),
    # low resolution first, working outwards, then hold
    "extend": np.concatenate(
        [np.arange(0.95, -0.15, -0.15), [-0.15, -0.15, -0.15]]
    ).astype(np.float32),
}


@dataclass
class ReconstructionConfig:
    """Everything the reconstructor needs that is not data.

    Attributes
    ----------
    pixel_size : float
        Real-space sampling of the input projections, in angstroms.  0.3 A is
        typical for atomic electron tomography.
    oversampling : float
        Zero-padding factor applied before gridding.  The padded array is
        ``round(n * oversampling)`` rounded up to an FFT-friendly even size, so
        non-integer ratios are allowed (upstream forced an integer).
    num_iterations : int
        GENFIRE iterations.
    threads : int
        FFTW worker threads.  ``None`` means every logical core.
    dtype : numpy dtype
        float32 (default) or float64.  float32 halves the memory and roughly
        doubles the FFT throughput; the iterative projection is robust to it.
    interpolation_cutoff : float
        Gridding kernel radius in pixels.  Values farther than this from a grid
        point do not contribute.
    grid_multiple : bool
        Let one measured value contribute to several neighbouring grid points
        (upstream ``permitMultipleGridding``).
    positivity, use_support : bool
        Real-space constraints.
    resolution_mode : str
        One of :data:`RESOLUTION_MODES`.
    calculate_rfree : bool
        Withhold ``rfree_fraction`` of the measured points per shell and track
        the error against them.
    max_gigabytes : float or None
        Advisory memory ceiling.  ``estimate_memory`` compares against it and
        :meth:`validate` raises when the plan cannot fit.
    """

    pixel_size: float = 0.3
    oversampling: float = 2.0
    num_iterations: int = 100
    threads: Optional[int] = None
    dtype: np.dtype = np.dtype(np.float32)
    planner: str = "FFTW_MEASURE"
    in_place_fft: bool = True

    interpolation_cutoff: float = 0.7
    grid_multiple: bool = True
    enforce_resolution_circle: bool = True
    grid_chunk_bytes: int = 1 << 28   # float64 accumulator budget per chunk
    grid_threads: Optional[int] = None  # collection threads; None = threads
    loop_threads: Optional[int] = None  # iteration-loop threads; 1 = serial

    positivity: bool = True
    use_support: bool = True
    support_shape: str = "box"        # "box" or "sphere"
    support_soft_edge: float = 0.0    # taper width in voxels; 0 is a hard edge
    prune_fft: bool = True            # skip FFT passes over the support's zeros
    resolution_mode: str = "none"
    track_best: bool = True           # keep the lowest-error iterate

    #: GENFIRE v2 (UCLA "GENFIRE-v2-DR") replaces the exact Fourier
    #: projection with a relaxed one and extrapolates between successive
    #: outputs.  Its three "DR" reconstructors all reduce to
    #:
    #:     v_k     = P_M^dt ( P_S ( x_k ) )
    #:     x_{k+1} = v_k + ds * ( v_k - v_{k-1} )
    #:
    #: once the auxiliary variable is eliminated -- momentum on the
    #: iterates, not the Douglas-Rachford operator the name suggests.
    #: Defaults here are v2's own (ds_value 0.5, dt_value 0.1); zero for
    #: both reproduces plain error reduction exactly.
    #:
    #: dt keeps a fraction of the current spectrum instead of overwriting
    #: it, so the constraint is no longer a projection onto the measured
    #: set.  It leaves the fixed points alone and stays averaged, so the
    #: POCS convergence argument survives it.  ds does not: inertial
    #: Krasnoselskii-Mann results need a small momentum coefficient
    #: (bounds near 1/3 appear in the literature) and v2 ships 0.5, with
    #: its scheduled variant starting near 1.  Convergence is then
    #: heuristic, not proven.
    fourier_relaxation: float = 0.0   # dt: 0 = overwrite, exact projection
    momentum: float = 0.0             # ds: 0 = plain error reduction

    calculate_rfree: bool = False
    rfree_fraction: float = 0.05
    rfree_shell_pixels: float = 1.0
    random_seed: int = 0

    max_gigabytes: Optional[float] = None
    verbose: bool = True
    progress_every: int = 1

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64")
        if self.threads is None:
            self.threads = default_threads()
        self.threads = max(1, int(self.threads))
        if self.grid_threads is None:
            self.grid_threads = self.threads
        self.grid_threads = max(1, int(self.grid_threads))
        if self.loop_threads is None:
            self.loop_threads = self.threads
        self.loop_threads = max(1, int(self.loop_threads))
        if self.resolution_mode not in RESOLUTION_MODES:
            raise ValueError(
                "resolution_mode must be one of %s" % sorted(RESOLUTION_MODES))
        if self.support_shape not in ("box", "sphere"):
            raise ValueError("support_shape must be 'box' or 'sphere'")
        if self.support_soft_edge < 0:
            raise ValueError("support_soft_edge must be >= 0")
        if self.oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")

    # -- derived quantities -----------------------------------------------
    @property
    def complex_dtype(self) -> np.dtype:
        return np.dtype(np.complex64 if self.dtype == np.float32 else np.complex128)

    @property
    def enforcement_cutoffs(self) -> np.ndarray:
        return RESOLUTION_MODES[self.resolution_mode]

    def padded_size(self, n: int) -> int:
        """Padded edge length for an ``n``-pixel projection.

        Rounded up to an even number with only 2/3/5/7 factors, which is where
        FFTW is fastest.  ``n=512, oversampling=2`` gives 1024.
        """
        target = int(np.ceil(n * self.oversampling))
        return next_fast_even(max(target, n))

    def nyquist_resolution(self) -> float:
        """Best resolution the sampling can represent, in angstroms."""
        return 2.0 * self.pixel_size

    def box_angstroms(self, n: int) -> float:
        return n * self.pixel_size

    #: mean number of grid points a measured sample contributes to, for the
    #: default 0.7-pixel cutoff with multiple gridding.  Measured at 1024**3:
    #: 65.3M contributions from 41 x 1024**2 samples = 1.52.
    CONTRIBUTIONS_PER_SAMPLE = 1.6

    def estimate_memory(self, padded_n: int, dtype=None,
                        fill_fraction: float = 0.05,
                        num_projections: Optional[int] = None) -> dict:
        """Peak resident bytes of the main arrays, by phase.

        The two phases have different peaks.  Gridding holds the whole
        half-grid, the accumulated contributions and a float64 ``bincount``
        output; the iteration loop holds the FFT buffer, the support and the
        compressed measured set, but by then the half-grid is gone.  ``total``
        is the larger of the two.

        ``fill_fraction`` is the share of half-grid points a tilt series
        actually reaches -- about 0.04 for a 3-degree single-axis series at
        1024**3.  ``num_projections`` sizes the gridding scratch; without it
        the gridding phase is under-reported.  The caller's projection stack
        and the Python interpreter sit on top of all of this.
        """
        dtype = np.dtype(dtype or self.dtype)
        itemsize = dtype.itemsize
        n = padded_n
        half = n * n * (n // 2 + 1)
        full = n * n * n
        needs_radial = self.calculate_rfree or self.resolution_mode != "none"
        nnz = int(half * fill_fraction)

        # The gridder accumulates (index, value, weight) per contribution in
        # buffers that double when they fill.  At the last doubling the old
        # and new buffers coexist, so the transient peak is twice what the
        # contributions themselves occupy -- not 1.5x, which is what the
        # filled fraction would suggest.
        per_contribution = 8 + 2 * itemsize + itemsize   # index, value, weight
        contributions = int((num_projections or 0) * n * n
                            * self.CONTRIBUTIONS_PER_SAMPLE)
        # a reallocation holds the old buffer and the new one at once, and the
        # new one is at most the content plus one growth step
        candidates = ((2 * contributions + (1 << 25) if contributions else 0)
                      * per_contribution)

        # The accumulator is chunked, so it costs whatever the chunk budget
        # says rather than a float64 copy of the whole half-grid, and the
        # radial map is never materialised at all.  Each chunk also selects
        # its contributions, which needs a bool over all of them plus the
        # gathered slice.
        chunk = min(int(self.grid_chunk_bytes), max(1, half * 8))
        num_chunks = max(1, -(-half * 8 // chunk))
        # The gridder emits the filled points directly, so neither the dense
        # half-grid nor its mask is ever allocated.  One float64 accumulator,
        # plus the float32 denominator and its cast, live at once.
        gridding = {
            "accumulator_chunk": chunk + chunk // 2,
            "gridding_candidates": candidates,
            "chunk_selection": contributions
            + contributions // num_chunks * (8 + 2 * itemsize + 2 * itemsize),
            "emitted_points": 2 * nnz * (8 + 2 * itemsize),
        }
        # index (int64), value, |value|, and |k| only where it is needed
        measured_set = nnz * (8 + 2 * itemsize + itemsize
                              + (itemsize if needs_radial else 0))
        # the support mask, when there is one, is stored over its bounding box
        box = int(full * min(1.0, (1.0 / self.oversampling) ** 3))
        iterating = {
            "fft_buffer": half * 2 * itemsize if self.in_place_fft
            else full * itemsize + half * 2 * itemsize,
            "support": 0 if self.support_shape == "box" and not self.support_soft_edge
            else box * itemsize,
            # two crop-sized buffers, swapped rather than copied
            "best_object": 2 * box * itemsize if self.track_best else 0,
            "measured_set": measured_set,
        }
        parts = dict(gridding)
        parts.update(iterating)
        parts["gridding_phase"] = sum(gridding.values())
        parts["iteration_phase"] = sum(iterating.values())
        parts["total"] = max(parts["gridding_phase"], parts["iteration_phase"])
        return parts

    def validate(self, projection_shape: Tuple[int, int, int]) -> dict:
        """Check the plan against the memory ceiling; return the memory report."""
        n = int(projection_shape[0])
        n_proj = int(projection_shape[2]) if len(projection_shape) > 2 else 1
        padded = self.padded_size(n)
        report = self.estimate_memory(padded, num_projections=n_proj)
        report["padded_size"] = padded
        limit = self.max_gigabytes
        if limit is None:
            limit = _available_gigabytes()
        if limit is not None and report["total"] / 2 ** 30 > limit:
            raise MemoryError(
                "reconstruction needs about %.1f GB for a %d**3 grid but only "
                "%.1f GB is available; lower the oversampling, use float32, or "
                "raise max_gigabytes"
                % (report["total"] / 2 ** 30, padded, limit))
        return report

    def describe(self, n: int, num_projections: Optional[int] = None) -> str:
        padded = self.padded_size(n)
        mem = self.estimate_memory(padded, num_projections=num_projections)
        return "\n".join([
            "projection size      : %d x %d px" % (n, n),
            "pixel size           : %.4f A" % self.pixel_size,
            "oversampling         : %.3g  ->  padded grid %d^3" % (self.oversampling, padded),
            "object box           : %.1f A   padded box %.1f A"
            % (self.box_angstroms(n), self.box_angstroms(padded)),
            "Nyquist resolution   : %.3f A" % self.nyquist_resolution(),
            "precision            : %s" % self.dtype.name,
            "threads              : %d" % self.threads,
            "iterations           : %d" % self.num_iterations,
            "support              : %s%s" % (
                self.support_shape,
                "" if not self.support_soft_edge
                else ", soft edge %g px (%.2f A)" % (
                    self.support_soft_edge,
                    self.support_soft_edge * self.pixel_size)),
            "resolution mode      : %s" % self.resolution_mode,
            "estimated peak memory: %.2f GB  (gridding %.2f, iterating %.2f)"
            % (mem["total"] / 2 ** 30, mem["gridding_phase"] / 2 ** 30,
               mem["iteration_phase"] / 2 ** 30),
        ])


def next_fast_even(n: int) -> int:
    """Smallest even m >= n whose only prime factors are 2, 3, 5 and 7."""
    if n <= 2:
        return 2
    m = n if n % 2 == 0 else n + 1
    while True:
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 2


def _available_gigabytes() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    # leave a little headroom for the OS and the gridding scratch
    return psutil.virtual_memory().available / 2 ** 30 * 0.9
