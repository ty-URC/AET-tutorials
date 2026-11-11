"""The GENFIRE iterative reconstruction loop.

Algorithmically this is upstream GENFIRE: alternate between real-space
constraints (positivity, support) and replacing the measured Fourier
components, with an optional resolution extension/suppression schedule.

The differences are all about making it survive a 1024**3 grid:

* the FFT plan and its buffer are created once and reused, and are **pruned**
  against the support -- the passes that would transform all-zero data are not
  run at all (see :mod:`.fft`),
* the measured set is addressed by linear index rather than by a boolean mask
  over the whole half-grid, so an iteration touches a few percent of the
  spectrum instead of sweeping all of it,
* the constraint steps are in-place and confined to the support box,
* float32 by default,
* the R-free bookkeeping actually runs (upstream builds a ragged nested list
  and calls ``np.shape`` on it, which raises on NumPy >= 1.24).

Coordinate conventions
----------------------
The **volume is centred in the array** -- the object sits in the middle, so
the support is one contiguous block and the FFT can be pruned against it with
three plans instead of seven.  The **spectrum stays in FFT order**, matching
``numpy.fft.rfftn`` and the layout :mod:`.grid` produces.

Those two conventions differ by a half-box shift, which in Fourier space is
exactly a factor ``(-1)**(kx+ky+kz)``.  It is folded into the measured values
once, before the loop, and costs nothing thereafter.  Nothing is ever
``fftshift``-ed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import ReconstructionConfig
from .fft import RealFFT3D
from .grid import fill_fourier_grid, radial_at
from .parallel import Parallel, split
from .support import Support, box_support, sphere_support, support_from_array

__all__ = ["ReconstructionResult", "reconstruct", "pad_projections",
           "build_support", "centre_offset"]


@dataclass
class ReconstructionResult:
    """Output of :func:`reconstruct`."""

    reconstruction: np.ndarray          # (n0, n0, n0) real, object-centred
    error_k: np.ndarray                 # reciprocal-space error per iteration
    rfree_total: Optional[np.ndarray] = None
    rfree_by_shell: Optional[np.ndarray] = None
    shell_resolution: Optional[np.ndarray] = None   # angstroms, per shell
    padded_size: int = 0
    seconds: float = 0.0
    gridding_seconds: float = 0.0
    planning_seconds: float = 0.0
    iteration_seconds: float = 0.0
    #: Process CPU time over the whole run.  Wall clock keeps advancing while
    #: a laptop is in modern standby but the process does not run, so a timing
    #: that spans a sleep is meaningless: one 1536**3 gridding was recorded at
    #: 9042 s and the identical one, same contribution count, at 105 s.
    #: process_time() does not tick while suspended, so the ratio of the two
    #: tells "slow" apart from "was not running".
    cpu_seconds: float = 0.0
    fft_work_fraction: float = 1.0
    support: Optional[Support] = None
    config: Optional[ReconstructionConfig] = None

    def summary(self) -> str:
        lines = [
            "grid                 : %d^3" % self.padded_size,
            "support              : %r" % (self.support,),
            "FFT work after prune : %.0f%%" % (100 * self.fft_work_fraction),
            "iterations           : %d" % self.error_k.size,
            "final k-space error  : %.5f" % float(self.error_k[-1]),
        ]
        if self.rfree_total is not None:
            lines.append("final R-free         : %.5f"
                         % float(self.rfree_total[-1]))
        lines += [
            "gridding             : %.1f s" % self.gridding_seconds,
            "FFT planning         : %.1f s" % self.planning_seconds,
            "iteration loop       : %.1f s  (%.2f s/iter)"
            % (self.iteration_seconds,
               self.iteration_seconds / max(1, self.error_k.size)),
            "total                : %.1f s" % self.seconds,
        ]
        if self.cpu_seconds and self.seconds:
            busy = self.cpu_seconds / self.seconds
            lines.append("CPU time             : %.1f s  (%.1f cores busy)"
                         % (self.cpu_seconds, busy))
            if busy < 0.5:
                lines.append("  the process was idle for most of that "
                             "wall clock; if this is a laptop it slept, "
                             "and the timings above mean nothing")
        return "\n".join(lines)


# -- helpers ---------------------------------------------------------------
def centre_offset(n: int, width: int) -> int:
    """Where a ``width``-wide window starts if it is centred on the FFT origin.

    Centred on ``n // 2``, not on ``(n - width) // 2``.  The two agree for even
    widths and differ by one voxel for odd ones, where the floor division puts
    a width-63 box at [-32, +30] about the origin instead of [-31, +31].
    Measured cost of getting that wrong: correlation with the ground truth
    0.931 at odd projection sizes against 0.968 at even ones.
    """
    return n // 2 - width // 2


def pad_projections(projections: np.ndarray, n: int, dtype=np.float32) -> np.ndarray:
    """Zero-pad each projection to ``n`` x ``n``, centred on the FFT origin."""
    n0 = projections.shape[0]
    if n0 == n:
        return np.ascontiguousarray(projections, dtype=dtype)
    if n < n0:
        raise ValueError("padded size %d smaller than projection size %d" % (n, n0))
    lo = centre_offset(n, n0)
    out = np.zeros((n, n, projections.shape[2]), dtype=dtype)
    out[lo:lo + n0, lo:lo + n0] = projections
    return out


def build_support(n: int, n0: int, cfg: ReconstructionConfig,
                  support=None) -> Support:
    """Turn whatever the caller gave us into a :class:`~.support.Support`."""
    if isinstance(support, Support):
        return support
    if support is not None:
        return support_from_array(n, np.asarray(support),
                                  soft_edge=cfg.support_soft_edge)
    if cfg.support_shape == "sphere":
        return sphere_support(n, n0, soft_edge=cfg.support_soft_edge)
    return box_support(n, n0, soft_edge=cfg.support_soft_edge)


#: How many partial sums the error metric is built from.  Fixed, and
#: deliberately independent of the thread count: adding partials in a
#: different order moves the last bits, so a serial run and a threaded run
#: have to divide the sum the same way to agree.  Changing this number
#: changes the reported error in its last digits.
ERROR_PARTITIONS = 16


def _error(spec_flat, lin, measured_abs, pieces, partials, parallel):
    """sum |‖spectrum‖ - ‖measured‖| over the measured points.

    Split into a fixed number of partials whether or not threads are used, so
    the value does not depend on how the machine ran it.
    """
    def piece(i, sl):
        cur = np.abs(spec_flat[lin[sl]])
        cur -= measured_abs[sl]
        np.abs(cur, out=cur)
        partials[i] = cur.sum()

    parallel.run([(lambda i=i, sl=sl: piece(i, sl))
                  for i, sl in enumerate(pieces)])
    return float(partials.sum())


def _centre_slice(n: int, width: int) -> Tuple[slice, slice, slice]:
    lo = centre_offset(n, width)
    return (slice(lo, lo + width),) * 3


def _halfgrid_parity(lin: np.ndarray, n: int, nh: int) -> np.ndarray:
    """``(kx + ky + kz) & 1`` at linear indices of an ``(n, n, nh)`` grid.

    Multiplying the spectrum by ``(-1)**(kx+ky+kz)`` is exactly the half-box
    real-space shift between FFT order and a centred volume.
    """
    kx, rem = np.divmod(lin, n * nh)
    ky, kz = np.divmod(rem, nh)
    kx += ky
    kx += kz
    return (kx & 1).astype(bool)


_radial_at = radial_at   # kept as a name for the tests


def _iteration_cutoffs(cutoffs: np.ndarray, num_iterations: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Map the enforcement cutoffs onto iteration numbers (1-based)."""
    marks = np.round(np.linspace(1, num_iterations, num=cutoffs.size))
    marks, keep = np.unique(marks, return_index=True)
    return marks, cutoffs[keep]


# -- main ------------------------------------------------------------------
def reconstruct(
    projections: np.ndarray,
    angles: np.ndarray,
    config: Optional[ReconstructionConfig] = None,
    support=None,
    initial_object: Optional[np.ndarray] = None,
    fft_plan: Optional[RealFFT3D] = None,
    on_iteration: Optional[Callable[[int, np.ndarray], None]] = None,
    on_progress: Optional[Callable] = None,
) -> ReconstructionResult:
    """Run a GENFIRE reconstruction.

    Parameters
    ----------
    projections : (n0, n0, n_proj) real array of measured projections.
    angles : (n_proj, 3) Euler angles in degrees, GENFIRE convention.
    config : :class:`~genfire_cpu.config.ReconstructionConfig`.
    support : a :class:`~genfire_cpu.support.Support`, or an array to build one
        from, or None to take ``config.support_shape`` at the projection size.
    initial_object : optional starting guess, object-centred.
    on_iteration : called as ``f(iteration, object)`` after each iteration's
        constraints, with the object cropped to ``n0`` and *not* copied.  A
        caller that keeps it must copy.  Reconstructing in chunks through
        ``initial_object`` would give the same iterates but redo the gridding
        every time, and that is the expensive half of a short run.
    on_progress : called as ``f(iteration, metrics, get_volume)`` after each
        iteration, where ``metrics`` is a dict of this iteration's error (and
        R-free) and ``get_volume()`` returns the current object cropped to
        ``n0`` (not copied).  Used by the tutorials' live monitor
        (:class:`aet_tools.LiveMonitor`).
    fft_plan : optional pre-built :class:`~genfire_cpu.fft.RealFFT3D` to reuse
        across runs.  If it is pruned, its bounds must match the support's.
    """
    cfg = config or ReconstructionConfig()
    t_start = time.perf_counter()
    cpu_start = time.process_time()

    projections = np.asarray(projections)
    if projections.ndim == 2:
        projections = projections[:, :, None]
    n0 = int(projections.shape[0])
    if projections.shape[1] != n0:
        raise ValueError("projections must be square, got %r"
                         % (projections.shape[:2],))

    mem = cfg.validate(projections.shape)
    n = int(mem["padded_size"])
    nh = n // 2 + 1
    dtype, cdtype = cfg.dtype, cfg.complex_dtype
    crop = _centre_slice(n, n0)

    if cfg.verbose:
        print(cfg.describe(n0, num_projections=projections.shape[2]))
        print("-" * 60)

    # --- 1. grid the projections ----------------------------------------
    t0 = time.perf_counter()
    padded = pad_projections(projections, n, dtype)
    # sparse: the gridder hands back the filled points directly, so the dense
    # half-grid and its mask are never allocated -- 4.5 GiB at 1024**3,
    # 36 GiB at 2048**3 -- and the compression step below disappears
    measured_lin, measured_vals = fill_fourier_grid(
        padded, angles, n,
        interpolation_cutoff=cfg.interpolation_cutoff,
        grid_multiple=cfg.grid_multiple,
        enforce_resolution_circle=cfg.enforce_resolution_circle,
        dtype=dtype, chunk_bytes=cfg.grid_chunk_bytes, sparse=True,
        threads=cfg.grid_threads, verbose=cfg.verbose)
    del padded
    gridding_seconds = time.perf_counter() - t0

    # --- 2. the measured set, already compressed ---------------------------
    # only a few percent of the half-grid is ever measured (3.9% for a typical
    # tilt series at 1024**3).  Addressing those points directly turns three
    # full sweeps of a 4 GB spectrum per iteration into two gathers over a few
    # percent of it.
    needs_radius = cfg.calculate_rfree or cfg.resolution_mode != "none"
    q_at_measured = radial_at(measured_lin, n, nh, dtype) if needs_radius else None

    # fold in the half-box shift that takes the FFT-order grid to a centred
    # volume; magnitudes are untouched, so neither the error metric nor R-free
    # can notice
    flip = _halfgrid_parity(measured_lin, n, nh)
    np.negative(measured_vals, out=measured_vals, where=flip)
    del flip

    # --- 3. R-free: withhold a fraction of the measured points -----------
    rfree_idx: List[np.ndarray] = []
    rfree_vals: List[np.ndarray] = []
    shell_resolution = None

    if cfg.calculate_rfree:
        rng = np.random.default_rng(cfg.random_seed)
        n_shell = max(1, int(round(n0 / 2 / cfg.rfree_shell_pixels)))
        edges = np.linspace(0.0, 1.0, n_shell + 1)
        shell_of = np.clip(
            np.searchsorted(edges, q_at_measured, side="right") - 1,
            0, n_shell - 1)
        keep = np.ones(measured_lin.size, dtype=bool)
        for s in range(n_shell):
            members = np.flatnonzero(shell_of == s)
            if members.size == 0:
                rfree_idx.append(members)
                rfree_vals.append(np.empty(0, dtype=cdtype))
                continue
            take = int(np.floor(members.size * cfg.rfree_fraction))
            picked = members[rng.permutation(members.size)[:take]]
            rfree_idx.append(measured_lin[picked])
            rfree_vals.append(measured_vals[picked].copy())
            keep[picked] = False
        del shell_of
        # drop the withheld points from the set the loop enforces
        measured_lin = measured_lin[keep]
        measured_vals = measured_vals[keep]
        if q_at_measured is not None:
            q_at_measured = q_at_measured[keep]
        del keep
        with np.errstate(divide="ignore"):
            shell_resolution = 2.0 * cfg.pixel_size / np.maximum(edges[1:], 1e-9)

    measured_abs = np.abs(measured_vals)

    # --- 4. support -------------------------------------------------------
    if cfg.use_support:
        sup = build_support(n, n0, cfg, support)
    else:
        sup = Support(n, (0, 0, 0), (n, n, n), name="none")
    if cfg.verbose:
        print("support              : %r" % (sup,))

    # --- 5. FFT plan ------------------------------------------------------
    bounds = sup.bounds if (cfg.use_support and cfg.prune_fft) else None
    if fft_plan is None:
        if cfg.verbose:
            print("planning %d^3 %s FFT on %d threads ..."
                  % (n, dtype.name, cfg.threads))
        fft_plan = RealFFT3D(shape=(n, n, n), dtype=dtype, threads=cfg.threads,
                             planner=cfg.planner, in_place=cfg.in_place_fft,
                             bounds=bounds)
        if cfg.verbose:
            print("  %s" % fft_plan)
    elif fft_plan.shape != (n, n, n):
        raise ValueError("supplied fft_plan has shape %r, need %r"
                         % (fft_plan.shape, (n, n, n)))
    elif fft_plan.pruned and fft_plan.bounds != sup.bounds:
        raise ValueError("supplied fft_plan is pruned to %r but this support "
                         "needs %r" % (fft_plan.bounds, sup.bounds))
    planning_seconds = fft_plan.plan_seconds

    obj = fft_plan.real
    spec_flat = fft_plan.spectrum.reshape(-1)

    obj[...] = 0
    if initial_object is not None:
        initial_object = np.asarray(initial_object, dtype=dtype)
        obj[_centre_slice(n, initial_object.shape[0])] = initial_object

    # --- 6. iterate -------------------------------------------------------
    num_iter = int(cfg.num_iterations)
    error_k = np.zeros(num_iter, dtype=np.float64)
    rfree_total = np.zeros(num_iter) if cfg.calculate_rfree else None
    rfree_by_shell = (np.zeros((len(rfree_idx), num_iter))
                      if cfg.calculate_rfree else None)

    error_denom = float(measured_abs.sum())
    if error_denom == 0:
        raise ValueError("no measured Fourier components; check the projections")

    marks, cutoffs = _iteration_cutoffs(cfg.enforcement_cutoffs, num_iter)
    marks = np.append(marks, np.inf)
    cutoff_num = 0
    constraint_lin, constraint_vals = measured_lin, measured_vals
    best_err = np.inf

    # the loop's non-FFT work is a set of sweeps over disjoint ranges; NumPy
    # does them on one core and none of them saturates memory bandwidth there
    parallel = Parallel(cfg.loop_threads)
    error_pieces = split(measured_lin.size, ERROR_PARTITIONS)
    error_partials = np.zeros(len(error_pieces), dtype=np.float64)

    def apply_constraints() -> None:
        # clear_outside goes through the FFT so it also wipes the in-place
        # padding, which a pruned forward would otherwise read back as data
        fft_plan.clear_outside(sup.bounds, parallel)
        sup.apply_inside(obj[sup.box], cfg.positivity, parallel)

    # error_k[it] describes the object constrained at the end of iteration
    # it-1, so that object waits in `pending` until its error is known and is
    # then promoted by swapping buffers rather than by copying
    # v2's extrapolation needs the previous iterate's pre-constraint value.
    # Only over the support box: everything outside is zeroed by
    # apply_constraints on the very next line anyway, so extrapolating it
    # would be arithmetic on values that are about to be discarded.
    ds = float(getattr(cfg, "momentum", 0.0))
    dt = float(getattr(cfg, "fourier_relaxation", 0.0))
    if not 0.0 <= dt < 1.0:
        raise ValueError("fourier_relaxation must be in [0, 1), got %r" % dt)
    prev_v = np.empty_like(obj[sup.box]) if ds else None
    have_prev = False

    pending = best = None
    if cfg.track_best:
        pending = np.empty((n0, n0, n0), dtype=dtype)
        best = np.empty((n0, n0, n0), dtype=dtype)

    apply_constraints()
    if cfg.track_best:
        np.copyto(pending, obj[crop])

    if cfg.verbose and parallel.enabled:
        print("loop threads         : %d" % parallel.threads)
    if cfg.verbose:
        print("-" * 60)
        print("iterating ...")
    t0 = time.perf_counter()
    for it in range(1, num_iter + 1):
        if it == marks[cutoff_num]:
            cut = float(cutoffs[cutoff_num])
            if cut <= -1.0 or q_at_measured is None:   # enforce everything
                constraint_lin, constraint_vals = measured_lin, measured_vals
            else:
                # the resolution indicator is 1 - |k|; enforce where it exceeds cut
                sel = q_at_measured < (1.0 - cut)
                constraint_lin = measured_lin[sel]
                constraint_vals = measured_vals[sel]
                del sel
            cutoff_num += 1
            best_err = np.inf

        fft_plan.forward()                        # obj -> spectrum (aliased)

        error_k[it - 1] = _error(spec_flat, measured_lin, measured_abs,
                                 error_pieces, error_partials,
                                 parallel) / error_denom

        if cfg.track_best and error_k[it - 1] < best_err:
            best_err = error_k[it - 1]
            pending, best = best, pending         # promote without copying

        if cfg.calculate_rfree:
            num = den = 0.0
            for s, (sel, vals) in enumerate(zip(rfree_idx, rfree_vals)):
                if sel.size == 0:
                    rfree_by_shell[s, it - 1] = np.nan
                    continue
                a = np.abs(spec_flat[sel])
                b = np.abs(vals)
                nu = float(np.abs(a - b).sum())
                de = float(b.sum())
                rfree_by_shell[s, it - 1] = nu / de if de else np.nan
                num += nu
                den += de
            rfree_total[it - 1] = num / den if den else np.nan

        # Fourier constraint.  dt = 0 overwrites, which is the orthogonal
        # projection onto the measured set; dt > 0 keeps a fraction of the
        # current value and is no longer a projection.
        if dt:
            def _blend(sl, dt=dt):
                idx = constraint_lin[sl]
                spec_flat[idx] = (dt * spec_flat[idx]
                                  + (1.0 - dt) * constraint_vals[sl])
            parallel.map_slices(constraint_lin.size, _blend)
        else:
            parallel.map_slices(
                constraint_lin.size,
                lambda sl: spec_flat.__setitem__(constraint_lin[sl],
                                                 constraint_vals[sl]))

        fft_plan.backward()                       # spectrum -> obj (aliased)

        # x_{k+1} = v_k + ds (v_k - v_{k-1}).  The first iteration has no
        # predecessor, so it only records v_0.
        if ds:
            cur = obj[sup.box]
            if have_prev:
                delta = cur - prev_v
                np.copyto(prev_v, cur)
                cur += ds * delta
            else:
                np.copyto(prev_v, cur)
                have_prev = True

        apply_constraints()
        if on_iteration is not None:
            on_iteration(it, obj[crop])
        if cfg.track_best:
            parallel.map_slices(
                n0, lambda sl: np.copyto(pending[sl], obj[crop][sl]))

        if cfg.verbose and (it % max(1, cfg.progress_every) == 0 or it == num_iter):
            msg = "  iter %4d/%d   error = %.5f" % (it, num_iter, error_k[it - 1])
            if cfg.calculate_rfree:
                msg += "   R-free = %.5f" % rfree_total[it - 1]
            print(msg, flush=True)
        if on_progress is not None:
            metrics = {"error": float(error_k[it - 1])}
            if cfg.calculate_rfree:
                metrics["R-free"] = float(rfree_total[it - 1])
            on_progress(it, metrics, lambda: obj[crop])

    iteration_seconds = time.perf_counter() - t0
    parallel.close()

    volume = best if (cfg.track_best and np.isfinite(best_err)) else obj[crop].copy()
    result = ReconstructionResult(
        reconstruction=volume,
        error_k=error_k,
        rfree_total=rfree_total,
        rfree_by_shell=rfree_by_shell,
        shell_resolution=shell_resolution,
        padded_size=n,
        seconds=time.perf_counter() - t_start,
        cpu_seconds=time.process_time() - cpu_start,
        gridding_seconds=gridding_seconds,
        planning_seconds=planning_seconds,
        iteration_seconds=iteration_seconds,
        fft_work_fraction=fft_plan.work_fraction,
        support=sup,
        config=cfg,
    )
    if cfg.verbose:
        print("-" * 60)
        print(result.summary())
    return result
