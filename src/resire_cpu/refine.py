"""Angular refinement: recover tilt angles that the microscope got wrong.

Measured tilt angles do not match the true orientations -- the GENFIRE paper
lists instrument misalignment, slipping, beam-induced motion, vibration and
thermal effects, and reports simulations starting from 2.1 degrees of average
error.  Neither released GENFIRE implements a correction: the Python and
MATLAB repositories have none, and the UCLA v2 archive ships the pieces but
the reconstructors never call them.  RESIRE's release does refine, with
``fmincon`` over the three Euler angles.

The loop is the one both papers describe: reconstruct, re-project the
reconstruction over a grid of candidate angles, keep whichever fits the
measured projection best, reconstruct again.

Two things make it cheaper than it looks.

The re-projection does **not** need an exact adjoint.  Exactness is required of
the back-projection because a gradient method's descent direction is defined
by it; here the projection only feeds a similarity score, and nothing is
differentiated through it.

And the candidates factorise.  Scoring offset ``d`` means projecting at
``theta_j + d`` for every j at once, which is one operator of exactly the same
size as the reconstruction's.  Building eleven of those costs a few seconds,
rather than building one operator per (projection, candidate) pair.

Only theta is refined here.  A tilt-axis misalignment lives in phi and psi and
needs a general 3D projector, which the single-axis operator cannot express;
that is a real gap, not a simplification of the error model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .project import SingleAxisProjector

__all__ = ["RefinementResult", "score_angles", "refine_theta",
           "refine_shifts", "shift_projections"]


@dataclass
class RefinementResult:
    angles: np.ndarray               #: refined theta, degrees
    initial: np.ndarray              #: what it started from
    shifts: np.ndarray               #: (rounds, n_proj) chosen angle offsets
    scores: np.ndarray               #: (rounds, n_proj) best score per round
    translations: np.ndarray = field(  #: (n_proj, 2) total dy, dx applied
        default_factory=lambda: np.zeros((0, 2)))
    projections: Optional[np.ndarray] = field(default=None, repr=False)
    seconds: float = 0.0
    truth: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = ["refinement rounds    : %d" % self.shifts.shape[0],
                 "total shift applied  : mean %.3f deg, max %.3f deg"
                 % (np.abs(self.angles - self.initial).mean(),
                    np.abs(self.angles - self.initial).max()),
                 "wall clock           : %.1f s" % self.seconds]
        if self.translations.size:
            r = np.hypot(*self.translations.T)
            lines.insert(2, "translation applied  : mean %.3f px, max %.3f px"
                         % (r.mean(), r.max()))
        if self.truth is not None:
            before = np.abs(self.initial - self.truth)
            after = np.abs(self.angles - self.truth)
            lines.append("error vs truth       : %.3f -> %.3f deg mean, "
                         "%.3f -> %.3f deg max"
                         % (before.mean(), after.mean(), before.max(),
                            after.max()))
        return "\n".join(lines)


def _parabolic(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Offset of a parabola's vertex through three equally spaced samples."""
    denom = lo - 2 * mid + hi
    return np.clip(np.where(np.abs(denom) > 1e-12, 0.5 * (lo - hi) / denom,
                            0.0), -0.5, 0.5)


def _ncc_over_shifts(measured: np.ndarray, calculated: np.ndarray):
    """Best cross-correlation over all translations, and where it occurred.

    Cross-correlation through the FFT searches every shift at once, which is
    why the reference prefers it over a real-space R factor: the R factor
    needs the two images aligned first, and that alignment is a search of its
    own.

    Returning the location matters as much as the score.  A first version took
    only the maximum, which makes the metric translation-invariant -- a
    misaligned tilt series scores just as well as an aligned one, so shift
    error is invisible to the search and survives into the reconstruction
    untouched.  Both stacks are ``(n, n, n_proj)``; the returned shift is what
    to apply to ``measured`` to bring it onto ``calculated``, refined to
    sub-pixel by a parabola through the correlation peak.
    """
    a = measured - measured.mean(axis=(0, 1), keepdims=True)
    b = calculated - calculated.mean(axis=(0, 1), keepdims=True)
    ny, nx = a.shape[:2]
    fa = np.fft.rfft2(a, axes=(0, 1))
    fb = np.fft.rfft2(b, axes=(0, 1))
    corr = np.fft.irfft2(fa * np.conj(fb), s=(ny, nx), axes=(0, 1))
    denom = np.linalg.norm(a, axis=(0, 1)) * np.linalg.norm(b, axis=(0, 1))
    denom[denom == 0] = np.inf

    flat = corr.reshape(ny * nx, -1)
    k = np.argmax(flat, axis=0)
    iy, ix = np.divmod(k, nx)
    j = np.arange(flat.shape[1])
    dy = iy + _parabolic(corr[(iy - 1) % ny, ix, j], corr[iy, ix, j],
                         corr[(iy + 1) % ny, ix, j])
    dx = ix + _parabolic(corr[iy, (ix - 1) % nx, j], corr[iy, ix, j],
                         corr[iy, (ix + 1) % nx, j])
    # the correlation is periodic: a peak past the midpoint is a shift back
    dy = np.where(dy > ny / 2, dy - ny, dy)
    dx = np.where(dx > nx / 2, dx - nx, dx)
    return flat[k, j] / denom, -dy, -dx


def shift_projections(projections: np.ndarray, dy: np.ndarray,
                      dx: np.ndarray) -> np.ndarray:
    """Translate each projection by a sub-pixel amount, via a phase ramp."""
    ny, nx = projections.shape[:2]
    f = np.fft.rfft2(projections, axes=(0, 1))
    ky = np.fft.fftfreq(ny)[:, None, None]
    kx = np.fft.rfftfreq(nx)[None, :, None]
    f *= np.exp(-2j * np.pi * (ky * np.asarray(dy) + kx * np.asarray(dx)))
    return np.fft.irfft2(f, s=(ny, nx), axes=(0, 1)).astype(
        projections.dtype, copy=False)


def score_angles(volume: np.ndarray, measured: np.ndarray, theta: np.ndarray,
                 offsets: Sequence[float], workers: int = 1,
                 dtype=np.float32) -> np.ndarray:
    """Score and best shift for each ``(projection, offset)`` pair.

    Each returned array is ``(n_offsets, n_projections)``.

    ``workers`` defaults to serial, and should usually stay there.  Each
    candidate needs its own matrix, so a process pool would be built and torn
    down once per candidate; on Windows that spawn cost swamps the one
    matrix-vector product it buys.  Measured at 256**3 over eleven candidates:
    106 s serial against 430 s with eight workers.
    """
    n = volume.shape[0]
    out = np.empty((len(offsets), theta.size), dtype=np.float64)
    sy = np.empty_like(out)
    sx = np.empty_like(out)
    for i, d in enumerate(offsets):
        P = SingleAxisProjector(n, np.asarray(theta) + d, dtype=dtype,
                                workers=workers, build_adjoint=False)
        try:
            out[i], sy[i], sx[i] = _ncc_over_shifts(measured, P.forward(volume))
        finally:
            P.close()
    return out, sy, sx


def _subgrid(table: np.ndarray, best: np.ndarray,
             offsets: np.ndarray) -> np.ndarray:
    """Parabola through the best grid point and its two neighbours.

    The score is smooth in the angle, so the maximum generally falls between
    samples.  Interior maxima only; a best value at either end of the search
    has no bracket and stays on the grid.
    """
    n = offsets.size
    out = offsets[best].astype(np.float64)
    interior = (best > 0) & (best < n - 1)
    if not np.any(interior):
        return out
    j = np.flatnonzero(interior)
    b = best[j]
    lo = table[b - 1, j]
    mid = table[b, j]
    hi = table[b + 1, j]
    denom = lo - 2 * mid + hi
    step = np.diff(offsets).mean()
    frac = np.where(np.abs(denom) > 1e-12, 0.5 * (lo - hi) / denom, 0.0)
    out[j] += np.clip(frac, -0.5, 0.5) * step
    return out


def refine_theta(projections: np.ndarray, theta: np.ndarray,
                 reconstruct_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 offsets: Optional[Sequence[float]] = None,
                 rounds: int = 3, workers: int = 1, dtype=np.float32,
                 subgrid: bool = True, translations: bool = True,
                 tol: float = 0.02,
                 truth: Optional[np.ndarray] = None,
                 verbose: bool = True) -> RefinementResult:
    """Alternate reconstruction and a grid search over per-projection tilt.

    ``reconstruct_fn(projections, theta) -> volume`` keeps this independent of
    which reconstructor is being refined, so the same loop can be wrapped
    around GENFIRE or RESIRE and the comparison stays fair.

    ``offsets`` defaults to the reference's search: +-5 degrees in steps of 1.
    The grid is re-centred on the current estimate each round, so successive
    rounds can travel further than one grid width.

    ``translations`` also aligns the projections.  The cross-correlation
    already searches every shift, so leaving this off makes the score blind to
    misalignment -- it scores an unaligned series exactly as well -- while the
    reconstruction still sees the unaligned data.

    ``tol`` stops early once no projection moves by more than this many
    degrees.

    ``subgrid`` interpolates between grid points.  Without it the residual
    error is the grid's own quantisation -- a 1 degree step leaves 1/sqrt(12)
    = 0.29 degrees of RMS no matter how good the score is, which is exactly
    where a first version stopped.
    """
    if offsets is None:
        offsets = np.arange(-5.0, 5.001, 1.0)
    offsets = np.asarray(offsets, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).copy()
    initial = theta.copy()
    shifts = np.zeros((rounds, theta.size))
    scores = np.zeros((rounds, theta.size))

    proj = np.array(projections, dtype=dtype, copy=True)
    total_dy = np.zeros(theta.size)
    total_dx = np.zeros(theta.size)

    t0 = time.perf_counter()
    for r in range(rounds):
        volume = reconstruct_fn(proj, theta)
        table, sy, sx = score_angles(volume, proj, theta, offsets,
                                     workers=1, dtype=dtype)
        best = np.argmax(table, axis=0)
        j = np.arange(theta.size)
        scores[r] = table[best, j]
        shifts[r] = (_subgrid(table, best, offsets) if subgrid
                     else offsets[best])
        theta = theta + shifts[r]
        if translations:
            # the shift belongs to the angle that won, not to the grid centre
            dy, dx = sy[best, j], sx[best, j]
            proj = shift_projections(proj, dy, dx)
            total_dy += dy
            total_dx += dx
        if verbose:
            msg = ("round %d/%d  moved mean %.3f deg, max %.3f deg  "
                   "(score %.5f)" % (r + 1, rounds, np.abs(shifts[r]).mean(),
                                     np.abs(shifts[r]).max(), scores[r].mean()))
            if truth is not None:
                msg += "  |  error vs truth %.3f deg" % np.abs(
                    theta - np.asarray(truth)).mean()
            print(msg, flush=True)
        # Sub-grid interpolation almost never returns exactly zero, so an
        # equality test here would silently disable the early stop and spend
        # full rounds moving angles by thousandths of a degree.
        if (np.abs(shifts[r]).max() < tol
                and (not translations
                     or np.hypot(dy, dx).max() < 0.05)):
            shifts, scores = shifts[:r + 1], scores[:r + 1]
            break

    return RefinementResult(angles=theta, initial=initial, shifts=shifts,
                            scores=scores,
                            translations=np.column_stack([total_dy, total_dx]),
                            projections=proj if translations else None,
                            seconds=time.perf_counter() - t0,
                            truth=None if truth is None
                            else np.asarray(truth, dtype=np.float64))


def refine_shifts(projections: np.ndarray, theta: np.ndarray,
                  reconstruct_fn, rounds: int = 3, workers: int = 1,
                  dtype=np.float32, tol: float = 0.01, verbose: bool = True):
    """Translation only, against a reconstruction, with the angles frozen.

    The last stage of alignment.  Centre of mass and common line each reduce a
    projection to one number -- a first moment, a 1D profile -- and stop where
    that summary stops being informative; matching against a reconstruction
    uses the whole image.

    Freezing the angles is what makes it safe.  Searching shift and angle
    together lets a translation be explained as a rotation: measured on this
    code, a 2 px shift error and no angle error at all came back as a 1.0
    degree angle error.  With the angles fixed there is nothing for the shift
    to hide in.

    Returns ``(projections, dy, dx)`` -- the aligned stack and the cumulative
    shift applied to it.
    """
    proj = np.array(projections, dtype=dtype, copy=True)
    n = proj.shape[0]
    total_dy = np.zeros(proj.shape[2])
    total_dx = np.zeros(proj.shape[2])
    for r in range(rounds):
        volume = reconstruct_fn(proj, theta)
        # forward only, so serial and without the adjoint: see score_angles
        P = SingleAxisProjector(n, theta, dtype=dtype, workers=1,
                                build_adjoint=False)
        try:
            _, dy, dx = _ncc_over_shifts(proj, P.forward(volume))
        finally:
            P.close()
        proj = shift_projections(proj, dy, dx)
        total_dy += dy
        total_dx += dx
        moved = float(np.hypot(dy, dx).max())
        if verbose:
            print("  shift round %d/%d  moved max %.4f px" % (r + 1, rounds,
                                                              moved), flush=True)
        if moved < tol:
            break
    return proj, total_dy, total_dx
