"""RESIRE: projected gradient descent on the real-space least-squares fit.

Unlike GENFIRE, this is an optimisation problem with an explicit objective:

    minimise  0.5 * ||M (A rho - b)||^2    subject to  rho >= 0

where A is the Radon operator, b the measured projections and M an optional
detector mask.  The gradient is ``A'M(A rho - b)``, so one iteration is a
forward projection, a back-projection of the residual, a step, and a clip.
There is no Fourier transform anywhere in the loop.

The consequence that matters for artefacts: starting from zero, the iterate
only ever moves within the range of ``A'``, so components of the volume that
the data cannot see stay at zero instead of being filled with whatever
satisfies a support constraint.  Positivity clipping does push the iterate out
of that range, so this is a tendency rather than a guarantee -- but it is the
opposite tendency to alternating projections, which give the unmeasured
directions their value on purpose.

The loop works in the operator's own flat layout and treats its buffer as the
iterate.  A first version passed 3D arrays across the boundary and paid two
cache-hostile transposes plus three full-volume temporaries per step; at
512**3 that measured 54% of the wall clock, an Amdahl ceiling of 1.8x however
many workers were added.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .progress import Progress
from .project import SingleAxisProjector
from .regularize import tgv_prox, tv_prox

__all__ = ["ReconstructionConfig", "ReconstructionResult", "reconstruct"]


@dataclass
class ReconstructionConfig:
    """Parameters of the descent."""

    #: Step length as a multiple of the stability limit's numerator.  The
    #: gradient of 0.5*||A rho - b||^2 is Lipschitz with constant ||A||^2,
    #: which for this geometry scales as (number of projections) x (ray
    #: length), so dividing by that puts step_size on a scale where the
    #: descent is stable below 2.  The reference implementation uses 2.
    step_size: float = 2.0
    num_iterations: int = 600
    #: "none" is plain projected gradient descent, the setting every
    #: measurement in this repository was made under.  "fista" adds Nesterov
    #: extrapolation, which for a convex objective with a projection step
    #: improves the rate from O(1/k) to O(1/k^2).
    #:
    #: It carries state between steps -- the previous iterate -- so a run
    #: resumed from a saved volume alone is not the run it claims to be.
    #: reconstruct() refuses that combination rather than quietly restarting
    #: the momentum.
    #:
    #: The mechanism argument survives it: the extrapolation is a linear
    #: combination of iterates that the same descent produced, and the clip
    #: still only raises negative values to zero.  Whether artefacts survive
    #: it is a separate question, and an empirical one -- GENFIRE v2's
    #: momentum made them worse at every dose tested.
    acceleration: str = "none"        # "none" | "fista"

    #: L1 penalty on the density, as a fraction of the value that would drive
    #: the solution to exactly zero.  Zero is off, which is what every
    #: measurement here was made under.
    #:
    #: Quoted relatively because the absolute penalty is not a portable
    #: number.  The data term sums over M*n*n projection pixels and the
    #: penalty over n**3 voxels, so their ratio moves with the grid and the
    #: projection count, and an absolute lambda has to be retuned for every
    #: configuration.  Holding lambda*dt fixed instead is not the invariant
    #: either -- lambda_max grows with the projection count while dt falls as
    #: 1/(M*n), so their product still carries a factor of n, and a 512**3
    #: condition set up that way ran at about twice the intended strength.
    #:
    #: lambda_max = max(A^T b) is the smallest penalty for which the proximal
    #: step zeroes everything, so it absorbs the data's magnitude, the grid
    #: and the projection count together and 0.01 means the same thing
    #: everywhere.  It costs one back-projection at start-up, against
    #: hundreds of iterations.
    #:
    #: The penalty is the natural prior for this object -- atoms are sparse,
    #: the background empty -- and answers acceleration's cost: FISTA reaches
    #: a lower residual than plain descent and produces more false peaks doing
    #: it, which is what happens when extrapolation puts amplitude into
    #: directions the data barely constrains.
    l1_relative: float = 0.0

    #: Total-variation penalty, on the same relative scale as l1_relative.
    #: Zero is off.  The two can be combined, but the point of having it is
    #: the case L1 handles badly: soft-thresholding removes weak signal along
    #: with the artefacts, and a chain carbon is weak, whereas TV penalises
    #: roughness and leaves a smooth weak feature alone.
    #:
    #: It costs an inner loop -- Chambolle's dual iteration, since the prox
    #: has no closed form -- and unlike the L1 step it cannot run in the
    #: workers, because the gradient couples voxels along y and that is the
    #: axis the volume is split on.
    tv_relative: float = 0.0
    tv_inner: int = 10               #: inner iterations per outer step

    #: "tv" is first order, "tgv" second.  TGV costs an auxiliary field and a
    #: saddle-point solve, and buys freedom from the staircasing TV produces
    #: on ramps -- which this object may not have any of.
    tv_order: str = "tv"             # "tv" | "tgv"

    #: Allow acceleration="fista" to begin from `initial`, restarting the
    #: momentum at t = 1 rather than continuing it.
    #:
    #: Resuming and restarting are different things and only one of them is a
    #: mistake.  Resuming means continuing a run that was interrupted, and the
    #: velocity is part of what was interrupted -- reconstruct() refuses that
    #: because a volume alone cannot carry it.  Restarting is deliberate: it
    #: is a standard technique for FISTA, and it is what makes a hybrid
    #: schedule expressible, plain descent first and acceleration after.
    fista_restart: bool = False
    positivity: bool = True
    #: Optional binary support, applied like positivity.  Off by default:
    #: leaving it off is what keeps the unmeasured directions at zero, and it
    #: is also what the reference implementation does -- its CUDA kernel's
    #: support argument is commented out and every shipped script constrains
    #: positivity alone.  Turning it on makes a comparison against GENFIRE
    #: like-for-like, at the cost of no longer being RESIRE.
    support: Optional[np.ndarray] = None
    #: Detector pixels to fit, as (n, n) or (n, n, n_proj); zero means "this
    #: pixel carries no usable measurement".  It multiplies the residual,
    #: which is exactly weighting those entries out of the least squares.  M
    #: is diagonal and symmetric, so the adjoint survives unchanged.
    #:
    #: A Fourier-gridding reconstructor cannot express the same thing: once a
    #: projection has been transformed every coefficient depends on every
    #: pixel, so zeroing a pixel asserts that it *is* zero rather than that it
    #: is unknown.
    detector_mask: Optional[np.ndarray] = None
    #: Keep the iterate with the lowest R factor rather than the last one.
    track_best: bool = True
    report_every: int = 50
    #: Append one JSON object per report to this file.  Printing alone is not
    #: enough: a pipe through ``tail`` buffers until the process exits, and a
    #: long run then has no observable state at all.
    progress_file: Optional[str] = None
    verbose: bool = True
    dtype: type = np.float32


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray
    r_factor: np.ndarray            #: L1 R factor per iteration
    loss: np.ndarray                #: the quantity actually minimised
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    iterations_run: int = 0
    best_iteration: int = -1
    workers: int = 1
    projector: Optional[SingleAxisProjector] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            "iterations           : %d" % self.iterations_run,
            "final R factor       : %.5f" % self.r_factor[-1],
            "best R factor        : %.5f  (iteration %d)"
            % (self.r_factor[self.best_iteration], self.best_iteration + 1),
            "final loss           : %.6e" % self.loss[-1],
            "wall clock           : %.1f s" % self.seconds,
        ]
        if self.cpu_seconds and self.seconds:
            busy = self.cpu_seconds / self.seconds
            lines.append("CPU time (this process): %.1f s  (%.1f cores busy)"
                         % (self.cpu_seconds, busy))
            # process_time() counts this process only.  With a worker pool the
            # arithmetic happens elsewhere, so a low ratio is expected and says
            # nothing about whether the machine was awake -- warning then would
            # train the reader to ignore a warning that exists to catch a
            # laptop sleeping through a timing run.
            if self.workers > 1:
                lines.append("  plus %d worker processes, whose CPU time is "
                             "not counted above" % self.workers)
            elif busy < 0.5:
                lines.append("  the process was idle for most of that wall "
                             "clock; if this is a laptop it slept, and the "
                             "timings above mean nothing")
        return "\n".join(lines)


def reconstruct(projections: np.ndarray, angles: np.ndarray,
                config: Optional[ReconstructionConfig] = None,
                projector: Optional[SingleAxisProjector] = None,
                initial: Optional[np.ndarray] = None,
                callback=None) -> ReconstructionResult:
    """Reconstruct ``(n, n, n_proj)`` projections taken at ``angles`` degrees.

    ``initial`` starts the descent from a given volume instead of from zero,
    which makes a long run resumable.  Resumption is exact rather than
    approximate: plain gradient descent carries no state between steps, so the
    iterate is the whole of it -- unlike a momentum method, where restarting
    from the volume alone would silently discard the velocity.

    ``callback``, if given, is called as ``f(iteration, metrics, get_volume)``
    after every iteration: ``metrics`` holds the R factor and the loss, and
    ``get_volume()`` returns the current iterate as an (n, n, n) array.  Used
    by the tutorials' live monitor (:class:`aet_tools.LiveMonitor`).

    Starting from zero is not merely a default.  It is what keeps the iterate
    in range(A^T), so a resumed run must continue from a volume the same
    descent produced, never from an arbitrary guess.
    """
    cfg = config or ReconstructionConfig()
    b = np.ascontiguousarray(projections, dtype=cfg.dtype)
    if b.ndim != 3:
        raise ValueError("projections must be (n, n, n_proj), got %r" % (b.shape,))
    n = b.shape[0]
    if b.shape[1] != n:
        raise ValueError("projections must be square, got %r" % (b.shape[:2],))

    t_start, cpu_start = time.perf_counter(), time.process_time()
    if projector is None:
        if cfg.verbose:
            print("building the projection operator ...", flush=True)
        projector = SingleAxisProjector(n, angles, dtype=cfg.dtype)
    if projector.num_projections != b.shape[2]:
        raise ValueError("projector has %d angles, projections have %d"
                         % (projector.num_projections, b.shape[2]))
    if cfg.verbose:
        print(projector, flush=True)

    support_flat = None
    if cfg.support is not None:
        support = np.ascontiguousarray(cfg.support)
        if support.shape != (n, n, n):
            raise ValueError("support must be (%d, %d, %d), got %r"
                             % (n, n, n, support.shape))
        support_flat = projector.flatten_volume(support.astype(cfg.dtype))

    mask3d = mask_flat = None
    if cfg.detector_mask is not None:
        mask3d = np.asarray(cfg.detector_mask, dtype=cfg.dtype)
        if mask3d.shape == (n, n):
            mask3d = mask3d[:, :, None]
        if mask3d.shape not in ((n, n, 1), b.shape):
            raise ValueError("detector_mask must be (%d, %d) or %r, got %r"
                             % (n, n, b.shape, mask3d.shape))
        mask3d = np.ascontiguousarray(np.broadcast_to(mask3d, b.shape))
        mask_flat = projector.flatten_projections(mask3d)

    # the R factor has to be normalised over the pixels actually being fitted,
    # or masking data silently improves the number it is reported against
    denom = float(np.abs(b if mask3d is None else b * mask3d).sum()) or 1.0

    # gradient step, scaled by the Lipschitz constant of the least-squares term
    dt = float(cfg.step_size / (projector.num_projections * n))
    if str(getattr(cfg, "acceleration", "none")).lower() == "fista":
        # Plain projected descent is stable for dt < 2/L; FISTA's proof wants
        # dt <= 1/L.  The default step was chosen for the former and sits
        # above 1/L -- an exact line search on the first step measured 0.6
        # times it -- so handing it to FISTA unchanged diverges, which is
        # exactly what a first version did.  Halving lands inside the bound
        # and, measured against plain descent, reaches a residual in 80
        # iterations that plain descent needs more than 320 for.
        dt *= 0.5

    l1_rel = float(getattr(cfg, "l1_relative", 0.0))
    if l1_rel < 0:
        raise ValueError("l1_relative must be >= 0, got %r" % l1_rel)
    if l1_rel and not cfg.positivity:
        # Soft-thresholding a signed variable is sign(x) max(|x|-t, 0); the
        # subtract-then-clip used here is only that operator on the
        # non-negative orthant.  Refusing beats silently applying the wrong
        # prox.
        raise ValueError("l1_relative requires positivity=True")

    # rho *is* the operator's buffer: nothing is copied in or out per step
    rho = projector.volume_buffer

    # lambda_max = max(A^T b) is the penalty at which the first proximal step
    # already zeroes the whole volume, so quoting the penalty against it makes
    # one number mean the same thing at any grid or projection count.
    # Measured before rho is filled, because the back-projection lands in that
    # same buffer -- doing it afterwards would need the iterate saved and put
    # back.
    tv_rel = float(getattr(cfg, "tv_relative", 0.0))
    if tv_rel < 0:
        raise ValueError("tv_relative must be >= 0, got %r" % tv_rel)
    if tv_rel and not cfg.positivity:
        raise ValueError("tv_relative requires positivity=True")
    tv_order = str(getattr(cfg, "tv_order", "tv")).lower()
    if tv_order not in ("tv", "tgv"):
        raise ValueError('tv_order must be "tv" or "tgv", got %r' % cfg.tv_order)

    threshold = 0.0
    tv_weight = 0.0
    tv_state: dict = {}
    if l1_rel or tv_rel:
        projector.projection_buffer[...] = projector.flatten_projections(b)
        projector.backward_inplace()
        lam_max = float(rho.max())
        threshold = l1_rel * lam_max * dt
        tv_weight = tv_rel * lam_max * dt

    if initial is None:
        rho[...] = 0
    else:
        init = np.asarray(initial, dtype=cfg.dtype)
        if init.shape == (n, n, n):
            init = projector.flatten_volume(init)
        if init.shape != rho.shape:
            raise ValueError("initial must be (n, n, n) or %r, got %r"
                             % (rho.shape, init.shape))
        rho[...] = init
    accel = str(getattr(cfg, "acceleration", "none")).lower()
    if accel not in ("none", "fista"):
        raise ValueError('acceleration must be "none" or "fista", got %r'
                         % cfg.acceleration)
    if (accel == "fista" and initial is not None
            and not getattr(cfg, "fista_restart", False)):
        # The extrapolation needs the previous iterate, and a checkpoint that
        # saved only the volume does not have it.  Silently restarting the
        # momentum would produce a run that is not the one it reports.
        raise ValueError("acceleration='fista' cannot resume from `initial`: "
                         "the previous iterate is part of the state and a "
                         "volume alone does not carry it")
    # y and rho differ under FISTA, so the previous iterate needs its own
    # array; one is enough, reused as scratch for the extrapolation.
    prev = np.empty_like(rho) if accel == "fista" else None
    if prev is not None:
        np.copyto(prev, rho)
    t_prev = 1.0

    resid = projector.projection_buffer
    b_flat = projector.flatten_projections(b)

    r_factor = np.zeros(cfg.num_iterations, dtype=np.float64)
    loss = np.zeros(cfg.num_iterations, dtype=np.float64)
    best_err, best, best_it = np.inf, None, -1

    prog = Progress(cfg.num_iterations, path=cfg.progress_file,
                    every=cfg.report_every,
                    stream=sys.stdout if cfg.verbose else None, label="resire")
    try:
        for it in range(cfg.num_iterations):
            projector.forward_inplace()              # resid <- A @ rho
            resid -= b_flat
            if mask_flat is not None:
                resid *= mask_flat
            r_factor[it] = float(np.abs(resid).sum()) / denom
            loss[it] = 0.5 * float(np.vdot(resid.ravel(), resid.ravel()))

            if cfg.track_best and r_factor[it] < best_err:
                best_err, best_it = r_factor[it], it
                if best is None:
                    best = rho.copy()
                else:
                    np.copyto(best, rho)

            projector.descend(dt, cfg.positivity, threshold)
            if tv_weight:
                # in the parent, and through the 3D shape: the gradient needs
                # neighbours along every axis, including the one the workers
                # are split on
                vol3 = projector.unflatten_volume(rho)
                prox = tgv_prox if tv_order == "tgv" else tv_prox
                vol3 = prox(vol3, tv_weight, cfg.tv_inner, state=tv_state)
                if cfg.positivity:
                    np.maximum(vol3, 0, out=vol3)
                rho[...] = projector.flatten_volume(vol3)
            if support_flat is not None:
                rho *= support_flat

            if prev is not None:
                # rho now holds x_k; prev holds x_{k-1}.  Nesterov:
                #     t_k = (1 + sqrt(1 + 4 t_{k-1}^2)) / 2
                #     y_k = x_k + (t_{k-1} - 1)/t_k * (x_k - x_{k-1})
                # Done through `prev` so no second full volume is needed: it
                # carries the difference, then the extrapolation is undone
                # from y to recover x_k for the next step.
                t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_prev * t_prev))
                beta = (t_prev - 1.0) / t
                t_prev = t
                np.subtract(rho, prev, out=prev)      # prev = x_k - x_{k-1}
                prev *= beta                          # prev = beta * delta
                rho += prev                           # rho  = y_k
                np.subtract(rho, prev, out=prev)      # prev = x_k

            prog.update(it + 1, R=r_factor[it])
            if callback is not None:
                cur = prev if prev is not None else rho     # x_k (prev holds it under FISTA)
                callback(it + 1, {"R factor": r_factor[it], "loss": loss[it]},
                         lambda cur=cur: projector.unflatten_volume(cur))
    finally:
        prog.close()

    # Under FISTA the buffer ends holding y, the extrapolated point, while
    # `prev` holds the last actual iterate.  Return the iterate.
    if prev is not None:
        rho = prev
    flat = best if (cfg.track_best and best is not None) else rho
    return ReconstructionResult(
        reconstruction=projector.unflatten_volume(flat),
        r_factor=r_factor,
        loss=loss,
        seconds=time.perf_counter() - t_start,
        cpu_seconds=time.process_time() - cpu_start,
        iterations_run=cfg.num_iterations,
        best_iteration=best_it if best_it >= 0 else cfg.num_iterations - 1,
        workers=getattr(projector, "workers", 1),
        projector=projector)
