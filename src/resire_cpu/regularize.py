"""Total variation as a proximal step, for the reconstruction loop.

The L1 penalty already in place empties the ligand band almost completely --
error is exactly zero out to the 99th percentile there, against GENFIRE's
0.108 -- and that is where its 27-fold advantage in false peaks comes from.
What it also does is lose weak signal: soft-thresholding cuts by magnitude,
and a chain carbon is weak, so 6% of them go with the artefacts.

Total variation penalises roughness instead of magnitude, so a feature that is
weak but smooth survives it.  Whether that helps here is not obvious and is
the reason to measure: an isolated atom is nothing *but* gradient, so a
gradient penalty attacks the thing being looked for, and the staircasing that
motivates second-order schemes like TGV needs ramps this object does not have.

The prox has no closed form.  Chambolle's dual algorithm solves it by
iterating on a vector field, which is why this costs an inner loop where the
L1 version was one subtraction.  It also has to run in the parent: the
gradient couples neighbours along y, and the workers are split along y.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tv_prox", "tgv_prox"]


def _grad(u: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Forward differences, zero at the far face.  ``out`` is (3, *u.shape)."""
    out[0, :-1] = u[1:] - u[:-1]
    out[0, -1] = 0.0
    out[1, :, :-1] = u[:, 1:] - u[:, :-1]
    out[1, :, -1] = 0.0
    out[2, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    out[2, :, :, -1] = 0.0
    return out


def _div(p: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Backward differences, the negative adjoint of ``_grad``."""
    out[...] = p[0]
    out[1:] -= p[0, :-1]
    out[:, 1:] += p[1, :, 1:] - p[1, :, :-1]
    out[:, 0] += p[1, :, 0]
    out[:, :, 1:] += p[2, :, :, 1:] - p[2, :, :, :-1]
    out[:, :, 0] += p[2, :, :, 0]
    return out


def tv_prox(u: np.ndarray, weight: float, iterations: int = 10,
            state: dict | None = None) -> np.ndarray:
    """``argmin_v  0.5||v - u||^2 + weight * TV(v)``, by Chambolle's method.

    ``state`` carries the dual field between calls.  Warm-starting it is worth
    the memory: consecutive outer iterations solve nearly the same subproblem,
    and from the previous field a handful of inner steps suffice where a cold
    start would need many more.

    The step is fixed at 1/12, inside the 1/(2d) that guarantees convergence
    in three dimensions.
    """
    if weight <= 0:
        return u
    shape = u.shape
    if state is None:
        state = {}
    p = state.get("p")
    if p is None or p.shape[1:] != shape:
        p = np.zeros((3,) + shape, dtype=np.float32)
        state["p"] = p

    tau = 1.0 / 12.0
    div = np.empty(shape, dtype=np.float32)
    g = np.empty_like(p)
    for _ in range(max(1, int(iterations))):
        _div(p, div)
        div -= u / weight
        _grad(div, g)
        # p <- (p + tau g) / (1 + tau |g|), the projection onto the unit ball
        norm = np.sqrt((g * g).sum(axis=0))
        norm *= tau
        norm += 1.0
        p += tau * g
        p /= norm
    _div(p, div)
    return u - weight * div


def _sym_grad(w: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Symmetrised gradient of a vector field: the six independent components.

    ``out`` is (6, *shape): the three diagonal terms then the three off-
    diagonal ones, each of which appears twice in the tensor and is therefore
    weighted by sqrt(2) in the norm below.
    """
    out[0, :-1] = w[0, 1:] - w[0, :-1];            out[0, -1] = 0.0
    out[1, :, :-1] = w[1, :, 1:] - w[1, :, :-1];   out[1, :, -1] = 0.0
    out[2, :, :, :-1] = w[2, :, :, 1:] - w[2, :, :, :-1]
    out[2, :, :, -1] = 0.0
    # off-diagonals: (d_i w_j + d_j w_i) / 2
    out[3, :-1] = 0.5 * (w[1, 1:] - w[1, :-1]);    out[3, -1] = 0.0
    out[3, :, :-1] += 0.5 * (w[0, :, 1:] - w[0, :, :-1])
    out[4, :-1] += 0.5 * (w[2, 1:] - w[2, :-1])
    out[4, :, :, :-1] += 0.5 * (w[0, :, :, 1:] - w[0, :, :, :-1])
    out[5, :, :-1] += 0.5 * (w[2, :, 1:] - w[2, :, :-1])
    out[5, :, :, :-1] += 0.5 * (w[1, :, :, 1:] - w[1, :, :, :-1])
    return out


def _sym_div(q: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Negative adjoint of ``_sym_grad``; returns a vector field."""
    out[...] = 0.0
    out[0, 1:] += q[0, 1:] - q[0, :-1];   out[0, 0] += q[0, 0]
    out[1, :, 1:] += q[1, :, 1:] - q[1, :, :-1];  out[1, :, 0] += q[1, :, 0]
    out[2, :, :, 1:] += q[2, :, :, 1:] - q[2, :, :, :-1]
    out[2, :, :, 0] += q[2, :, :, 0]
    out[0, :, 1:] += 0.5 * (q[3, :, 1:] - q[3, :, :-1])
    out[1, 1:] += 0.5 * (q[3, 1:] - q[3, :-1])
    out[0, :, :, 1:] += 0.5 * (q[4, :, :, 1:] - q[4, :, :, :-1])
    out[2, 1:] += 0.5 * (q[4, 1:] - q[4, :-1])
    out[1, :, :, 1:] += 0.5 * (q[5, :, :, 1:] - q[5, :, :, :-1])
    out[2, :, 1:] += 0.5 * (q[5, :, 1:] - q[5, :, :-1])
    return out


def tgv_prox(u: np.ndarray, weight: float, iterations: int = 20,
             alpha1: float = 1.0, alpha0: float = 2.0,
             state: dict | None = None) -> np.ndarray:
    """Second-order total generalised variation, by Chambolle-Pock.

    TGV replaces ``|grad u|`` with an infimal convolution over an auxiliary
    field w:

        TGV(u) = min_w  alpha1 |grad u - w| + alpha0 |sym_grad w|

    Where the object is flat, w goes to zero and this is ordinary TV; where it
    ramps, w absorbs the slope and the penalty falls on the *curvature*
    instead.  That is what removes the staircasing TV produces on gradients,
    and it is why TGV is the usual second-order choice rather than a plain
    Hessian penalty, which would blur edges.

    Whether it helps here is a separate question.  The staircasing TGV cures
    needs ramps, and an object of isolated atoms on an empty background has
    none; what TV does to this object is not staircase it but erase it.  The
    measurement is the point.
    """
    if weight <= 0:
        return u
    shape = u.shape
    state = {} if state is None else state
    v = state.get("v")
    if v is None or v.shape != shape:
        v = u.astype(np.float32, copy=True)
        state.update(v=v, w=np.zeros((3,) + shape, np.float32),
                     p=np.zeros((3,) + shape, np.float32),
                     q=np.zeros((6,) + shape, np.float32))
    v, w, p, q = state["v"], state["w"], state["p"], state["q"]
    vb, wb = v.copy(), w.copy()

    tau, sigma = 1.0 / 12.0, 1.0 / 12.0
    gp = np.empty_like(p)
    gq = np.empty_like(q)
    div = np.empty(shape, np.float32)
    sdiv = np.empty_like(w)

    for _ in range(max(1, int(iterations))):
        # dual ascent, then project onto the balls the penalties define
        _grad(vb, gp)
        p += sigma * (gp - wb)
        n = np.maximum(1.0, np.sqrt((p * p).sum(axis=0)) / (weight * alpha1))
        p /= n
        _sym_grad(wb, gq)
        q += sigma * gq
        n = np.maximum(1.0, np.sqrt((q * q).sum(axis=0)) / (weight * alpha0))
        q /= n

        # primal descent; the data term's prox is a shrink toward u
        v_old, w_old = v.copy(), w.copy()
        _div(p, div)
        v += tau * div
        v += tau * u
        v /= 1.0 + tau
        _sym_div(q, sdiv)
        w += tau * (p + sdiv)

        vb[...] = 2.0 * v - v_old        # extrapolation, as the method requires
        wb[...] = 2.0 * w - w_old
    state.update(v=v, w=w, p=p, q=q)
    return v
