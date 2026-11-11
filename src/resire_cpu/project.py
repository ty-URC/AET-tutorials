"""The single-axis Radon operator, as an explicit sparse matrix.

RESIRE descends a gradient, so the back-projection has to be the *exact*
adjoint of the forward projection.  Rotating a volume with an interpolating
resampler and calling the reverse rotation its adjoint does not satisfy that:
the two differ by the interpolation weights, and the resulting direction is
not the gradient of anything.  Building the operator once as a sparse matrix
makes the adjoint literally ``A.T`` and removes the question.

Single-axis geometry buys the decomposition that makes this affordable.  A
tilt about y never mixes y with anything, so every y slice is the same 2D
problem with the same matrix; ``A`` maps one (x, z) plane to one detector row
and is applied to all y at once as a matrix-matrix product.  For a 256**3
volume that is a 36k x 65k matrix rather than the 16M x 16M one the full 3D
problem would need.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .parallel import ParallelMatmul

__all__ = ["SingleAxisProjector"]

#: Sub-samples per voxel per axis, following the reference implementation's
#: ``pt_o_ratio = 4``: the source point is offset by +-0.25 in x and z, each
#: offset is projected separately, and the four are averaged.  Without it a
#: voxel lands in one detector bin and the projection aliases at high tilt.
SUBSAMPLES = ((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25))

#: The kernels ``kernel=`` selects.  All three are the same construction --
#: the Radon transform of the voxel basis, convolved with a detector response
#: -- and differ only in that response:
#:
#:   tent    T_theta * Lambda, the width-2 triangle, approximated by the four
#:           sub-samples above.  What the released implementation does.
#:   exact   T_theta itself.  The correct operator once the probe has been
#:           deconvolved out, because the detector then point-samples the
#:           Radon transform rather than integrating over a pixel.
#:   area    T_theta * Pi, the unit box.  Strip/area weighting, which is what
#:           SIRT implementations mean by the overlap of the projected voxel
#:           square with a detector bin.
#:
#: One caveat that is invisible until you look for it: sum_i K(p - i) is
#: exactly 1 for every sub-bin position p only when K-hat vanishes at 2*pi*m.
#: Lambda and Pi both do, so "tent" and "area" conserve each voxel's mass
#: exactly.  The bare trapezoid does not, so "exact" lets a voxel's total
#: deposited weight ripple with where it falls inside a bin.  Summed over a
#: lattice the ripple largely cancels, which is why "exact" still measures
#: best against a continuous phantom, but it is a real difference in A.
KERNELS = ("tent", "exact", "area")


def _uniform_sum_pdf(widths, x: np.ndarray) -> np.ndarray:
    """Density of a sum of independent Uniform(-w/2, w/2), at ``x``.

    Every kernel here is such a sum -- the projected offset inside a voxel is
    ``u cos(theta) - w sin(theta)`` with u, w uniform, and each detector
    response convolves in one more unit-width box -- so one closed form
    covers all of them.  The signed inclusion-exclusion below is the standard
    Irwin-Hall generalisation to unequal widths.
    """
    w = [float(v) for v in widths if float(v) > 1e-9]
    k = len(w)
    x = np.asarray(x, dtype=np.float64)
    if k == 0:
        return np.zeros_like(x)
    if k == 1:
        # t**0 is 1 even at t == 0, so the general branch would widen the box
        # by a bin at each end.  This is reached only at theta = 0 and 90.
        return np.where(np.abs(x) < w[0] / 2.0, 1.0 / w[0], 0.0)
    total, prod = sum(w), float(np.prod(w))
    out = np.zeros_like(x)
    for mask in range(1 << k):
        t = x + total / 2.0
        bits = 0
        for j in range(k):
            if mask >> j & 1:
                t = t - w[j]
                bits += 1
        np.maximum(t, 0.0, out=t)
        out += (-1.0 if bits & 1 else 1.0) * t ** (k - 1)
    return out / (math.factorial(k - 1) * prod)


class SingleAxisProjector:
    """Forward and back projection for a tilt series about the y axis.

    The volume is ``(n, n, n)`` in (x, y, z) order and object-centred.  A
    projection at angle theta sums along the rotated z direction, so a source
    point (x, z) lands at ``x cos(theta) - z sin(theta)`` on the detector.

    Parameters
    ----------
    n : edge length of the cubic volume.
    angles : (n_proj,) tilt angles in degrees.
    dtype : working precision of the matrix.
    """

    def __init__(self, n: int, angles: np.ndarray, dtype=np.float32,
                 workers: int = 1, build_adjoint: bool = True,
                 kernel: str = "tent", precondition: Optional[str] = None) -> None:
        if kernel not in KERNELS:
            raise ValueError("kernel must be one of %r, got %r" % (KERNELS, kernel))
        if precondition not in (None, "sirt"):
            raise ValueError("precondition must be None or 'sirt', got %r"
                             % (precondition,))
        if not build_adjoint and int(workers) > 1:
            # The pool shares both matrices; there is no adjoint to share, and
            # the empty stand-in below cannot be mapped.  Refusing beats
            # silently overriding the caller -- and costs nothing, since a
            # single forward projection does not repay a pool anyway: measured
            # at 256**3 over eleven candidates, 106 s serial against 430 s
            # with eight workers, before this option halved the serial time.
            raise ValueError("build_adjoint=False requires workers=1")
        self.n = int(n)
        self.angles = np.asarray(angles, dtype=np.float64).ravel()
        self.dtype = np.dtype(dtype)
        self.kernel = kernel
        self.precondition = precondition
        self.A = self._build() if kernel == "tent" else self._build_smooth()
        # SIRT is the same descent with two diagonal preconditioners,
        #     x <- x - C A' R (A x - b),   R = 1/rowsum,  C = 1/colsum,
        # and substituting x = C^(1/2) y turns it into plain descent on
        # A~ = R^(1/2) A C^(1/2).  Folding the scaling into the matrix here
        # means the iteration, the fused `descend` and the worker pool all
        # stay exactly as they are; the caller scales b on the way in and the
        # volume on the way out.  Positivity survives the substitution
        # because C^(1/2) is positive, so the two arms remain comparable.
        self.row_scale = self.col_scale = None
        if precondition == "sirt":
            eps = np.finfo(np.float64).tiny
            row = np.asarray(self.A.sum(axis=1)).ravel()
            col = np.asarray(self.A.sum(axis=0)).ravel()
            self.row_scale = (1.0 / np.sqrt(np.maximum(row, eps))).astype(self.dtype)
            self.col_scale = (1.0 / np.sqrt(np.maximum(col, eps))).astype(self.dtype)
            self.row_scale[row <= 0] = 0.0     # detector rows no voxel reaches
            self.col_scale[col <= 0] = 0.0     # voxels no projection sees
            self.A = (sp.diags(self.row_scale) @ self.A
                      @ sp.diags(self.col_scale)).tocsr().astype(self.dtype)
        # Both directions are used every iteration, so keep both in CSR.
        # Transposing an 11 M nonzero matrix is not cheap, though, and the
        # angular search only ever projects forward -- it scores a candidate,
        # it does not descend a gradient -- so it can skip this entirely.
        self.AT = self.A.T.tocsr() if build_adjoint else self.A[:0].T.tocsr()
        self.has_adjoint = bool(build_adjoint)
        self.workers = max(1, int(workers))
        self._mm = ParallelMatmul(self.A, self.AT, self.n, self.workers)

    # -- construction -----------------------------------------------------
    def _build(self) -> sp.csr_matrix:
        n = self.n
        centre = (n + 1) / 2.0 - 1.0        # 0-based index of the array centre
        ax = np.arange(n) - centre
        # column order is (x fastest within z), matching _flatten below
        zz, xx = np.meshgrid(ax, ax, indexing="ij")
        xx = xx.ravel()
        zz = zz.ravel()
        npts = xx.size
        col = np.arange(npts)

        blocks = []
        for theta in self.angles:
            c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
            rows, cols, vals = [], [], []
            for dx, dz in SUBSAMPLES:
                pos = (xx + dx) * c - (zz + dz) * s + centre
                lo = np.floor(pos)
                frac = pos - lo
                lo = lo.astype(np.int64)
                # linear interpolation splits each sample over two detector
                # bins; samples rotated off the detector simply drop out
                for idx, w in ((lo, 1.0 - frac), (lo + 1, frac)):
                    keep = (idx >= 0) & (idx < n) & (w > 0)
                    rows.append(idx[keep])
                    cols.append(col[keep])
                    vals.append(w[keep])
            blocks.append(sp.coo_matrix(
                (np.concatenate(vals) / len(SUBSAMPLES),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(n, npts)))
        return sp.vstack(blocks, format="csr", dtype=self.dtype)

    def _build_smooth(self) -> sp.csr_matrix:
        """``_build`` with the footprint integrated in closed form.

        The tent path above approximates the integral over the voxel with
        four sub-samples.  Here the whole kernel -- footprint and detector
        response together -- is evaluated exactly, so there is no quadrature
        error and no ``pt_o_ratio`` to choose.  The support is at most
        (cos + sin + 1)/2 < 1.21 bins wide, so four candidate bins per voxel
        cover it for every angle.
        """
        n = self.n
        centre = (n + 1) / 2.0 - 1.0
        ax = np.arange(n) - centre
        zz, xx = np.meshgrid(ax, ax, indexing="ij")
        xx = xx.ravel()
        zz = zz.ravel()
        npts = xx.size
        col = np.arange(npts)
        extra = [1.0] if self.kernel == "area" else []

        blocks = []
        for theta in self.angles:
            c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
            widths = [abs(c), abs(s)] + extra
            pos = xx * c - zz * s + centre
            base = np.floor(pos).astype(np.int64)
            rows, cols, vals = [], [], []
            for k in (-1, 0, 1, 2):
                idx = base + k
                w = _uniform_sum_pdf(widths, pos - idx)
                keep = (idx >= 0) & (idx < n) & (w > 0)
                rows.append(idx[keep])
                cols.append(col[keep])
                vals.append(w[keep])
            blocks.append(sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(n, npts)))
        return sp.vstack(blocks, format="csr", dtype=self.dtype)

    # -- shape helpers ----------------------------------------------------
    def _flatten(self, volume: np.ndarray) -> np.ndarray:
        """(n, n, n) in (x, y, z) -> (n*n, n) with y as the column index."""
        return np.ascontiguousarray(
            volume.transpose(2, 0, 1).reshape(self.n * self.n, self.n))

    def _unflatten(self, flat: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            flat.reshape(self.n, self.n, self.n).transpose(1, 2, 0))

    # -- the operator -----------------------------------------------------
    @property
    def num_projections(self) -> int:
        return self.angles.size

    def forward(self, volume: np.ndarray) -> np.ndarray:
        """(n, n, n) volume -> (n, n, n_proj) projections."""
        flat = self._mm.forward(
            self._flatten(volume).astype(self.dtype, copy=False))
        # a copy, not a view: with workers this is the shared buffer and
        # the next call overwrites it under the caller's feet
        return np.ascontiguousarray(
            flat.reshape(self.num_projections, self.n, self.n).transpose(1, 2, 0))

    def backward(self, projections: np.ndarray) -> np.ndarray:
        """(n, n, n_proj) projections -> (n, n, n) volume.  Exact adjoint."""
        self._require_adjoint()
        flat = np.ascontiguousarray(
            np.asarray(projections).transpose(2, 0, 1).reshape(
                self.num_projections * self.n, self.n))
        return self._unflatten(
            self._mm.backward(flat.astype(self.dtype, copy=False)))

    # -- flat-layout fast path --------------------------------------------
    # The 3D API above transposes on the way in and on the way out.  At
    # 512**3 that is two cache-hostile passes over 537 MB per call, and it was
    # measured as the dominant serial cost of a run.  Callers that iterate can
    # instead keep the volume in the operator's own buffer, in the layout the
    # matrix wants, and transpose exactly twice for the whole run.

    @property
    def volume_buffer(self) -> np.ndarray:
        """The working volume, shape ``(n*n, n)``.  Written in place."""
        return self._mm.vol

    @property
    def projection_buffer(self) -> np.ndarray:
        """The working projections, shape ``(n_proj*n, n)``.  Written in place."""
        return self._mm.proj

    def flatten_volume(self, volume: np.ndarray) -> np.ndarray:
        return self._flatten(volume)

    def unflatten_volume(self, flat: np.ndarray) -> np.ndarray:
        return self._unflatten(flat)

    def flatten_projections(self, projections: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(projections).transpose(2, 0, 1).reshape(
                self.num_projections * self.n, self.n))

    def unflatten_projections(self, flat: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            flat.reshape(self.num_projections, self.n, self.n).transpose(1, 2, 0))

    def forward_inplace(self) -> np.ndarray:
        """``projection_buffer <- A @ volume_buffer``."""
        return self._mm.forward_inplace()

    def backward_inplace(self) -> np.ndarray:
        """``volume_buffer <- A.T @ projection_buffer``."""
        self._require_adjoint()
        return self._mm.backward_inplace()

    def descend(self, dt: float, positivity: bool = True,
                threshold: float = 0.0) -> np.ndarray:
        """One gradient step, entirely inside the workers.

        ``projection_buffer`` must hold the residual.  Fusing the
        back-projection, the scaling, the subtraction and the clip keeps four
        full-volume passes off the parent.
        """
        return self._mm.descend(dt, positivity, threshold)

    # -- diagnostics ------------------------------------------------------
    def _require_adjoint(self) -> None:
        if not self.has_adjoint:
            raise RuntimeError(
                "this projector was built with build_adjoint=False; it can "
                "project forward but not back")

    def adjoint_error(self, seed: int = 0) -> float:
        """Relative violation of ``<Av, p> == <v, A'p>``.

        The one test that matters: a back-projection that is not the exact
        adjoint makes the descent direction not a gradient, and the failure is
        silent -- the reconstruction merely converges to the wrong thing.
        """
        self._require_adjoint()
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((self.n,) * 3).astype(self.dtype)
        p = rng.standard_normal(
            (self.n, self.n, self.num_projections)).astype(self.dtype)
        lhs = float(np.vdot(self.forward(v).ravel(), p.ravel()))
        rhs = float(np.vdot(v.ravel(), self.backward(p).ravel()))
        return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)

    def __repr__(self) -> str:
        return ("SingleAxisProjector(n=%d, %d projections, %.1fM nonzeros, "
                "%.0f MB, %d workers)"
                % (self.n, self.num_projections, self.A.nnz / 1e6,
                   (self.A.data.nbytes + self.A.indices.nbytes
                    + self.AT.data.nbytes + self.AT.indices.nbytes) / 2**20,
                   self.workers))

    def close(self) -> None:
        """Release the worker pool and its shared buffers."""
        self._mm.close()
