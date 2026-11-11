"""Support-film background: estimate it under the particle and subtract it.

A particle on a carbon film is never in vacuum.  The film adds counts to every
pixel, including the ones behind the particle, and those counts are not part
of the object being reconstructed: left in, they bias every centre of mass
toward the middle of the field and give the reconstruction a slab of density
it has no room for.  They have to come off first.

Outside the particle the film is measured directly.  Under it, it is not, and
has to be inferred.  The standard answer (the Miao group's AET pipelines) is
to assume the background is smooth there and interpolate it harmonically:
solve Laplace's equation inside the particle's footprint with the measured
background as the boundary value.  A harmonic function has no interior
extremum, so the estimate cannot invent a bump or a dip that the boundary
does not imply -- the least presumptuous smooth fill there is.

What it cannot recover is the film's texture under the particle: amorphous
carbon is atoms, and at 0.3 A per pixel its granularity is resolved.  Only the
smooth part comes off; the granular residue stays in the data as structured
noise, exactly as it does in an experiment.

The footprint itself is found from the data: smooth, threshold above the
film level, keep the largest connected region, fill it and dilate by a
margin so the boundary samples film and not the particle's faint edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

__all__ = ["BackgroundResult", "particle_mask", "laplace_fill",
           "subtract_background"]


@dataclass
class BackgroundResult:
    """Background-subtracted stack and what was taken off."""

    projections: np.ndarray     #: (n, n, n_proj) data minus background
    background: np.ndarray      #: (n, n, n_proj) the estimate removed
    mask: np.ndarray            #: (n, n, n_proj) bool, particle footprint


def particle_mask(image: np.ndarray, smooth_px: float = 3.0,
                  threshold: float = 0.1, margin_px: int = 8) -> np.ndarray:
    """Footprint of the particle in one projection.

    The smoothed image is thresholded ``threshold`` of the way from the film
    level (median of the field) to a robust particle level (99th percentile),
    so the cut adapts to dose and to the film thickening at high tilt.

    The cut has to be low.  A sphere is thin at its rim, so its projection
    fades out toward the edge; at 0.5 the mask stopped short of the rim, the
    harmonic fill took its boundary values from the particle, and the film
    was overestimated by a third of the particle's own total.  At 0.1 with an
    8 px margin no particle pixel is left outside at any dose tried (30-300)
    and the residual bias is about 0.5% of the particle total.
    """
    s = ndi.gaussian_filter(np.asarray(image, np.float64), smooth_px)
    lo, hi = np.median(s), np.percentile(s, 99.0)
    m = s > lo + threshold * (hi - lo)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(image.shape, bool)
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    return ndi.binary_dilation(m, iterations=margin_px)


def laplace_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace ``values`` inside ``mask`` by the harmonic interpolant.

    Solves the 5-point discrete Laplace equation on the masked pixels with the
    unmasked neighbours as Dirichlet data.  The mask must not touch the image
    border (it is a particle footprint, so it does not).
    """
    out = np.asarray(values, np.float64).copy()
    idx = np.full(mask.shape, -1, np.int64)
    ii, jj = np.nonzero(mask)
    n = ii.size
    if n == 0:
        return out
    idx[ii, jj] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    rhs = np.zeros(n)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        inside = idx[ni, nj] >= 0
        rows.append(np.arange(n)[inside])
        cols.append(idx[ni[inside], nj[inside]])
        vals.append(np.full(inside.sum(), -1.0))
        rhs[~inside] += out[ni[~inside], nj[~inside]]
    A = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    out[ii, jj] = spsolve(A, rhs)
    return out


def subtract_background(projections: np.ndarray, smooth_px: float = 3.0,
                        threshold: float = 0.1, margin_px: int = 8,
                        bg_smooth_px: float = 4.0) -> BackgroundResult:
    """Estimate the film behind each projection and subtract it.

    Outside the footprint the background is the data smoothed by
    ``bg_smooth_px`` -- only the film's slowly varying part is claimed as
    background; its atomic granularity is noise and stays.  Inside, it is the
    harmonic fill of that.  Nothing is clipped: negative values after
    subtraction are the zero-mean noise they look like, and clipping them
    would bias the particle's total upward.
    """
    p = np.asarray(projections, np.float64)
    bg = np.empty_like(p)
    masks = np.empty(p.shape, bool)
    for k in range(p.shape[2]):
        m = particle_mask(p[..., k], smooth_px, threshold, margin_px)
        # smooth the film without letting the particle bleed into the
        # boundary values: normalised convolution over the unmasked pixels
        w = (~m).astype(np.float64)
        num = ndi.gaussian_filter(p[..., k] * w, bg_smooth_px)
        den = ndi.gaussian_filter(w, bg_smooth_px)
        s = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        bg[..., k] = laplace_fill(s, m)
        masks[..., k] = m
    return BackgroundResult((p - bg).astype(np.float32),
                            bg.astype(np.float32), masks)
