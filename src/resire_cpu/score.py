"""Score a reconstruction against known atom positions, without charging it
for the pose it was never able to determine.

Single-axis tomography fixes the object only up to a rigid pose.  Alignment
matches every projection to a reference projection, so wherever that reference
sits becomes where everything sits; the centre-of-mass fit picks the other
axis the same way; and adding a constant to every tilt angle reconstructs the
true object rotated by that constant, fitting the data exactly as well.  None
of this is error.  A translated, slightly rotated reconstruction is a correct
reconstruction.

Sampling at the *true* coordinates is not invariant to any of it, and is
punishing about translation in particular -- measured on a 5 nm particle whose
atoms have a 1.5 px radius, a rigid shift of a quarter pixel costs 11% of the
recovered density, half a pixel 32%, one pixel 84%, and by 1.5 px there is
nothing left.  An earlier version of this work read a pipeline at 38% of
reference for exactly that reason: the alignment residuals had discounted the
constant, correctly, and then the score charged for it.

So fit the pose first, then measure.  What remains is the part of the
disagreement the reconstruction is actually responsible for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

__all__ = ["Pose", "fit_pose", "sample_atoms"]


@dataclass
class Pose:
    """The rigid pose that best puts ``positions`` onto a volume's density."""

    offset: np.ndarray       #: (3,) translation in voxels
    theta_deg: float         #: rotation about the tilt axis, degrees
    score: float             #: mean sampled intensity at the fitted pose
    raw_score: float         #: the same measured at the untouched positions

    @property
    def shift_px(self) -> float:
        return float(np.linalg.norm(self.offset))

    def summary(self) -> str:
        return ("pose: %.2f px translation (%.2f, %.2f, %.2f), %.3f deg; "
                "score %.2f vs %.2f unfitted"
                % (self.shift_px, *self.offset, self.theta_deg, self.score,
                   self.raw_score))


def _place(positions: np.ndarray, centre: np.ndarray, offset: np.ndarray,
           theta_deg: float) -> np.ndarray:
    """Rotate about the tilt axis through ``centre``, then translate."""
    p = positions - centre
    if theta_deg:
        t = np.deg2rad(theta_deg)
        c, s = np.cos(t), np.sin(t)
        # axis 1 is the tilt axis, so the rotation mixes axes 0 and 2
        p = np.column_stack([c * p[:, 0] + s * p[:, 2], p[:, 1],
                             -s * p[:, 0] + c * p[:, 2]])
    return p + centre + offset


def sample_atoms(volume: np.ndarray, positions: np.ndarray,
                 offset: Sequence[float] = (0.0, 0.0, 0.0),
                 theta_deg: float = 0.0, order: int = 1) -> np.ndarray:
    """Interpolate the volume at each position under the given pose."""
    centre = np.array(volume.shape, dtype=np.float64) / 2.0
    p = _place(np.asarray(positions, dtype=np.float64), centre,
               np.asarray(offset, dtype=np.float64), theta_deg)
    return map_coordinates(volume, p.T, order=order, mode="constant")


def fit_pose(volume: np.ndarray, positions: np.ndarray,
             weights: Optional[np.ndarray] = None, span: float = 3.0,
             step: float = 0.5, rotate: bool = True) -> Pose:
    """Find the pose maximising the mean density sampled at ``positions``.

    A coarse grid first, because the objective has a sharp maximum an atom
    wide and a local optimiser started at the wrong pixel will not find it,
    then Nelder-Mead to get off the grid.  Only translation is gridded; the
    angular gauge is small once the shifts are right, and is left to the
    local step.
    """
    pos = np.asarray(positions, dtype=np.float64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    def negative(params):
        v = sample_atoms(volume, pos, params[:3],
                         params[3] if rotate else 0.0)
        return -float(np.average(v, weights=w))

    raw = -negative(np.zeros(4))
    grid = np.arange(-span, span + 1e-9, step)
    best, best_p = raw, np.zeros(4)
    for dx in grid:
        for dy in grid:
            for dz in grid:
                p = np.array([dx, dy, dz, 0.0])
                s = -negative(p)
                if s > best:
                    best, best_p = s, p
    out = minimize(negative, best_p, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000})
    if -out.fun > best:
        best, best_p = -out.fun, out.x
    return Pose(offset=best_p[:3].copy(),
                theta_deg=float(best_p[3]) if rotate else 0.0,
                score=float(best), raw_score=float(raw))
