"""Real-space supports.

A support does three jobs here.  It constrains the reconstruction, it gives
the FFT a bounding box to prune against (see :mod:`.fft`), and -- if it has a
soft edge -- it keeps the object from ending in a step discontinuity.

That last one matters more than it looks.  A hard-edged support makes the
object jump to zero across one voxel, and the transform of a step is a
sin(x)/x tail: energy smeared across the whole Fourier volume, strongest along
the three axes normal to the faces of a cube.  Every iteration then fights
that ringing against the measured data.  Tapering the last few voxels to zero
removes the discontinuity, and a sphere removes the corners, which are where a
cubic support rings worst.

Volumes here are **centred in the array**, not in FFT order: the object sits
in the middle, so a support is one contiguous block rather than eight corner
blocks.  :mod:`.reconstruct` works in that convention throughout.

The mask, when there is one, is stored over the bounding box only -- at 2x
oversampling that is an eighth of the grid, so a 1024**3 reconstruction carries
a 512**3 mask (0.5 GB in float32) instead of a 4 GB one.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["Support", "BoxSupport", "MaskedSupport", "centre_offset",
           "box_support", "sphere_support", "support_from_array",
           "raised_cosine_ramp"]


def centre_offset(n: int, width: int) -> int:
    """Where a ``width``-wide window starts if it is centred on the FFT origin.

    Centred on ``n // 2``, not on ``(n - width) // 2``.  The two agree for even
    widths and differ by one voxel for odd ones, where the floor division puts
    a width-63 box at [-32, +30] about the origin instead of [-31, +31].
    Measured cost of getting that wrong: correlation with the ground truth
    0.931 at odd projection sizes against 0.968 at even ones.
    """
    return n // 2 - width // 2


def raised_cosine_ramp(width: int, dtype=np.float32) -> np.ndarray:
    """A length-``width`` ramp rising 0 -> 1 with zero slope at both ends.

    Half a Hann window.  The half-sample offset keeps the first entry off
    exact zero, so no voxel of the taper is wasted.
    """
    if width <= 0:
        return np.ones(0, dtype=dtype)
    t = (np.arange(width, dtype=np.float64) + 0.5) / width
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(dtype)


class Support:
    """A support described by its bounding box; hard-edged by default."""

    def __init__(self, n: int, lo: Sequence[int], hi: Sequence[int],
                 name: str = "box") -> None:
        self.n = int(n)
        self.lo = tuple(int(v) for v in lo)
        self.hi = tuple(int(v) for v in hi)
        self.name = name
        if any(a < 0 or b > self.n or a >= b for a, b in zip(self.lo, self.hi)):
            raise ValueError("bad support bounds %r..%r on a %d grid"
                             % (self.lo, self.hi, self.n))

    # -- geometry ---------------------------------------------------------
    @property
    def bounds(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(zip(self.lo, self.hi))

    @property
    def box(self) -> Tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.lo, self.hi))

    @property
    def extent(self) -> Tuple[int, int, int]:
        return tuple(b - a for a, b in zip(self.lo, self.hi))

    @property
    def is_whole_grid(self) -> bool:
        return self.lo == (0, 0, 0) and self.hi == (self.n,) * 3

    @property
    def box_fraction(self) -> float:
        """Share of the grid the bounding box covers.

        Multiplied in python ints rather than through ``np.prod``, which
        infers int32 from a tuple on Windows and wraps above 1290**3.
        """
        a, b, c = self.extent
        return (a * b * c) / float(self.n ** 3)

    # -- application ------------------------------------------------------
    def zero_outside(self, obj: np.ndarray) -> None:
        """Zero everything outside the bounding box, with slice writes."""
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        if x0:
            obj[:x0] = 0
        if x1 < self.n:
            obj[x1:] = 0
        if y0:
            obj[x0:x1, :y0] = 0
        if y1 < self.n:
            obj[x0:x1, y1:] = 0
        if z0:
            obj[x0:x1, y0:y1, :z0] = 0
        if z1 < self.n:
            obj[x0:x1, y0:y1, z1:] = 0

    def apply_inside(self, sub: np.ndarray, positivity: bool = True,
                     parallel=None) -> None:
        """Constrain the bounding-box sub-volume, in place.

        ``parallel`` splits the work along the first axis.  Every voxel is
        handled independently, so the split cannot change the result.
        """
        if not positivity:
            return
        if parallel is None or not parallel.enabled:
            np.maximum(sub, 0, out=sub)
        else:
            parallel.map_slices(
                sub.shape[0], lambda s: np.maximum(sub[s], 0, out=sub[s]))

    def apply(self, obj: np.ndarray, positivity: bool = True,
              parallel=None) -> None:
        self.zero_outside(obj)
        self.apply_inside(obj[self.box], positivity, parallel)

    # -- introspection ----------------------------------------------------
    def as_array(self, dtype=np.float32) -> np.ndarray:
        """The full ``n**3`` weight volume.  For tests and for inspection."""
        out = np.zeros((self.n,) * 3, dtype=dtype)
        out[self.box] = 1
        self.apply_inside(out[self.box], positivity=False)
        return out

    def __repr__(self) -> str:
        return ("%s(n=%d, extent=%r, box=%.1f%% of grid)"
                % (type(self).__name__, self.n, self.extent,
                   100 * self.box_fraction))


class BoxSupport(Support):
    """A centred cube, optionally tapered over its last ``soft_edge`` voxels.

    The taper is separable -- the product of one 1D ramp per axis -- so it
    costs three short vectors and touches only the six edge slabs.  For a 512
    box with a 4-voxel edge that is under 5% of the box, versus a full
    multiply by an ``n**3`` mask.
    """

    def __init__(self, n: int, width: int, soft_edge: float = 0.0) -> None:
        lo = centre_offset(n, width)
        hi = lo + width
        super().__init__(n, (lo,) * 3, (hi,) * 3, name="box")
        self.width = int(width)
        self.soft_edge = int(round(soft_edge))
        if self.soft_edge * 2 > min(self.extent):
            raise ValueError("soft_edge %d does not fit in a %d-voxel box"
                             % (self.soft_edge, min(self.extent)))
        self._ramp = raised_cosine_ramp(self.soft_edge)

    def apply_inside(self, sub: np.ndarray, positivity: bool = True,
                     parallel=None) -> None:
        Support.apply_inside(self, sub, positivity, parallel)
        e = self.soft_edge
        if not e:
            return
        # the taper touches six thin slabs, a few percent of the box; not
        # worth splitting, and the axis-0 slabs could not be split anyway
        ramp = self._ramp.astype(sub.dtype, copy=False)
        for axis in range(3):
            shape = [1, 1, 1]
            shape[axis] = e
            low = [slice(None)] * 3
            low[axis] = slice(0, e)
            sub[tuple(low)] *= ramp.reshape(shape)
            high = [slice(None)] * 3
            high[axis] = slice(sub.shape[axis] - e, sub.shape[axis])
            sub[tuple(high)] *= ramp[::-1].reshape(shape)

    def __repr__(self) -> str:
        return ("BoxSupport(n=%d, width=%d, soft_edge=%d, box=%.1f%% of grid)"
                % (self.n, self.width, self.soft_edge, 100 * self.box_fraction))


class MaskedSupport(Support):
    """An arbitrary support, stored as a weight array over its bounding box."""

    def __init__(self, n: int, lo: Sequence[int], weights: np.ndarray,
                 name: str = "mask") -> None:
        weights = np.asarray(weights)
        hi = [a + s for a, s in zip(lo, weights.shape)]
        super().__init__(n, lo, hi, name=name)
        self.weights = weights

    def apply_inside(self, sub: np.ndarray, positivity: bool = True,
                     parallel=None) -> None:
        if parallel is None or not parallel.enabled:
            if positivity:
                np.maximum(sub, 0, out=sub)
            sub *= self.weights
            return

        def piece(s):
            if positivity:
                np.maximum(sub[s], 0, out=sub[s])
            sub[s] *= self.weights[s]

        parallel.map_slices(sub.shape[0], piece)

    @property
    def nbytes(self) -> int:
        return self.weights.nbytes

    def __repr__(self) -> str:
        return ("MaskedSupport(%s, n=%d, extent=%r, %s, box=%.1f%% of grid)"
                % (self.name, self.n, self.extent, self.weights.dtype.name,
                   100 * self.box_fraction))


# -- constructors ----------------------------------------------------------
def box_support(n: int, width: int, soft_edge: float = 0.0) -> BoxSupport:
    """A centred cube of side ``width``, tapered over ``soft_edge`` voxels."""
    return BoxSupport(n, width, soft_edge)


def sphere_support(n: int, diameter: float, soft_edge: float = 0.0,
                   dtype=np.float32) -> MaskedSupport:
    """A centred ball of the given diameter in voxels.

    A ball has no corners, which is where a cubic support rings hardest, and
    it matches the shape of most tomography specimens.  With ``soft_edge`` the
    last few voxels of the radius are tapered instead of cut.
    """
    radius = diameter / 2.0
    width = int(2 * np.ceil(radius + max(soft_edge, 0.0)))
    width = min(width + (width % 2), n)
    lo = centre_offset(n, width)

    # built one plane at a time: the obvious three-way broadcast would make
    # several float64 arrays of width**3, which is 1.1 GB each at width=520
    weights = np.empty((width,) * 3, dtype=dtype)
    ax = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    yz = ax[:, None] ** 2 + ax[None, :] ** 2
    for i in range(width):
        r = np.sqrt(ax[i] ** 2 + yz)
        if soft_edge > 0:
            t = np.clip((radius - r) / float(soft_edge), 0.0, 1.0, out=r)
            np.cos(np.pi * t, out=t)
            t *= -0.5
            t += 0.5
            weights[i] = t
        else:
            weights[i] = r <= radius
    return MaskedSupport(n, (lo,) * 3, weights, name="sphere")


def support_from_array(n: int, array: np.ndarray, soft_edge: float = 0.0,
                       threshold: float = 0.0, dtype=np.float32
                       ) -> MaskedSupport:
    """Wrap a user-supplied support, cropped to its bounding box.

    ``array`` may be the projection-sized volume or the full padded grid; it
    is centred either way.  With ``soft_edge`` the mask is tapered inwards
    from its own boundary using an exact distance transform, so the edge width
    is in voxels regardless of the shape.

    The distance transform is the expensive part -- it works in float64 over
    the bounding box, so budget roughly 8 bytes per voxel of it and a few tens
    of seconds at 512**3.  It runs once, before the iteration loop.
    """
    array = np.asarray(array)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError("support must be a cubic 3D array, got %r"
                         % (array.shape,))
    mask = array > threshold
    m = mask.shape[0]
    if m > n:
        raise ValueError("support of size %d does not fit a %d grid" % (m, n))
    if m < n:                                   # centre it on the padded grid
        full = np.zeros((n,) * 3, dtype=bool)
        off = centre_offset(n, m)
        full[off:off + m, off:off + m, off:off + m] = mask
        mask = full

    if not mask.any():
        raise ValueError("support is empty")
    idx = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != a)))
           for a in range(3)]
    lo = [int(i[0]) for i in idx]
    hi = [int(i[-1]) + 1 for i in idx]

    if soft_edge > 0:
        from scipy.ndimage import distance_transform_edt

        # pad by one so a mask touching the box wall still has a boundary to
        # measure distance from
        pad = 1
        sub = np.zeros([h - l + 2 * pad for l, h in zip(lo, hi)], dtype=bool)
        sub[pad:-pad, pad:-pad, pad:-pad] = mask[
            lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        dist = distance_transform_edt(sub)[pad:-pad, pad:-pad, pad:-pad]
        t = np.clip(dist / float(soft_edge), 0.0, 1.0)
        weights = (0.5 * (1.0 - np.cos(np.pi * t))).astype(dtype)
    else:
        weights = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(dtype)

    return MaskedSupport(n, lo, weights, name="array")
