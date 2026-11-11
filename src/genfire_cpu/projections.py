"""Forward projector (Fourier slice theorem).

Used to build test data and to check a reconstruction against the projections
it came from.  Same Euler convention as :mod:`.grid`, so a volume projected
here grids back onto the slice it came from.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grid import euler_to_matrix

__all__ = ["project_volume", "single_tilt_series"]


def single_tilt_series(start: float = -60.0, stop: float = 60.0,
                       step: float = 2.0) -> np.ndarray:
    """(n, 3) Euler angles for a single tilt series about the theta axis."""
    theta = np.arange(start, stop + step / 2, step, dtype=np.float64)
    angles = np.zeros((theta.size, 3))
    angles[:, 1] = theta
    return angles


def project_volume(volume: np.ndarray, angles: np.ndarray,
                   oversampling: float = 2.0,
                   dtype=np.float32,
                   verbose: bool = False) -> np.ndarray:
    """Project ``volume`` at each orientation in ``angles`` (degrees).

    The volume is zero-padded by ``oversampling`` before the 3D FFT so that the
    central-slice interpolation has something to interpolate between; each
    projection is cropped back to the original size.

    Returns (n0, n0, n_proj).
    """
    from scipy.interpolate import RegularGridInterpolator

    volume = np.asarray(volume, dtype=np.float64)
    n0 = volume.shape[0]
    pad = int(round(n0 * oversampling)) // 2 * 2
    lo = (pad - n0) // 2
    padded = np.zeros((pad, pad, pad), dtype=np.float64)
    padded[lo:lo + n0, lo:lo + n0, lo:lo + n0] = volume

    vol_k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(padded)))
    del padded

    c = pad // 2
    axis = np.arange(pad) - c
    interp = RegularGridInterpolator((axis, axis, axis), vol_k,
                                     bounds_error=False, fill_value=0)

    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    flat = np.stack([kx.ravel(), ky.ravel(), np.zeros(kx.size)], axis=0)

    angles = np.atleast_2d(np.asarray(angles, dtype=np.float64))
    out = np.empty((n0, n0, angles.shape[0]), dtype=dtype)
    for i, (phi, theta, psi) in enumerate(angles):
        rot = euler_to_matrix(phi, theta, psi) @ flat
        plane = interp(rot.T).reshape(pad, pad)
        img = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(plane))))
        out[:, :, i] = img[lo:lo + n0, lo:lo + n0]
        if verbose and (i + 1) % 10 == 0:
            print("  projected %d/%d" % (i + 1, angles.shape[0]))
    return out
