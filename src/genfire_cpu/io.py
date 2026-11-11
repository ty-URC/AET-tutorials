"""Volume, projection and angle I/O.

MRC is handled by ``mrcfile`` (upstream hand-rolls a header parser that gets
the machine stamp wrong on files written by MATLAB).  ``.npy``/``.npz``, plain
text and MATLAB ``.mat`` are also accepted so existing GENFIRE inputs load
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["read_volume", "write_volume", "read_projections", "read_angles",
           "save_results"]


def _suffix(path) -> str:
    return Path(path).suffix.lower()


def read_volume(path, dtype=np.float32) -> np.ndarray:
    """Read a 3D volume from .mrc/.mrcs/.rec/.npy/.mat."""
    ext = _suffix(path)
    if ext in (".mrc", ".mrcs", ".rec", ".map"):
        import mrcfile

        with mrcfile.open(str(path), permissive=True) as m:
            return np.asarray(m.data, dtype=dtype)
    if ext == ".npy":
        return np.load(str(path)).astype(dtype, copy=False)
    if ext == ".npz":
        with np.load(str(path)) as z:
            return z[list(z.keys())[0]].astype(dtype, copy=False)
    if ext == ".mat":
        import scipy.io

        data = scipy.io.loadmat(str(path))
        arrays = [v for k, v in data.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray)]
        if not arrays:
            raise ValueError("no array found in %s" % path)
        return max(arrays, key=lambda a: a.size).astype(dtype, copy=False)
    raise ValueError("unsupported volume format: %s" % ext)


def write_volume(path, volume: np.ndarray, pixel_size: Optional[float] = None,
                 overwrite: bool = True) -> None:
    """Write a volume, recording ``pixel_size`` (angstroms) in the MRC header."""
    ext = _suffix(path)
    if ext in (".mrc", ".mrcs", ".rec", ".map"):
        import mrcfile

        arr = np.ascontiguousarray(volume, dtype=np.float32)
        with mrcfile.new(str(path), overwrite=overwrite) as m:
            m.set_data(arr)
            if pixel_size:
                m.voxel_size = float(pixel_size)
        return
    if ext == ".npy":
        np.save(str(path), volume)
        return
    raise ValueError("unsupported volume format: %s" % ext)


def read_projections(path, dtype=np.float32) -> np.ndarray:
    """Read a projection stack as (n, n, n_proj)."""
    arr = read_volume(path, dtype=dtype)
    if arr.ndim != 3:
        raise ValueError("projection stack must be 3D, got %r" % (arr.shape,))
    # MRC stores stacks slice-major (n_proj, n, n); GENFIRE wants them last
    if arr.shape[0] != arr.shape[1] and arr.shape[1] == arr.shape[2]:
        arr = np.moveaxis(arr, 0, -1)
    return np.ascontiguousarray(arr)


def read_angles(path) -> np.ndarray:
    """Read Euler angles as (n_proj, 3) in degrees.

    Accepts a single tilt series (one column/row of theta values) and expands
    it to phi = psi = 0.
    """
    ext = _suffix(path)
    if ext in (".txt", ".dat", ".csv"):
        arr = np.loadtxt(str(path), delimiter="," if ext == ".csv" else None)
    elif ext == ".npy":
        arr = np.load(str(path))
    elif ext == ".mat":
        import scipy.io

        data = scipy.io.loadmat(str(path))
        arrays = [v for k, v in data.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray)]
        arr = max(arrays, key=lambda a: a.size)
    else:
        raise ValueError("unsupported angle format: %s" % ext)

    arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.ndim == 2 and 1 in arr.shape:          # single tilt series
        theta = arr.ravel()
        out = np.zeros((theta.size, 3))
        out[:, 1] = theta
        return out
    if arr.shape[1] != 3:
        raise ValueError("angles must be (n, 3) or a single tilt series, got %r"
                         % (arr.shape,))
    return arr


def save_results(path, result, pixel_size: Optional[float] = None) -> None:
    """Write the reconstruction plus its error curves next to it."""
    path = Path(path)
    write_volume(path, result.reconstruction, pixel_size)
    meta = {"error_k": result.error_k}
    if result.rfree_total is not None:
        meta["rfree_total"] = result.rfree_total
        meta["rfree_by_shell"] = result.rfree_by_shell
        meta["shell_resolution"] = result.shell_resolution
    np.savez(str(path.with_suffix(".metrics.npz")), **meta)
