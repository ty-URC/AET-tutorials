"""表示の道具. 画像は横軸 x（傾斜軸に垂直）, 縦軸 y（傾斜軸）で表示する."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["show_projections", "show_slices", "show_orthoslices"]


def _img(ax, a, title=None, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(np.asarray(a).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


def show_projections(projections: np.ndarray, theta: np.ndarray, picks=None,
                     ncols: int = 5, size: float = 2.0):
    """傾斜シリーズからいくつかの角度の投影像を並べる."""
    if picks is None:
        picks = np.linspace(0, len(theta) - 1, ncols).round().astype(int)
    picks = list(picks)
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), picks):
        ax.axis("on")
        _img(ax, projections[:, :, k], "%+.1f°" % theta[k])
    fig.tight_layout()
    return fig


def show_slices(volumes: dict, z: int | None = None, thickness: int = 1, size: float = 3.0):
    """複数の三次元像の, 同じ高さ z の断面（xy 面）を並べる.

    thickness > 1 なら z のまわりの数枚を足し合わせる（原子の層を見やすくする）。
    """
    names = list(volumes)
    n = next(iter(volumes.values())).shape[2]
    z = n // 2 if z is None else z
    lo, hi = z - thickness // 2, z - thickness // 2 + thickness
    fig, axes = plt.subplots(1, len(names), figsize=(size * len(names), size), squeeze=False)
    for ax, nm in zip(axes[0], names):
        _img(ax, volumes[nm][:, :, lo:hi].sum(2), "%s  (z = %d)" % (nm, z))
    fig.tight_layout()
    return fig


def show_orthoslices(volume: np.ndarray, title: str = "", size: float = 2.6):
    """中心を通る 3 つの直交断面（xy, xz, yz）."""
    n = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(3 * size, size))
    _img(axes[0], volume[:, :, n[2] // 2], "xy (beam direction at 0°)")
    _img(axes[1], volume[:, n[1] // 2, :], "xz (perpendicular to tilt axis)")
    _img(axes[2], volume[n[0] // 2, :, :], "yz")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
