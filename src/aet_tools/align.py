"""傾斜シリーズの前処理：背景除去と位置合わせ / Preprocessing: background removal and alignment.

投影像の配列は (x, y, 傾斜番号)。y が傾斜軸。
Projections are (x, y, tilt index); y is the tilt axis.

流れ / Workflow
  1. subtract_background      支持膜などの背景を差し引く / remove the support-film background
  2. align_cross_correlation  平均像との相互相関で大きなずれを取る / coarse: cross-correlation with the mean image
  3. align_common_line        傾斜軸方向（y）を共通線法で合わせる / along the tilt axis (y): common line
  4. align_center_of_mass     傾斜軸に垂直な方向（x）を重心法で合わせる / across the tilt axis (x): centre of mass
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["particle_mask", "subtract_background", "shift_images",
           "align_cross_correlation", "align_common_line",
           "align_center_of_mass"]


def particle_mask(image: np.ndarray, frac: float = 0.3, grow: int = 6) -> np.ndarray:
    """試料の領域：ぼかした像をしきい値で切り, 最大の連結領域を少し広げる.

    Particle footprint: threshold the blurred image, keep the largest
    connected region and dilate it a little.
    """
    g = ndimage.gaussian_filter(np.asarray(image, float), 3)
    g = g - np.median(g)
    lab, n = ndimage.label(g > frac * g.max())
    if n == 0:
        return np.zeros(g.shape, bool)
    sizes = ndimage.sum(np.ones_like(g), lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_dilation(ndimage.binary_fill_holes(m), iterations=grow)


def subtract_background(projections: np.ndarray, order: int = 2):
    """試料の外側の値に 2 次の曲面を当てはめて背景（支持膜など）とし, 差し引く.

    支持膜の厚さはゆっくり変わるので, 試料の外側の画素に低次の多項式の曲面を
    最小二乗で当てはめ, 試料の下も同じ曲面で背景を見積もる.

    Fit a low-order polynomial surface (default: quadratic) to the pixels
    outside the particle and subtract it as the background (support film,
    detector offset). The film thickness varies slowly, so the same surface is
    a good estimate under the particle.
    戻り値 / returns: (背景を引いた像 / subtracted stack, 背景 / background, 試料の領域 / masks)
    """
    p = np.asarray(projections, float)
    nx, ny = p.shape[:2]
    u, v = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    terms = np.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)], -1)
    out, bg, masks = np.empty_like(p), np.empty_like(p), np.empty(p.shape, bool)
    for k in range(p.shape[2]):
        im = p[:, :, k]
        m = particle_mask(im)
        coef, *_ = np.linalg.lstsq(terms[~m], im[~m], rcond=None)
        b = terms @ coef
        bg[:, :, k], out[:, :, k], masks[:, :, k] = b, im - b, m
    return out.astype(np.float32), bg.astype(np.float32), masks


def shift_images(projections: np.ndarray, shifts: np.ndarray, order: int = 3) -> np.ndarray:
    """各像を (dx, dy) だけ動かす（3 次のスプライン補間）. shifts は (傾斜数, 2).

    Move each image by (dx, dy) with cubic-spline interpolation; shifts is (n_tilts, 2).
    """
    p = np.asarray(projections)
    out = np.empty_like(p)
    for k in range(p.shape[2]):
        out[:, :, k] = ndimage.shift(p[:, :, k], shifts[k], order=order, mode="nearest")
    return out


def _xcorr_shift(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """mov を ref に重ねるための移動量（窓をかけた相互相関, 放物線でサブピクセル）.

    Shift that brings ``mov`` onto ``ref`` (windowed cross-correlation, parabolic sub-pixel peak).
    """
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    a = (ref - ref.mean()) * win
    b = (mov - mov.mean()) * win
    cc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    cc = np.fft.fftshift(cc)
    i = np.array(np.unravel_index(np.argmax(cc), cc.shape))
    shift = i - np.array(cc.shape) // 2
    sub = np.zeros(2)
    for ax in range(2):
        lo, hi = i.copy(), i.copy()
        lo[ax] -= 1
        hi[ax] += 1
        if 0 <= lo[ax] and hi[ax] < cc.shape[ax]:
            c0, cl, ch = cc[tuple(i)], cc[tuple(lo)], cc[tuple(hi)]
            den = cl - 2 * c0 + ch
            sub[ax] = 0.5 * (cl - ch) / den if den != 0 else 0.0
    return shift + sub


def align_cross_correlation(projections: np.ndarray, iterations: int = 4,
                            smooth: float = 3.0) -> np.ndarray:
    """粗い位置合わせ：全体の平均像を参照にした相互相関を繰り返す.

    像を強くぼかし（格子の縞を消して粒子の輪郭で比べる）, 各像を平均像に
    重ねる移動量を求める. 平均像を作り直して数回繰り返す.

    Coarse alignment: cross-correlate each (strongly blurred) image with the
    mean image, then rebuild the mean and repeat. Blurring removes the lattice
    fringes so that the particle outline is matched, not a lattice period.
    戻り値 / returns: shifts (傾斜数 / n_tilts, 2) = (dx, dy)
    """
    sil = np.clip(ndimage.gaussian_filter(np.asarray(projections, float), (smooth, smooth, 0)), 0, None)
    shifts = np.zeros((sil.shape[2], 2))
    for _ in range(iterations):
        cur = shift_images(sil, shifts, order=1)
        ref = cur.mean(axis=2)
        for k in range(sil.shape[2]):
            shifts[k] += _xcorr_shift(ref, cur[:, :, k])
    return shifts


def _shift1d(ref: np.ndarray, mov: np.ndarray, rmax: int = 6) -> float:
    r, m = ref - ref.mean(), mov - mov.mean()
    lags = np.arange(-rmax, rmax + 1)
    cc = np.array([np.sum(r * np.roll(m, L)) for L in lags])
    i = int(np.clip(np.argmax(cc), 1, len(lags) - 2))
    den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
    return float(lags[i] + (0.5 * (cc[i - 1] - cc[i + 1]) / den if den != 0 else 0.0))


def align_common_line(projections: np.ndarray, iterations: int = 3) -> np.ndarray:
    """傾斜軸方向（y）の位置合わせ：共通線法.

    y 軸まわりに傾けても, 各像を x 方向に足し合わせた y 方向の分布は同じになる
    （試料の y ごとの総量）. この分布が全ての像でそろうように y 方向に動かす.

    Along the tilt axis (y): the common-line method. Summing each projection
    over x gives the same profile along y at every tilt (the total mass per
    y), so each image is shifted along y until the profiles agree.
    戻り値 / returns: dy (傾斜数 / n_tilts,)
    """
    p = np.asarray(projections, float)
    dy = np.zeros(p.shape[2])
    for _ in range(iterations):
        cur = shift_images(p, np.column_stack([np.zeros_like(dy), dy]))
        for k in range(p.shape[2]):              # 試料の外（雑音だけ）は使わない / ignore the empty field
            cur[:, :, k] *= particle_mask(cur[:, :, k])
        lines = cur.sum(axis=0)                  # (y, 傾斜数 / n_tilts)
        ref = lines.mean(axis=1)
        for k in range(p.shape[2]):
            dy[k] += _shift1d(ref, lines[:, k])
    return dy


def align_center_of_mass(projections: np.ndarray, center: float | None = None,
                         iterations: int = 3) -> np.ndarray:
    """傾斜軸に垂直な方向（x）の位置合わせ：重心法.

    各像の（試料の領域内の）重心が x = center にくるように動かす. 全ての像の
    重心がそろうと, 試料の三次元の重心が回転軸の上に乗る. center の既定値は
    n/2（GENFIRE と同じ回転中心）.

    Across the tilt axis (x): the centre-of-mass method. Each image is moved so
    that its centre of mass (within the particle footprint) lies at
    x = center; then the 3D centre of mass of the specimen sits on the rotation
    axis. The default centre is n/2, GENFIRE's rotation centre.
    戻り値 / returns: dx (傾斜数 / n_tilts,)
    """
    p = np.asarray(projections, float)
    c = p.shape[0] / 2 if center is None else center
    x = np.arange(p.shape[0])
    dx = np.zeros(p.shape[2])
    for _ in range(iterations):
        cur = shift_images(p, np.column_stack([dx, np.zeros_like(dx)]))
        for k in range(p.shape[2]):
            im = cur[:, :, k]
            prof = (np.clip(im, 0, None) * particle_mask(im)).sum(axis=1)
            dx[k] += c - (prof * x).sum() / prof.sum()
    return dx
