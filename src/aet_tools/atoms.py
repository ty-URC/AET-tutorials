"""原子の抽出（原子のトレース）, 元素の分類, 正解との照合."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

__all__ = ["find_atoms", "classify_two_species", "match_atoms"]


def _ball(r: float) -> np.ndarray:
    k = int(np.ceil(r))
    g = np.arange(-k, k + 1)
    d2 = g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2
    return np.argwhere(d2 <= r * r) - k


def find_atoms(volume: np.ndarray, min_distance_px: float,
               threshold: float = 0.1, integrate_radius_px: float | None = None):
    """三次元像の局所最大を原子の候補として取り出す.

    1. 隣り合う原子より近い極大は同じ原子とみなし, 明るい方だけ残す
       （min_distance_px は最近接原子間距離より少し短く取る）。
    2. 最大値の threshold 倍より暗い極大は捨てる（背景のノイズ）。
    3. 極大のまわり 3x3x3 の重心で, ピクセルより細かい位置を求める。
    4. 半径 integrate_radius_px の球の中の強度の和を, その原子の強度とする
       （元素の分類に使う）。

    戻り値: (位置 (N, 3) [配列の添字], 強度 (N,))。明るい順。
    """
    vol = np.asarray(volume, dtype=np.float64)
    n = np.array(vol.shape)
    size = int(2 * np.floor(min_distance_px / 2) + 1)
    peak = (vol == maximum_filter(vol, size=size)) & (vol > threshold * vol.max())
    idx = np.argwhere(peak)
    idx = idx[np.all((idx > 0) & (idx < n - 1), axis=1)]
    idx = idx[np.argsort(-vol[tuple(idx.T)])]

    # 明るい順に見て, すでに採用した原子に近すぎるものを捨てる
    keep: list[int] = []
    tree_pts: list[np.ndarray] = []
    for i, p in enumerate(idx):
        if tree_pts and np.min(np.linalg.norm(np.array(tree_pts) - p, axis=1)) < min_distance_px:
            continue
        keep.append(i)
        tree_pts.append(p)
    idx = idx[keep]

    # 3x3x3 の重心でピクセル以下の位置を求める
    off = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = np.empty(idx.shape, dtype=np.float64)
    for j, p in enumerate(idx):
        q = np.clip(p + off, 0, n - 1)
        w = np.maximum(vol[tuple(q.T)], 0)
        pos[j] = p + (w[:, None] * off).sum(0) / max(w.sum(), 1e-12)

    r = integrate_radius_px if integrate_radius_px is not None else 0.5 * min_distance_px
    ball = _ball(r)
    inten = np.empty(len(idx))
    for j, p in enumerate(idx):
        q = np.clip(p + ball, 0, n - 1)
        inten[j] = vol[tuple(q.T)].sum()
    return pos, inten


def classify_two_species(intensity: np.ndarray, iterations: int = 100):
    """原子の強度を 2 つのグループに分ける（1 次元の k-means 法）.

    戻り値: (元素ラベル 0/1, しきい値, 2 つのグループの平均強度)。
    0 が暗い方（軽い元素）, 1 が明るい方（重い元素）。
    """
    x = np.asarray(intensity, dtype=np.float64)
    c = np.percentile(x, [25, 75])
    for _ in range(iterations):
        lab = (np.abs(x - c[1]) < np.abs(x - c[0])).astype(np.int64)
        new = np.array([x[lab == k].mean() if np.any(lab == k) else c[k] for k in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    return lab, float(c.mean()), c


def match_atoms(true_pos: np.ndarray, found_pos: np.ndarray,
                true_species=None, found_species=None,
                tolerance_px: float = 2.0, pixel_size: float = 1.0,
                remove_offset: bool = False) -> dict:
    """正解の原子と見つけた原子を 1 対 1 に対応させて, 精度を数える.

    tolerance_px より離れた組は対応させない。
    remove_offset=True なら, 粒子全体の平行移動（位置合わせで決まらない 3 次元の
    ずれ）を求めて差し引いてから数える / if True, the overall 3D translation
    (which alignment cannot fix) is estimated and removed before counting.
    戻り値の辞書:
      found / missed / false : 正しく見つけた数, 見落とし, 誤検出
      rmsd_pm                : 対応した原子の位置ずれの二乗平均平方根（pm）
      species_accuracy       : 対応した原子のうち, 元素が正しい割合
      pairs                  : (正解の番号, 見つけた原子の番号) の配列
      offset_px              : 差し引いた平行移動（ピクセル） / removed translation (px)
    """
    t = np.asarray(true_pos, float)
    f = np.asarray(found_pos, float)
    offset = np.zeros(3)
    if remove_offset:
        for tol in (3 * tolerance_px, 2 * tolerance_px, tolerance_px, tolerance_px):
            d, j = cKDTree(f - offset).query(t, distance_upper_bound=tol)
            ok = np.isfinite(d)
            if ok.sum() < 3:
                break
            offset = offset + np.median(f[j[ok]] - offset - t[ok], axis=0)
        f = f - offset
    tree = cKDTree(f)
    pairs = tree.query_ball_point(t, tolerance_px)
    cost = np.full((len(t), len(f)), 1e6)
    for i, js in enumerate(pairs):
        for j in js:
            cost[i, j] = np.sum((t[i] - f[j]) ** 2)
    ri, ci = linear_sum_assignment(cost)
    ok = cost[ri, ci] < 1e6
    ri, ci = ri[ok], ci[ok]
    d = np.linalg.norm(t[ri] - f[ci], axis=1)
    out = dict(found=int(ok.sum()), missed=int(len(t) - ok.sum()),
               false=int(len(f) - ok.sum()),
               rmsd_pm=float(np.sqrt(np.mean(d ** 2)) * pixel_size * 100) if len(d) else np.nan,
               displacement_pm=d * pixel_size * 100,
               offset_px=offset,
               pairs=np.stack([ri, ci], 1))
    if true_species is not None and found_species is not None:
        same = np.asarray(true_species)[ri] == np.asarray(found_species)[ci]
        out["species_accuracy"] = float(same.mean()) if len(same) else np.nan
    return out
