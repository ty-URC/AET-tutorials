"""試料モデル（fcc 二元合金ナノ粒子）と, その傾斜シリーズ.

座標の約束（genfire_cpu, resire_cpu と共通）
------------------------------------------
- 体積は (x, y, z) の順の 3 次元配列。y 軸が傾斜軸, 電子線は傾斜 0° で z 方向。
- 投影像は (x, y, 傾斜番号) の 3 次元配列。x は傾斜軸に垂直, y は傾斜軸に平行。
- 原子の座標はピクセル単位で, 箱の中心 (n/2, n/2, n/2) からの相対位置。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from genfire_cpu.phantom import Spheres

__all__ = ["Particle", "fcc_particle", "tilt_angles", "tilt_series",
           "add_poisson_noise"]


@dataclass
class Particle:
    """原子モデル. 位置は箱の中心からの相対座標（ピクセル）."""
    positions: np.ndarray        # (N, 3) ピクセル, 箱の中心が原点
    species: np.ndarray          # (N,) 0 = 元素 A, 1 = 元素 B
    names: tuple                 # 元素名, 例 ("Fe", "Pt")
    amplitudes: tuple            # 1 原子あたりの散乱の強さ（相対値）
    n: int                       # 箱の一辺（ピクセル）
    pixel_size: float            # Å / ピクセル
    radius_px: float = 1.0       # 原子を表す球の半径
    blur_px: float = 0.8         # さらにかけるガウスぼかし（σ, ピクセル）

    def __len__(self) -> int:
        return self.positions.shape[0]

    @property
    def spheres(self) -> Spheres:
        amp = np.asarray(self.amplitudes, dtype=float)[self.species]
        # 球の体積で割り, 原子 1 個の積分強度が amplitudes になるようにする
        amp = amp / (4.0 / 3.0 * np.pi * self.radius_px ** 3)
        return Spheres(self.positions, self.radius_px, amp)

    def positions_in_box(self) -> np.ndarray:
        """配列の添字で表した原子の位置（0 始まり）."""
        return self.positions + self.n / 2.0

    def volume(self) -> np.ndarray:
        """正解の三次元像（比較・表示用）."""
        vol = self.spheres.volume(self.n)
        return gaussian_filter(vol, self.blur_px).astype(np.float32)

    def summary(self) -> str:
        d = 2 * np.linalg.norm(self.positions, axis=1).max() * self.pixel_size
        counts = ", ".join("%s %d" % (nm, int((self.species == k).sum()))
                           for k, nm in enumerate(self.names))
        return ("%d atoms (%s), diameter about %.1f nm, box %d^3 px (%.2f Å/px, %.1f nm wide)"
                % (len(self), counts, d / 10, self.n, self.pixel_size,
                   self.n * self.pixel_size / 10))


def fcc_particle(n: int = 64, pixel_size: float = 0.4, diameter_nm: float = 1.8,
                 lattice_a: float = 3.85, names=("Fe", "Pt"),
                 atomic_numbers=(26, 78), z_exponent: float = 1.7,
                 fraction_b: float = 0.5, seed: int = 0) -> Particle:
    """fcc 構造の球形ナノ粒子. 各サイトの元素はランダム（不規則相, A1）.

    HAADF-STEM の像強度は原子番号 Z のおよそ 1.7 乗に比例するとして,
    元素 A を 1 とした相対強度を割り当てる（Fe と Pt なら約 1 : 6.5）。
    """
    a = lattice_a / pixel_size
    r = diameter_nm * 10 / pixel_size / 2
    k = int(np.ceil(r / a)) + 1
    g = np.arange(-k, k + 1) * a
    corners = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    pos = (corners[:, None, :] + basis[None]).reshape(-1, 3)
    pos = pos[np.linalg.norm(pos, axis=1) <= r]
    rng = np.random.default_rng(seed)
    species = (rng.random(pos.shape[0]) < fraction_b).astype(np.int64)
    z = np.asarray(atomic_numbers, dtype=float)
    amp = tuple((z / z[0]) ** z_exponent)
    if 2 * r + 8 > n:
        raise ValueError("粒子が箱に対して大きすぎます / the particle (%.0f px across) "
                         "does not fit in the box (%d px); increase n" % (2 * r, n))
    return Particle(pos, species, tuple(names), amp, n, pixel_size)


def tilt_angles(start: float = -70.0, stop: float = 70.0, step: float = 2.5) -> np.ndarray:
    """傾斜角（度）の 1 次元配列. stop も含む."""
    return np.arange(start, stop + step / 2, step, dtype=np.float64)


def tilt_series(particle: Particle, theta: np.ndarray) -> np.ndarray:
    """ノイズのない投影像 (n, n, 傾斜数). 1 軸傾斜（y 軸まわり）.

    球の投影は解析的に計算でき（genfire_cpu.phantom.Spheres）,
    それに 2 次元のガウスぼかしをかける。三次元のガウスぼかしと投影は
    順序を入れ替えても同じなので, Particle.volume() の正確な投影になっている。
    """
    euler = np.zeros((len(theta), 3))
    euler[:, 1] = theta
    proj = particle.spheres.project(euler, particle.n, dtype=np.float64)
    for i in range(proj.shape[2]):
        proj[:, :, i] = gaussian_filter(proj[:, :, i], particle.blur_px)
    return proj.astype(np.float32)


def add_poisson_noise(projections: np.ndarray, counts_per_atom: float = 50.0,
                      seed: int = 0):
    """ポアソンノイズ（電子の数のばらつき）を加える.

    counts_per_atom は「元素 A の原子 1 個あたり, 投影像全体で数えられる
    電子の数」。大きいほど電子線量が多く, ノイズが小さい。
    戻り値は (ノイズ入りの投影像, 1 原子 A の積分強度あたりのカウント数)。
    """
    rng = np.random.default_rng(seed)
    scale = counts_per_atom          # 原子 A 1 個の積分強度 = 1
    counts = rng.poisson(np.maximum(projections, 0) * scale)
    return (counts / scale).astype(np.float32), scale
