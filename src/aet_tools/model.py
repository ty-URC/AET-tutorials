"""試料モデル（fcc 二元合金ナノ粒子）と, その傾斜シリーズ / Atomic model and its tilt series.

座標の約束（genfire_cpu, resire_cpu と共通） / Conventions (shared with genfire_cpu, resire_cpu)
------------------------------------------------------------------------------------------
- 体積は (x, y, z) の順の 3 次元配列. y 軸が傾斜軸, 電子線は傾斜 0° で z 方向.
  A volume is an (x, y, z) array; y is the tilt axis, and the beam is along z at 0°.
- 投影像は (x, y, 傾斜番号). x は傾斜軸に垂直, y は傾斜軸に平行.
  Projections are (x, y, tilt index); x is across the tilt axis, y along it.
- 原子の座標はピクセル単位で, 箱の中心（配列の添字 n/2）からの相対位置.
  Atomic positions are in pixels, relative to the box centre (array index n/2).

原子は三次元のガウス関数で表す. その投影は同じ幅の二次元のガウス関数なので,
投影像も三次元像も解析的に正確に計算できる. 幅 atom_sigma（Å）は, 原子の熱振動と
像の分解能をまとめて表す.
Each atom is a 3D Gaussian. Its projection is a 2D Gaussian of the same width,
so both the projections and the volume are computed exactly. The width
atom_sigma (Å) lumps together thermal vibration and the imaging resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genfire_cpu.grid import euler_to_matrix

__all__ = ["Particle", "fcc_particle", "tilt_angles", "tilt_series",
           "add_poisson_noise"]


@dataclass
class Particle:
    """原子モデル / atomic model. 位置は箱の中心からの相対座標（ピクセル） / positions in px about the box centre."""
    positions: np.ndarray        # (N, 3) ピクセル / px
    species: np.ndarray          # (N,) 0 = 元素 A / element A, 1 = 元素 B / element B
    names: tuple                 # 元素名 / element names, e.g. ("Fe", "Pt")
    amplitudes: tuple            # 1 原子の積分強度（相対値） / integrated intensity per atom (relative)
    n: int                       # 箱の一辺 / box edge (px)
    pixel_size: float            # Å / px
    atom_sigma: float = 0.5      # 原子像の広がり（ガウスの σ, Å） / atom width (Gaussian σ, Å)

    @classmethod
    def from_saved(cls, d, n: int | None = None) -> "Particle":
        """np.savez で保存したデータから作る（n を変えて広い視野にもできる）.
        Rebuild from data saved with np.savez (optionally on a different box size n)."""
        return cls(d["positions"], d["species"], tuple(str(s) for s in d["names"]),
                   tuple(float(a) for a in d["amplitudes"]),
                   int(d["n"]) if n is None else int(n), float(d["pixel_size"]),
                   float(d["atom_sigma"]))

    def save_dict(self) -> dict:
        """np.savez に渡す辞書 / dict to pass to np.savez."""
        return dict(positions=self.positions, species=self.species, names=np.array(self.names),
                    amplitudes=np.array(self.amplitudes), n=self.n, pixel_size=self.pixel_size,
                    atom_sigma=self.atom_sigma)

    def __len__(self) -> int:
        return self.positions.shape[0]

    @property
    def sigma_px(self) -> float:
        """原子像の広がり（ピクセル） / atom width in pixels."""
        return self.atom_sigma / self.pixel_size

    def weights(self) -> np.ndarray:
        return np.asarray(self.amplitudes, dtype=float)[self.species]

    def positions_in_box(self) -> np.ndarray:
        """配列の添字で表した原子の位置 / positions as array indices."""
        return self.positions + self.n / 2.0

    def volume(self) -> np.ndarray:
        """正解の三次元像 / ground-truth volume."""
        return _splat(self.positions_in_box(), self.weights(), self.sigma_px, (self.n,) * 3)

    def summary(self) -> str:
        d = 2 * np.linalg.norm(self.positions, axis=1).max() * self.pixel_size
        counts = ", ".join("%s %d" % (nm, int((self.species == k).sum()))
                           for k, nm in enumerate(self.names))
        return ("%d atoms (%s), diameter about %.1f nm, atom sigma %.2f Å, "
                "box %d^3 px (%.2f Å/px, %.1f nm wide)"
                % (len(self), counts, d / 10, self.atom_sigma, self.n, self.pixel_size,
                   self.n * self.pixel_size / 10))


def _splat(centres: np.ndarray, weights: np.ndarray, sigma: float, shape) -> np.ndarray:
    """ガウス関数を足し合わせた像（2 次元または 3 次元）. 各ガウスの積分は weights.
    Sum of Gaussians (2D or 3D) whose integrals are ``weights``."""
    dim = len(shape)
    out = np.zeros(shape, dtype=np.float64)
    half = int(np.ceil(4 * sigma))
    off = np.arange(-half, half + 1)
    grids = np.meshgrid(*([off] * dim), indexing="ij")
    grids = [g.ravel() for g in grids]
    base = np.floor(centres).astype(np.int64)
    frac = centres - base
    norm = weights / ((2 * np.pi) ** (dim / 2) * sigma ** dim)
    for chunk in np.array_split(np.arange(len(centres)), max(1, len(centres) // 200)):
        d2 = 0.0
        idx = []
        for ax in range(dim):
            o = grids[ax][None, :] - frac[chunk, ax][:, None]
            d2 = d2 + o * o
            idx.append(base[chunk, ax][:, None] + grids[ax][None, :])
        val = norm[chunk][:, None] * np.exp(-0.5 * d2 / sigma ** 2)
        ok = np.ones(val.shape, bool)
        for ax in range(dim):
            ok &= (idx[ax] >= 0) & (idx[ax] < shape[ax])
        np.add.at(out, tuple(i[ok] for i in idx), val[ok])
    return out.astype(np.float32)


def fcc_particle(n: int = 64, pixel_size: float = 0.4, diameter_nm: float = 1.8,
                 lattice_a: float = 3.85, names=("Fe", "Pt"),
                 atomic_numbers=(26, 78), z_exponent: float = 1.7,
                 fraction_b: float = 0.5, atom_sigma: float = 0.5, seed: int = 0) -> Particle:
    """fcc 構造の球形ナノ粒子. 各サイトの元素はランダム（不規則相, A1）.

    HAADF-STEM の像強度は原子番号 Z のおよそ 1.7 乗に比例するとして, 元素 A を 1 とした
    相対強度を割り当てる（Fe と Pt なら約 1 : 6.5）. atom_sigma は原子像の広がり（Å）.

    A spherical fcc nanoparticle with randomly occupied sites (disordered A1
    phase). HAADF intensity is taken as about Z^1.7, relative to element A
    (Fe : Pt is about 1 : 6.5). atom_sigma is the atom width in Å.
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
    amp = tuple(float(v) for v in (z / z[0]) ** z_exponent)
    if 2 * r + 8 > n:
        raise ValueError("粒子が箱に対して大きすぎます / the particle (%.0f px across) "
                         "does not fit in the box (%d px); increase n" % (2 * r, n))
    return Particle(pos, species, tuple(names), amp, n, pixel_size, atom_sigma)


def tilt_angles(start: float = -70.0, stop: float = 70.0, step: float = 2.5) -> np.ndarray:
    """傾斜角（度）の 1 次元配列. stop も含む / tilt angles in degrees, including stop."""
    return np.arange(start, stop + step / 2, step, dtype=np.float64)


def tilt_series(particle: Particle, theta: np.ndarray) -> np.ndarray:
    """ノイズのない投影像 (n, n, 傾斜数). 1 軸傾斜（y 軸まわり）.

    三次元のガウス関数の投影は同じ幅の二次元のガウス関数なので, 原子の位置を
    投影面に写して二次元のガウス関数を足すだけで, 正確な投影像になる.
    Noise-free projections (n, n, n_tilts) for a single tilt axis (y). The
    projection of a 3D Gaussian is a 2D Gaussian of the same width, so each
    projection is exact.
    """
    n, c = particle.n, particle.n / 2.0
    w = particle.weights()
    out = np.empty((n, n, len(theta)), dtype=np.float32)
    for i, t in enumerate(theta):
        R = euler_to_matrix(0.0, float(t), 0.0)      # genfire_cpu と同じ向き / same convention as genfire_cpu
        uv = np.column_stack([particle.positions @ R[:, 0], particle.positions @ R[:, 1]]) + c
        out[:, :, i] = _splat(uv, w, particle.sigma_px, (n, n))
    return out


def add_poisson_noise(projections: np.ndarray, counts_per_atom: float = 50.0,
                      seed: int = 0):
    """ポアソンノイズ（電子の数のばらつき）を加える / add Poisson (shot) noise.

    counts_per_atom は「元素 A の原子 1 個あたり, 投影像 1 枚で数えられる電子の数」.
    大きいほど電子線量が多く, ノイズが小さい.
    counts_per_atom is the number of electrons counted per atom of element A
    in one projection; larger means higher dose and less noise.
    戻り値 / returns: (ノイズ入りの投影像 / noisy projections, scale)
    """
    rng = np.random.default_rng(seed)
    scale = counts_per_atom          # 原子 A 1 個の積分強度 = 1 / one A atom integrates to 1
    counts = rng.poisson(np.maximum(projections, 0) * scale)
    return (counts / scale).astype(np.float32), scale
