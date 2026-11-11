"""再構成中の進捗を JupyterLab に表示し続ける / Live progress display for JupyterLab.

使い方 / Usage::

    mon = LiveMonitor(num_iterations=100, every=5, truth=truth)
    result = rs.reconstruct(b, theta, config, callback=mon)          # RESIRE
    result = gc.reconstruct(proj, euler, config, on_progress=mon)    # GENFIRE

数 回ごとに図を描き直す（同じ場所を上書きする）. 図には
  - 進捗（反復回数, 経過時間, 残り時間の見積もり）
  - 再構成の指標（R 因子, 誤差, R-free など）の変化
  - 正解との相関係数（truth を与えたとき. シミュレーションでのみ可能）
  - 現在の三次元像の中心断面（xy と xz）
を表示する.

Every ``every`` iterations the figure is redrawn in place. It shows the
progress (iteration, elapsed time, estimated time left), the reconstruction
metrics (R factor, error, R-free, ...), the correlation with the ground truth
(when ``truth`` is given; possible only for simulations) and central sections
(xy and xz) of the current volume.
"""

from __future__ import annotations

import io
import time

import numpy as np
from matplotlib.figure import Figure

__all__ = ["LiveMonitor"]


class LiveMonitor:
    def __init__(self, num_iterations: int, every: int = 5, truth=None,
                 title: str = "", log_scale: bool = True):
        self.num_iterations = int(num_iterations)
        self.every = max(1, int(every))
        self.truth = None if truth is None else np.asarray(truth, dtype=np.float64)
        self.title = title
        self.log_scale = log_scale
        self.history: dict[str, list] = {}
        self.iterations: list[int] = []
        self.corr_it: list[int] = []
        self.corr: list[float] = []
        self.t0 = None
        self.handle = None

    # 再構成の関数から毎回呼ばれる / called by the reconstruction every iteration
    def __call__(self, it: int, metrics: dict, get_volume) -> None:
        if self.t0 is None:
            self.t0 = time.perf_counter()
        self.iterations.append(it)
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))
        if it % self.every == 0 or it == self.num_iterations or it == 1:
            vol = np.asarray(get_volume())
            if self.truth is not None:
                a = vol.ravel().astype(np.float64)
                self.corr_it.append(it)
                self.corr.append(float(np.corrcoef(a, self.truth.ravel())[0, 1]))
            self._draw(it, vol)

    def _status(self, it: int) -> str:
        el = time.perf_counter() - self.t0
        eta = el / max(it, 1) * (self.num_iterations - it)
        parts = ["iteration %d / %d (%d%%)" % (it, self.num_iterations, 100 * it // self.num_iterations),
                 "elapsed %.1f s" % el, "left %.1f s" % eta]
        parts += ["%s %.4f" % (k, v[-1]) for k, v in self.history.items() if k != "loss"]
        if self.corr:
            parts.append("corr. with truth %.4f" % self.corr[-1])
        return "   ".join(parts)

    def _draw(self, it: int, vol: np.ndarray) -> None:
        from IPython.display import Image, display

        ncol = 4 if self.truth is not None else 3
        fig = Figure(figsize=(3.0 * ncol, 3.0))
        axes = fig.subplots(1, ncol)
        ax = axes[0]
        for k, v in self.history.items():
            if k == "loss":
                continue
            ax.plot(self.iterations, v, label=k)
        ax.set_xlim(0, self.num_iterations)
        if self.log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("metric")
        ax.legend(fontsize=8)
        k = 1
        if self.truth is not None:
            ax = axes[1]
            ax.plot(self.corr_it, self.corr, "o-", ms=3, color="k")
            ax.set_xlim(0, self.num_iterations)
            ax.set_xlabel("iteration")
            ax.set_ylabel("correlation with truth")
            k = 2
        n = vol.shape
        for ax, img, name in ((axes[k], vol[:, :, n[2] // 2], "xy"),
                              (axes[k + 1], vol[:, n[1] // 2, :], "xz")):
            ax.imshow(img.T, origin="lower", cmap="gray", interpolation="nearest")
            ax.set_title("%s section, iteration %d" % (name, it), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle((self.title + "   " if self.title else "") + self._status(it), fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)     # PNG にして表示 / show as a PNG image
        img = Image(data=buf.getvalue(), format="png")
        if self.handle is None:
            self.handle = display(img, display_id=True)
        else:
            self.handle.update(img)

