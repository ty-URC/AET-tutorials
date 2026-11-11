# AET Tutorials

原子分解能電子線トモグラフィー（atomic electron tomography, AET）の入門用チュートリアルです．数 nm の合金ナノ粒子（FePt）を例に，**傾斜シリーズ → 前処理（背景除去・位置合わせ）→ 三次元再構成（GENFIRE / RESIRE）→ 原子の抽出と元素の分類**までを，JupyterLab のノートブックで自分の PC 上で一通り体験できます．正解がわかっているシミュレーションデータを使うので，再構成や原子の抽出がどれだけ正しいかを確かめられます．再構成中は，進捗・R 因子などの指標・三次元像の断面が図として更新され続けます．

An introductory tutorial on atomic electron tomography (AET). Using an alloy (FePt) nanoparticle a few nanometres across, you go through **tilt series → preprocessing (background removal, alignment) → 3D reconstruction (GENFIRE / RESIRE) → atom tracing and species classification** in JupyterLab notebooks on your own computer. Because the data are simulated and the answer is known, you can check how accurate each step is. During reconstruction, a live figure shows the progress, the R factor and other metrics, and sections of the current volume.

今後も内容を更新していく予定です． / The tutorial will continue to be updated.

## 環境構築 / Setup

Python の環境は [uv](https://docs.astral.sh/uv/) で作ります．uv は Python 本体と必要なパッケージをまとめて用意してくれるツールで，あらかじめ Python をインストールしておく必要はありません．コマンドは**ターミナル**（Windows では「PowerShell」，macOS / Linux では「ターミナル」）で実行します．

The Python environment is set up with [uv](https://docs.astral.sh/uv/). uv installs Python itself together with all required packages, so you do not need to install Python beforehand. Run the commands below in a **terminal** (PowerShell on Windows, Terminal on macOS / Linux).

### (1) uv のインストール（最初の 1 回だけ） / Install uv (once)

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストールが終わったらターミナルを開き直し，`uv --version` でバージョンが表示されることを確認します．

After the installation, open a new terminal and check that `uv --version` prints a version number.

### (2) ダウンロードと環境構築（最初の 1 回だけ） / Download and set up (once)

```bash
git clone https://github.com/ty-URC/AET-tutorials.git
cd AET-tutorials
uv sync
```

`uv sync` で，Python 3.12 と必要なパッケージ（NumPy, SciPy, Matplotlib, pyFFTW, JupyterLab など）が `AET-tutorials/.venv` フォルダにインストールされます．パッケージのバージョンは `uv.lock` に記録されているので，誰が実行しても同じ環境になります．

`uv sync` installs Python 3.12 and the required packages (NumPy, SciPy, Matplotlib, pyFFTW, JupyterLab, ...) into the `AET-tutorials/.venv` folder. The package versions are pinned in `uv.lock`, so everyone gets the same environment.

### (3) JupyterLab の起動（毎回） / Start JupyterLab (every time)

```bash
cd AET-tutorials
uv run jupyter lab
```

- `git` がない場合は，このページの「Code」→「Download ZIP」でダウンロードして展開し，そのフォルダで `cd` 以降を実行します．
  Without `git`, use "Code" → "Download ZIP" on this page, unzip it, and run the commands from `cd` onwards in that folder.
- `uv run jupyter lab` を実行するとブラウザが開きます．`notebooks` フォルダを開き，`00_setup.ipynb` から順に開いてください．
  `uv run jupyter lab` opens a browser. Open the `notebooks` folder and work through the notebooks in order, starting from `00_setup.ipynb`.
- セルは `Shift + Enter` で 1 つずつ実行できます．「Run」→「Run All Cells」で全部を実行することもできます．
  Run cells one by one with `Shift + Enter`, or all at once with "Run" → "Run All Cells".
- 終えるときは，ブラウザを閉じてからターミナルで `Ctrl + C` を押します．
  To quit, close the browser and press `Ctrl + C` in the terminal.
- 新しい版に更新するときは，`AET-tutorials` フォルダで `git pull` → `uv sync` を実行します．
  To update to a new version, run `git pull` and then `uv sync` in the `AET-tutorials` folder.

## ノートブック / Notebooks

| ノートブック / Notebook | 内容 / Contents |
|---|---|
| [00_setup.ipynb](notebooks/00_setup.ipynb) | 環境構築と動作確認<br>Setup and check |
| [01_model_and_tilt_series.ipynb](notebooks/01_model_and_tilt_series.ipynb) | 直径 3 nm の FePt ナノ粒子の原子モデルと，ノイズを含む傾斜シリーズ（±70°, 2.5° 間隔）を作る<br>Atomic model of a 3 nm FePt nanoparticle and a noisy tilt series (±70°, 2.5° steps) |
| [02_alignment.ipynb](notebooks/02_alignment.ipynb) | 前処理：わざとずらし支持膜の背景を加えた傾斜シリーズの背景除去と位置合わせ（相互相関法・共通線法・重心法）<br>Preprocessing: background removal and alignment (cross-correlation, common line, centre of mass) of a deliberately shifted tilt series with a support-film background |
| [03_genfire.ipynb](notebooks/03_genfire.ipynb) | GENFIRE（フーリエ空間の反復再構成）<br>GENFIRE (Fourier-space iterative reconstruction) |
| [04_resire.ipynb](notebooks/04_resire.ipynb) | RESIRE（実空間の反復再構成）<br>RESIRE (real-space iterative reconstruction) |
| [05_atom_tracing.ipynb](notebooks/05_atom_tracing.ipynb) | 原子の抽出，元素の分類，正解との比較<br>Atom tracing, species classification and comparison with the ground truth |

各ノートブックは，ノート PC で 10〜30 秒程度で実行できます（メモリ 8 GB 以上を推奨）．GitHub 上でも実行結果つきで閲覧できます．

Each notebook runs in about 10–30 seconds on a laptop (8 GB of memory or more recommended). The notebooks can also be viewed with their outputs on GitHub.

## 構成 / Layout

```
AET-tutorials/
├── notebooks/        ノートブック / notebooks
├── src/
│   ├── genfire_cpu/  GENFIRE の CPU 実装 / CPU implementation of GENFIRE
│   ├── resire_cpu/   RESIRE の CPU 実装 / CPU implementation of RESIRE
│   └── aet_tools/    モデル・前処理・原子の抽出・表示・進捗表示 / model, preprocessing, atom tracing, plotting, live monitor
├── data/             ノートブックが作るデータ / data written by the notebooks
├── pyproject.toml    必要なパッケージ / required packages (read by uv)
└── uv.lock           パッケージのバージョン / pinned package versions (written by uv)
```

- `genfire_cpu` は GENFIRE-Python（https://github.com/genfire-em/GENFIRE-Python）と同じアルゴリズムを，大きな格子でも CPU で速く計算できるように実装し直したものです．
  `genfire_cpu` reimplements the GENFIRE-Python algorithm so that it runs fast on a CPU, even on large grids.
- `resire_cpu` は RESIRE（Pham *et al.*, 2023）のアルゴリズムを，1 軸傾斜の場合について NumPy / SciPy で実装したものです．
  `resire_cpu` implements the RESIRE algorithm (Pham *et al.*, 2023) for single-axis tilt series with NumPy / SciPy.

## 参考文献 / References

1. J. Miao, P. Ercius, S. J. L. Billinge, "Atomic electron tomography: 3D structures without crystals", *Science* **353**, aaf2157 (2016).
2. A. Pryor Jr. *et al.*, "GENFIRE: A generalized Fourier iterative reconstruction algorithm for high-resolution 3D imaging", *Sci. Rep.* **7**, 10409 (2017).
3. M. Pham *et al.*, "Accurate real space iterative reconstruction (RESIRE) algorithm for tomography", *Sci. Rep.* **13**, 5624 (2023).
4. Y. Yang *et al.*, "Deciphering chemical order/disorder and material properties at the single-atom level", *Nature* **542**, 75–79 (2017).

## ライセンス / License

GNU General Public License v3.0（[LICENSE](LICENSE)）．GENFIRE-Python と同じライセンスです．

Released under the GNU General Public License v3.0 (see [LICENSE](LICENSE)), the same licence as GENFIRE-Python.

## 作者 / Author

Tomokazu Yamamoto
