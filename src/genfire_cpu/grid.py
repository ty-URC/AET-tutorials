"""Assemble a 3D Fourier half-grid from 2D projections.

This is the other half of the 1024**3 problem.  Upstream
``fillInFourierGrid`` grows three python-level arrays with ``np.append``
inside a triple-nested shift loop -- every append copies the whole array, so
the cost is quadratic in the number of measured points -- then builds the
*full* complex128 grid and calls ``hermitianSymmetrize``, which allocates a
further ``(n+1)**3`` complex128 array.  At n=1024 that last step alone wants
17 GB, before anything else.

Here we

* keep only the rfft half-grid ``(n, n, n//2 + 1)``, folding each measured
  point onto its Hermitian mate as it is inserted (the projections are real,
  so a point and its mate carry identical information -- folding at insertion
  is exactly equivalent to upstream's fill-half-then-mirror, without the
  ``(n+1)**3`` temporary),
* append into geometrically-growing preallocated buffers, so gridding is
  linear rather than quadratic,
* accumulate with three ``np.bincount`` passes and keep the running sums in
  the working precision.

Grid layout is FFT order (origin at index 0), matching ``numpy.fft.rfftn``, so
no ``fftshift`` of a multi-gigabyte volume is ever needed.  Axis 0 is x, axis 1
is y, axis 2 is the halved z -- the same axis assignment upstream uses.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

__all__ = ["fill_fourier_grid", "euler_to_matrix", "radial_frequency_halfgrid",
           "radial_at"]


def euler_to_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """GENFIRE's ZYZ-style Euler convention, angles in degrees.

    Kept bit-for-bit identical to upstream ``fillInFourierGrid`` (including the
    final transpose) so that angle files transfer between the two codes.
    """
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.array([
        [cps * cth * cph - sps * sph, cps * cth * sph + sps * cph, -cps * sth],
        [-sps * cth * cph - cps * sph, -sps * cth * sph + cps * cph, sps * sth],
        [sth * cph, sth * sph, cth],
    ])
    return R.T


class _Appender:
    """A 1D array that grows without ``np.append``'s quadratic copying.

    Doubling while the buffer is small, then in fixed steps.  Pure doubling
    would leave the final capacity anywhere up to 2N for N contributions, and
    a reallocation holds the old buffer and the new one at once, so the
    transient peak reaches 3N -- 7.5 GiB for the three buffers at 1536**3.
    Switching to a fixed step caps the capacity at N + step, which brings the
    transient to the 2N that any growable array has to pay.
    """

    STEP = 1 << 25          # elements; 0.6 GiB across index+value+weight

    def __init__(self, dtype, capacity: int = 1 << 20):
        self._buf = np.empty(capacity, dtype=dtype)
        self._n = 0

    def _grow_to(self, need: int) -> int:
        cap = self._buf.size
        while cap < need:
            cap = cap * 2 if cap < self.STEP else cap + self.STEP
        return cap

    def extend(self, values: np.ndarray) -> None:
        need = self._n + values.size
        if need > self._buf.size:
            grown = np.empty(self._grow_to(need), dtype=self._buf.dtype)
            grown[:self._n] = self._buf[:self._n]
            self._buf = grown
        self._buf[self._n:need] = values
        self._n = need

    def take(self) -> np.ndarray:
        """Return the filled part and release the buffer."""
        out = self._buf[:self._n]
        self._buf = np.empty(0, dtype=self._buf.dtype)
        return out

    def __len__(self) -> int:
        return self._n


def radial_frequency_halfgrid(n: int, dtype=np.float32) -> np.ndarray:
    """|k| on the rfft half-grid, normalised so 1.0 is Nyquist.

    Built one x-plane at a time; the full ``meshgrid`` upstream uses would cost
    three ``n**3`` float64 arrays (25 GB at n=1024).
    """
    nh = n // 2 + 1
    ax = np.fft.fftfreq(n, d=0.5).astype(dtype)  # -1 .. 1 in Nyquist units
    az = ax[:nh].copy()
    az[-1] = abs(az[-1])  # rfftfreq's Nyquist bin is positive
    out = np.empty((n, n, nh), dtype=dtype)
    yz = (ax[:, None] ** 2 + az[None, :] ** 2)  # (n, nh)
    for i in range(n):
        np.sqrt(ax[i] ** 2 + yz, out=out[i])
    return out


def radial_at(lin: np.ndarray, n: int, nh: int, dtype=np.float32) -> np.ndarray:
    """|k| / k_Nyquist at the given half-grid linear indices.

    The same values :func:`radial_frequency_halfgrid` would give, evaluated
    only where they are wanted.  Only a few percent of the half-grid is ever
    measured, so this is 0.08 GiB at 1024**3 against 2.0 GiB for the full map,
    and 0.3 against 6.8 at 1536**3.

    The summation order matches the full map's -- ``kx**2 + (ky**2 + kz**2)``
    -- so the two agree to the last bit, and a point cannot land in a
    different R-free shell depending on which routine produced its radius.
    """
    ax = np.fft.fftfreq(n, d=0.5).astype(dtype)
    az = ax[:nh].copy()
    az[-1] = abs(az[-1])
    kx, rem = np.divmod(lin, n * nh)
    ky, kz = np.divmod(rem, nh)
    r = ax[ky] ** 2
    r += az[kz] ** 2
    r += ax[kx] ** 2
    return np.sqrt(r, out=r)


def _accumulate_sparse(lin, vals, wgts, size, cdtype, dtype, chunk_bytes):
    """Like :func:`_accumulate`, but emit only the points that got a value.

    At 1024**3 a tilt series fills 3.9% of the half-grid, at 1536**3 2.6% and
    at 2048**3 under 2%.  The dense form writes a 4 GiB grid (32 GiB at
    2048**3) that the caller immediately compresses to a list of addresses and
    throws away.  Emitting the list directly skips the grid and its mask
    entirely, and is *faster*: measured at 1024**3, the output stage plus the
    Nyquist cut plus the caller's compression went from 13.58 s to 3.23 s,
    because the dense path writes and re-reads 4.5 GiB of which 96% is zero.

    Chunks are visited in increasing index order and ``flatnonzero`` returns
    sorted positions, so the concatenated output is sorted exactly as
    ``np.flatnonzero`` on the dense mask would be.
    """
    out_lin = _Appender(np.int64)
    out_val = _Appender(cdtype)
    per_chunk = max(1, int(chunk_bytes) // 8)

    for start in range(0, size, per_chunk):
        stop = min(start + per_chunk, size)
        sel = (lin >= start) & (lin < stop)
        if not sel.any():
            continue
        idx = lin[sel]
        idx -= start
        w = wgts[sel]
        v = vals[sel]
        width = stop - start

        den = np.bincount(idx, weights=w, minlength=width).astype(dtype)
        hit = np.flatnonzero(den)
        if hit.size == 0:
            continue

        acc = np.bincount(idx, weights=w * v.real, minlength=width)
        block = np.empty(hit.size, dtype=cdtype)
        block.real = acc[hit].astype(dtype)
        acc = np.bincount(idx, weights=w * v.imag, minlength=width)
        block.imag = acc[hit].astype(dtype)
        del acc, idx, w, v

        block /= den[hit]
        hit += start
        out_lin.extend(hit)
        out_val.extend(block)
        del den, hit, block

    return out_lin.take(), out_val.take()


def _truncate_beyond_nyquist_sparse(lin, val, n, dtype):
    """Drop the emitted points past |k| = Nyquist."""
    keep = radial_at(lin, n, n // 2 + 1, dtype) <= 1.0
    return lin[keep], val[keep]


def _symmetrize_self_conjugate_planes_sparse(lin, val, n, nh):
    """Hermitian symmetry within the kz=0 and kz=Nyquist planes, on a list.

    Those two planes are their own Hermitian mates, so the half-grid is only a
    valid spectrum of a real volume if each is symmetric within itself.  The
    dense version reverses whole planes; here the mate of a point is found by
    index arithmetic and located with a binary search.

    A point whose mate is missing has to be *added*, which is why the result
    is merged back rather than written in place.  In practice the gridder
    folds Hermitian mates as it inserts, so there are few or none.
    """
    if lin.size == 0:
        return lin, val
    kz = lin % nh
    on_plane = np.flatnonzero((kz == 0) | (kz == nh - 1))
    if on_plane.size == 0:
        return lin, val

    sub = lin[on_plane]
    kx, rem = np.divmod(sub, n * nh)
    ky, kz_sub = np.divmod(rem, nh)
    mate = ((-kx) % n) * (n * nh) + ((-ky) % n) * nh + kz_sub

    where = np.searchsorted(lin, mate)
    np.clip(where, 0, lin.size - 1, out=where)
    found = lin[where] == mate

    # both present: each takes the average of itself and its mate's conjugate
    have = np.flatnonzero(found)
    if have.size:
        pos = on_plane[have]
        val[pos] = 0.5 * (val[pos] + np.conj(val[where[have]]))

    # mate missing: it has to appear, carrying the conjugate
    missing = np.flatnonzero(~found)
    if missing.size == 0:
        return lin, val
    add_lin = mate[missing]
    add_val = np.conj(val[on_plane[missing]])
    order = np.argsort(add_lin, kind="stable")
    add_lin, add_val = add_lin[order], add_val[order]
    at = np.searchsorted(lin, add_lin)
    return np.insert(lin, at, add_lin), np.insert(val, at, add_val)


def _symmetrize_self_conjugate_planes(measured_k: np.ndarray,
                                      mask: np.ndarray) -> None:
    """Enforce Hermitian symmetry within the z=0 and z=Nyquist planes.

    Those two planes are their own Hermitian mates, so an rfft half-grid is
    only a valid spectrum of a real volume if each is symmetric within itself.
    Everywhere else the mate lives in the discarded half and folding at
    insertion already guarantees it.  The residual asymmetry here is tiny --
    it comes from ``round`` breaking ties towards even, which is not an odd
    function -- but it costs only n**2 to remove.
    """
    n = measured_k.shape[0]
    rev = (-np.arange(n)) % n
    for z in {0, measured_k.shape[2] - 1}:
        plane = measured_k[:, :, z]
        mplane = mask[:, :, z]
        conj = np.conj(plane[rev][:, rev])
        mconj = mplane[rev][:, rev]
        both = mplane & mconj
        plane[both] = 0.5 * (plane[both] + conj[both])
        only_conj = mconj & ~mplane
        plane[only_conj] = conj[only_conj]
        mplane |= mconj


def _accumulate(lin, vals, wgts, size, cdtype, dtype, chunk_bytes):
    """Inverse-distance-weighted average per grid point, in bounded memory.

    ``np.bincount`` allocates a float64 array of the full output size to
    produce a float32 result -- 4.0 GiB at 1024**3, 13.5 GiB at 1536**3, and
    three of them in sequence.  Chunking the *output index range* bounds that.

    It changes no value at all.  Every grid point belongs to exactly one chunk,
    and boolean selection preserves order, so each bin is summed from the same
    terms in the same sequence, in the same float64 accumulator, and rounded to
    float32 at the same point.  ``chunk_bytes`` therefore trades memory against
    the number of passes over the contribution list and nothing else.

    Two costs pull against each other.  Each chunk scans the whole contribution
    list to find its own, so fewer, larger chunks scan less; but each chunk
    also scatters randomly into its accumulator, and that gets sharply slower
    once the accumulator grows past about half a gigabyte.  Measured at
    1024**3, 21 projections, accumulation only::

        chunk    chunks   time
         16 MB      257   50.0 s
         64 MB       65   27.3 s
        256 MB       17   24.4 s
        512 MB        9   23.3 s
       1024 MB        5  106.3 s     <- cliff

    So the default sits on the broad plateau and the size, not the count, is
    the thing to tune.  Trying it the other way round -- capping the chunk
    count so the scanning would not grow with the grid -- pushed a 1536**3
    grid to 0.85 GiB chunks and made gridding 2.2x slower (175 s -> 390 s).

    Only one float64 accumulator is alive at a time and the result is written
    straight into ``measured_k``; holding all three at once cost 3.8 GiB per
    chunk at 1536**3 instead of 1.8.
    """
    measured_k = np.zeros(size, dtype=cdtype)
    mask = np.zeros(size, dtype=bool)
    per_chunk = max(1, int(chunk_bytes) // 8)

    for start in range(0, size, per_chunk):
        stop = min(start + per_chunk, size)
        sel = (lin >= start) & (lin < stop)
        if not sel.any():
            continue
        idx = lin[sel]
        idx -= start
        w = wgts[sel]
        v = vals[sel]
        width = stop - start

        den = np.bincount(idx, weights=w, minlength=width).astype(dtype)
        ok = den > 0

        out = measured_k[start:stop]            # a view, not a copy
        acc = np.bincount(idx, weights=w * v.real, minlength=width)
        out.real = acc.astype(dtype)
        acc = np.bincount(idx, weights=w * v.imag, minlength=width)
        out.imag = acc.astype(dtype)
        del acc, idx, w, v

        out[ok] /= den[ok]
        mask[start:stop] = ok
        del den, ok

    return measured_k, mask


def _truncate_beyond_nyquist(measured_k, mask, n, dtype):
    """Zero everything past |k| = Nyquist, one x-plane at a time.

    The full radial map is an ``n**2 (n/2+1)`` array -- 2.0 GiB at 1024**3,
    6.8 GiB at 1536**3 -- built to serve a comparison that only ever looks at
    one plane.  Same arithmetic, same dtype, same rounding as
    :func:`radial_frequency_halfgrid`, so the same points are kept.
    """
    nh = n // 2 + 1
    ax = np.fft.fftfreq(n, d=0.5).astype(dtype)
    az = ax[:nh].copy()
    az[-1] = abs(az[-1])
    yz = ax[:, None] ** 2 + az[None, :] ** 2
    for i in range(n):
        outside = np.sqrt(ax[i] ** 2 + yz) > 1.0
        measured_k[i][outside] = 0
        mask[i][outside] = False


def fill_fourier_grid(
    projections: np.ndarray,
    angles: np.ndarray,
    n: int,
    interpolation_cutoff: float = 0.7,
    grid_multiple: bool = True,
    enforce_resolution_circle: bool = True,
    dtype=np.float32,
    chunk_bytes: int = 1 << 28,
    sparse: bool = False,
    threads: int = 1,
    verbose: bool = True,
    progress=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid ``projections`` into an rfft-layout Fourier half-volume.

    Parameters
    ----------
    projections : (n, n, n_proj) real array, already zero-padded to the grid
        size ``n``.
    angles : (n_proj, 3) Euler angles in degrees.
    n : edge length of the (cubic) reconstruction grid.
    interpolation_cutoff : kernel radius in pixels.
    grid_multiple : allow a measured point to reach neighbouring grid points.
    enforce_resolution_circle : zero everything beyond Nyquist.
    chunk_bytes : size of the float64 accumulator used while averaging.  Only
        affects memory and speed; the result is identical for any value.  See
        :func:`_accumulate` for the measured curve -- the plateau runs from
        about 64 MB to 512 MB.
    sparse : return ``(lin, values)`` for the filled points instead of the
        dense half-grid and its mask.  Same points, same values, in the same
        order ``np.flatnonzero`` would give -- but the dense grid is never
        allocated, which is 4.5 GiB at 1024**3 and 36 GiB at 2048**3.
    threads : worker threads for the per-projection collection, which is
        embarrassingly parallel.  Results are consumed in projection order, so
        contributions land in the buffers in the same sequence whatever the
        thread count and the answer does not change.  Measured 3.4x at 12
        threads on 41 projections at 1024**3; it saturates around 6 because
        NumPy holds the GIL for fancy indexing and allocation.

    Returns
    -------
    dense (default) : ``(measured_k, measured_mask)`` -- an
        ``(n, n, n//2+1)`` complex half-grid and the bool mask of filled points.
    sparse : ``(lin, values)`` -- sorted int64 linear indices into that same
        half-grid, and the complex value at each.
    """
    dtype = np.dtype(dtype)
    cdtype = np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
    t0 = time.perf_counter()

    if projections.ndim == 2:
        projections = projections[:, :, None]
    if projections.shape[0] != n or projections.shape[1] != n:
        raise ValueError("projections must already be padded to %d x %d, got %r"
                         % (n, n, projections.shape[:2]))
    n_proj = projections.shape[2]
    angles = np.asarray(angles, dtype=np.float64).reshape(n_proj, 3)

    n2 = n // 2
    nh = n2 + 1
    cutoff2 = float(interpolation_cutoff) ** 2
    shift_max = int(round(interpolation_cutoff)) if grid_multiple else 0
    shifts = range(-shift_max, shift_max + 1)

    # centred 2D k-coordinates of one projection slice, F-ordered to match the
    # F-ordered ravel used below
    kx2, ky2 = np.meshgrid(np.arange(-n2, n2, dtype=np.float64),
                           np.arange(-n2, n2, dtype=np.float64), indexing="ij")
    kx2 = kx2.ravel(order="F")
    ky2 = ky2.ravel(order="F")

    idx_buf = _Appender(np.int64)
    val_buf = _Appender(cdtype)
    wgt_buf = _Appender(dtype)

    def collect(p):
        """Every contribution one projection makes, in shift order.

        Pure: it reads the shared inputs and returns its own arrays, which is
        what lets the projections run on separate threads.
        """
        produced = []
        # centred 2D FFT of this projection -> the central slice of its 3D FT
        slice_k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
            projections[:, :, p].astype(np.float64)))).ravel(order="F")

        R = euler_to_matrix(*angles[p])
        gx = R[0, 0] * kx2 + R[0, 1] * ky2
        gy = R[1, 0] * kx2 + R[1, 1] * ky2
        gz = R[2, 0] * kx2 + R[2, 1] * ky2

        rx, ry, rz = np.round(gx), np.round(gy), np.round(gz)
        for sx in shifts:
            tx = rx + sx
            dx2 = (gx - tx) ** 2
            keep_x = (tx >= -n2) & (tx <= n2 - 1) & (dx2 <= cutoff2)
            if not keep_x.any():
                continue
            for sy in shifts:
                ty = ry + sy
                d2xy = dx2 + (gy - ty) ** 2
                keep_xy = keep_x & (ty >= -n2) & (ty <= n2 - 1) & (d2xy <= cutoff2)
                if not keep_xy.any():
                    continue
                for sz in shifts:
                    tz = rz + sz
                    d2 = d2xy + (gz - tz) ** 2
                    keep = keep_xy & (tz >= -n2) & (tz <= n2 - 1) & (d2 <= cutoff2)
                    if not keep.any():
                        continue

                    cx = tx[keep].astype(np.int64)
                    cy = ty[keep].astype(np.int64)
                    cz = tz[keep].astype(np.int64)
                    vals = slice_k[keep]

                    # fold onto the Hermitian mate when the point lands in the
                    # negative-z half; the mate carries the conjugate value
                    neg = cz < 0
                    if neg.any():
                        cx = np.where(neg, -cx, cx)
                        cy = np.where(neg, -cy, cy)
                        cz = np.where(neg, -cz, cz)
                        vals = np.where(neg, np.conj(vals), vals)

                    lin = (cx % n) * (n * nh) + (cy % n) * nh + cz
                    # inverse-distance weighting, as upstream (the epsilon keeps
                    # an exact grid hit from becoming infinite)
                    w = 1.0 / (d2[keep] + 1e-5)

                    produced.append((lin, vals.astype(cdtype), w.astype(dtype)))
        return produced

    def store(p, produced):
        for lin_p, val_p, wgt_p in produced:
            idx_buf.extend(lin_p)
            val_buf.extend(val_p)
            wgt_buf.extend(wgt_p)
        if progress is not None:
            progress(p + 1, n_proj)
        elif verbose and (p + 1) % 25 == 0:
            print("  gridded %d/%d projections (%d points so far)"
                  % (p + 1, n_proj, len(idx_buf)))

    workers = max(1, int(threads))
    if workers == 1:
        for p in range(n_proj):
            store(p, collect(p))
    else:
        # submit in batches so at most a couple of rounds of results are held
        # at once; consuming in order keeps the buffers deterministic
        batch = 2 * workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for base in range(0, n_proj, batch):
                idx = range(base, min(base + batch, n_proj))
                for p, produced in zip(idx, pool.map(collect, idx)):
                    store(p, produced)

    lin = idx_buf.take()
    vals = val_buf.take()
    wgts = wgt_buf.take()
    if verbose:
        print("  %d contributions collected in %.1f s"
              % (lin.size, time.perf_counter() - t0))

    size = n * n * nh

    if sparse:
        out_lin, out_val = _accumulate_sparse(lin, vals, wgts, size, cdtype,
                                              dtype, chunk_bytes)
        del lin, vals, wgts
        if enforce_resolution_circle:
            out_lin, out_val = _truncate_beyond_nyquist_sparse(
                out_lin, out_val, n, dtype)
        out_lin, out_val = _symmetrize_self_conjugate_planes_sparse(
            out_lin, out_val, n, nh)
        np.nan_to_num(out_val, copy=False)
        if verbose:
            print("Fourier grid assembled in %.1f s -- %d/%d half-grid points "
                  "filled (%.2f%%)"
                  % (time.perf_counter() - t0, out_lin.size, size,
                     100.0 * out_lin.size / size))
        return out_lin, out_val

    measured_k, mask = _accumulate(lin, vals, wgts, size, cdtype, dtype,
                                   chunk_bytes)
    del lin, vals, wgts

    measured_k = measured_k.reshape(n, n, nh)
    mask = mask.reshape(n, n, nh)

    if enforce_resolution_circle:
        _truncate_beyond_nyquist(measured_k, mask, n, dtype)

    _symmetrize_self_conjugate_planes(measured_k, mask)

    # a gridded point can still be exactly zero; keep the mask authoritative
    np.nan_to_num(measured_k, copy=False)

    if verbose:
        filled = int(mask.sum())
        print("Fourier grid assembled in %.1f s -- %d/%d half-grid points filled "
              "(%.2f%%)" % (time.perf_counter() - t0, filled, mask.size,
                            100.0 * filled / mask.size))
    return measured_k, mask
