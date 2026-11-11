"""An analytic test object and its exact projections.

A sum of uniform spheres has a closed-form line integral -- a chord length
``2*sqrt(r^2 - d^2)`` -- so its projections can be produced at any orientation
without ever building the 3D volume.  That matters at the sizes this package
targets: a Fourier-slice forward projector for a 512**3 object needs a 1024**3
complex128 array (17 GB) before it can start, while this needs none.

Used for benchmarks and for end-to-end tests where the ground truth has to be
known exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .grid import euler_to_matrix

__all__ = ["Spheres", "random_cluster", "atomic_lattice",
           "fcc_alloy_nanoparticle", "copt_nanoparticle",
           "mackay_icosahedron", "icosahedral_alloy_nanoparticle"]


class Spheres:
    """A set of uniform spheres, in voxel units, centred on the box centre."""

    def __init__(self, centres: np.ndarray, radii: np.ndarray,
                 amplitudes: Optional[np.ndarray] = None):
        self.centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
        self.radii = np.broadcast_to(np.asarray(radii, dtype=np.float64),
                                     (self.centres.shape[0],)).copy()
        if amplitudes is None:
            amplitudes = np.ones(self.centres.shape[0])
        self.amplitudes = np.broadcast_to(
            np.asarray(amplitudes, dtype=np.float64),
            (self.centres.shape[0],)).copy()

    def __len__(self) -> int:
        return self.centres.shape[0]

    # -- ground truth -----------------------------------------------------
    def volume(self, n: int, dtype=np.float32) -> np.ndarray:
        """Rasterise onto an ``n**3`` grid (memory-hungry; small n only)."""
        out = np.zeros((n, n, n), dtype=dtype)
        c = n / 2.0
        for (cx, cy, cz), r, a in zip(self.centres, self.radii, self.amplitudes):
            lo = [max(0, int(np.floor(v + c - r))) for v in (cx, cy, cz)]
            hi = [min(n, int(np.ceil(v + c + r)) + 1) for v in (cx, cy, cz)]
            if any(l >= h for l, h in zip(lo, hi)):
                continue
            gx = np.arange(lo[0], hi[0]) - c - cx
            gy = np.arange(lo[1], hi[1]) - c - cy
            gz = np.arange(lo[2], hi[2]) - c - cz
            d2 = (gx[:, None, None] ** 2 + gy[None, :, None] ** 2
                  + gz[None, None, :] ** 2)
            out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += a * (d2 <= r * r)
        return out

    # -- exact projections -------------------------------------------------
    def project(self, angles: np.ndarray, n: int, dtype=np.float32,
                verbose: bool = False) -> np.ndarray:
        """Exact parallel-beam projections onto ``n`` x ``n`` images.

        The projection axis matches :func:`genfire_cpu.grid.euler_to_matrix`,
        so the output grids straight back onto the slice it came from.
        """
        angles = np.atleast_2d(np.asarray(angles, dtype=np.float64))
        out = np.zeros((n, n, angles.shape[0]), dtype=np.float64)
        c = n / 2.0
        rmax = float(self.radii.max())
        span = int(np.ceil(rmax)) + 1
        off = np.arange(-span, span + 1)
        ox, oy = np.meshgrid(off, off, indexing="ij")
        ox = ox.ravel()
        oy = oy.ravel()

        for i, (phi, theta, psi) in enumerate(angles):
            # the gridder maps a detector coordinate (kx, ky) to R @ (kx, ky, 0),
            # so the detector axes are columns 0 and 1 of R
            R = euler_to_matrix(phi, theta, psi)
            px = self.centres @ R[:, 0]
            py = self.centres @ R[:, 1]

            ix = np.round(px + c).astype(np.int64)
            iy = np.round(py + c).astype(np.int64)
            plane = out[:, :, i]
            for dx, dy in zip(ox, oy):
                gx = ix + dx
                gy = iy + dy
                d2 = (gx - c - px) ** 2 + (gy - c - py) ** 2
                inside = (d2 < self.radii ** 2) & (gx >= 0) & (gx < n) \
                    & (gy >= 0) & (gy < n)
                if not inside.any():
                    continue
                chord = 2.0 * np.sqrt(self.radii[inside] ** 2 - d2[inside])
                np.add.at(plane, (gx[inside], gy[inside]),
                          self.amplitudes[inside] * chord)
            if verbose and (i + 1) % 20 == 0:
                print("  projected %d/%d" % (i + 1, angles.shape[0]))
        return out.astype(dtype)


def random_cluster(n_spheres: int = 200, box: float = 100.0,
                   radius: Tuple[float, float] = (3.0, 8.0),
                   seed: int = 0) -> Spheres:
    """Random non-overlapping-ish spheres inside a cube of side ``box`` voxels."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-box / 2, box / 2, size=(n_spheres, 3))
    radii = rng.uniform(radius[0], radius[1], size=n_spheres)
    amps = rng.uniform(0.5, 1.0, size=n_spheres)
    return Spheres(centres, radii, amps)


def atomic_lattice(box_px: float, spacing_px: float, radius_px: float = 1.2,
                   jitter_px: float = 0.0, seed: int = 0,
                   sphere_fraction: float = 1.0) -> Spheres:
    """A cubic lattice of small spheres -- a stand-in for an atomic model.

    At 0.3 A per pixel, ``spacing_px=8`` is a 2.4 A lattice constant and
    ``radius_px=1.2`` a 0.36 A atom, which is roughly what atomic electron
    tomography reconstructions resolve.
    """
    rng = np.random.default_rng(seed)
    k = int(box_px // spacing_px)
    ax = (np.arange(k) - (k - 1) / 2) * spacing_px
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    # keep a sphere of atoms, not a cube, so the support is meaningful
    keep = np.linalg.norm(centres, axis=1) <= box_px / 2
    centres = centres[keep]
    if sphere_fraction < 1.0:
        take = rng.random(centres.shape[0]) < sphere_fraction
        centres = centres[take]
    if jitter_px:
        centres = centres + rng.normal(0, jitter_px, centres.shape)
    return Spheres(centres, np.full(centres.shape[0], radius_px))


def fcc_alloy_nanoparticle(diameter_px: float, lattice_px: float,
                           amplitudes: Sequence[float] = (27.0, 78.0),
                           fractions: Sequence[float] = (0.5, 0.5),
                           radius_px: float = 1.5, jitter_px: float = 0.0,
                           seed: int = 0) -> Tuple["Spheres", np.ndarray]:
    """A chemically disordered FCC alloy particle -- an A1 CoPt by default.

    Every FCC site inside the sphere gets one species drawn independently, so
    there is no chemical order: this is the A1 phase, the case atomic electron
    tomography of binary alloys actually has to solve.  Species are told apart
    only by their amplitude, which defaults to the atomic numbers of Co and Pt
    (27 and 78) -- enough of a contrast to be classifiable, which is the point
    of the exercise.  A real scattering model would not be a plain Z ratio,
    but nothing here depends on the exact value.

    Returns the :class:`Spheres` and a per-atom species index, so a
    reconstruction can be scored on whether it separates the two.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.size != len(amplitudes):
        raise ValueError("need one fraction per species")
    fractions = fractions / fractions.sum()

    radius = diameter_px / 2.0
    reach = int(np.ceil(radius / lattice_px)) + 1
    cells = np.arange(-reach, reach + 1) * lattice_px
    cx, cy, cz = np.meshgrid(cells, cells, cells, indexing="ij")
    corners = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * lattice_px
    centres = (corners[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    centres = centres[np.linalg.norm(centres, axis=1) <= radius]

    rng = np.random.default_rng(seed)
    species = rng.choice(len(amplitudes), size=centres.shape[0], p=fractions)
    if jitter_px:
        centres = centres + rng.normal(0, jitter_px, centres.shape)
    amps = np.asarray(amplitudes, dtype=np.float64)[species]
    return Spheres(centres, np.full(centres.shape[0], radius_px), amps), species


def copt_nanoparticle(pixel_size: float = 0.3, diameter_a: float = 50.0,
                      lattice_a: float = 3.75, atom_radius_a: float = 0.45,
                      jitter_a: float = 0.0, seed: int = 0
                      ) -> Tuple["Spheres", np.ndarray]:
    """A 5 nm disordered CoPt particle, in pixels, at ``pixel_size`` angstroms.

    ``lattice_a`` is the A1 CoPt cubic lattice constant.  ``atom_radius_a`` is
    kept just under the point where a uniform sphere's first Fourier zero
    crosses Nyquist, so the projections are not badly aliased.
    """
    return fcc_alloy_nanoparticle(
        diameter_px=diameter_a / pixel_size,
        lattice_px=lattice_a / pixel_size,
        radius_px=atom_radius_a / pixel_size,
        jitter_px=jitter_a / pixel_size,
        seed=seed)


#: The twelve vertex directions of a regular icosahedron, unit length.
def _icosahedron_vertices() -> np.ndarray:
    phi = (1 + np.sqrt(5)) / 2
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v[0])


def _icosahedron_faces(vertices: np.ndarray) -> np.ndarray:
    """The twenty triples of mutually adjacent vertices."""
    edge = np.min([np.linalg.norm(vertices[0] - w)
                   for w in vertices[1:]])
    faces = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.linalg.norm(vertices[i] - vertices[j]) - edge) > 1e-6:
                continue
            for k in range(j + 1, n):
                if (abs(np.linalg.norm(vertices[i] - vertices[k]) - edge) < 1e-6
                        and abs(np.linalg.norm(vertices[j] - vertices[k]) - edge) < 1e-6):
                    faces.append((i, j, k))
    return np.array(faces)


def mackay_icosahedron(shells: int, spacing_px: float) -> np.ndarray:
    """Atom positions of a Mackay icosahedron, in voxel units.

    Twenty tetrahedral units share the centre, each spanned by three adjacent
    icosahedral vertex directions.  Because those directions are unit length
    while the icosahedron's edge is 1.0515 of its circumradius, the in-shell
    neighbour spacing comes out about 5% longer than the radial one -- the
    intrinsic strain that makes a real icosahedral particle, and the reason
    this is a harder reconstruction target than a single-crystal sphere: five
    twin boundaries meet at every vertex and the surface is twenty {111}
    facets rather than a sphere.

    ``shells`` counts the atomic shells, so the particle is ``2 * shells``
    spacings across; the atom count is (10 k^3 + 15 k^2 + 11 k + 3) / 3.
    """
    verts = _icosahedron_vertices()
    faces = _icosahedron_faces(verts)
    out = [np.zeros((1, 3))]
    for a, b, c in faces:
        A, B, C = verts[a], verts[b], verts[c]
        for n in range(1, shells + 1):
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    k = n - i - j
                    out.append(((i * A + j * B + k * C) * spacing_px)[None, :])
    pts = np.concatenate(out, axis=0)
    # atoms on shared faces and edges are generated by more than one unit
    keys = np.round(pts / (spacing_px * 1e-3)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(keep)]


def icosahedral_alloy_nanoparticle(diameter_px: float, spacing_px: float,
                                   amplitudes: Sequence[float] = (27.0, 78.0),
                                   fractions: Sequence[float] = (0.5, 0.5),
                                   radius_px: float = 1.5, jitter_px: float = 0.0,
                                   seed: int = 0) -> Tuple["Spheres", np.ndarray]:
    """A Mackay icosahedron with the two species drawn independently per site."""
    shells = max(1, int(round(diameter_px / (2 * spacing_px))))
    centres = mackay_icosahedron(shells, spacing_px)
    rng = np.random.default_rng(seed)
    fractions = np.asarray(fractions, dtype=np.float64)
    fractions = fractions / fractions.sum()
    species = rng.choice(len(amplitudes), size=centres.shape[0], p=fractions)
    if jitter_px:
        centres = centres + rng.normal(0, jitter_px, centres.shape)
    amps = np.asarray(amplitudes, dtype=np.float64)[species]
    return Spheres(centres, np.full(centres.shape[0], radius_px), amps), species
