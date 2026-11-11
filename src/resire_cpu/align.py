"""Translational alignment, before any angle is touched.

Projection matching cannot be trusted to do this.  Measured on this code, a
2 px translation error fed into the joint refinement came back as a 1.0 degree
*angle* error that was not there to begin with: rotating a projection moves
most of its mass sideways, so a shift can be partly explained as a rotation
and the search takes that explanation.  Both estimates end up wrong.

So translation is removed first, by two methods that do not involve the angles
at all -- which is exactly why the Miao group ships them as a separate step and
tells users to run them before GENFIRE's angular refinement.

Along the tilt axis, the constraint is exact.  For a rotation about y,

    sum over x of P_theta(x, y) = integral of rho over x and z = m(y)

is the same curve for every projection, because the integral is taken over the
whole plane the rotation acts in.  Every projection therefore carries the same
1D profile, and misalignment is whatever displaces them.  This is the real
space form of the Fourier statement that all central slices share the tilt
axis line -- the "common line".  Summing n_x pixels also multiplies the signal
by n_x and the noise by sqrt(n_x), so the curves being correlated have 16 times
the SNR of the image at n = 256.

Perpendicular to it there is no such invariant, and this is the hard
direction.  The centre of mass is the usual estimator and it is fragile: its
noise variance goes as ``sum(x_i^2)`` over *every* pixel, so empty background
far from the object dominates the error, and a background offset biases it
outright.  Two things help, and it is worth being clear about which does what.
Masking to the object cuts ``sum(x_i^2)`` and genuinely reduces the noise.
Fitting a sinusoid across the tilt series does not reduce it -- it corrects a
*model* error, since an object off the rotation axis has a centre of mass that
really does move as ``x0 cos(theta) + z0 sin(theta)``, and forcing every
projection to a single constant would fight that real motion.

Neither gets to the precision a reconstruction can.  Both use one number per
projection -- a first moment, or a 1D profile -- where projection matching
uses the whole image.  These exist to remove the bulk translation so that the
angular refinement is no longer degenerate; polishing the remainder is a job
for matching against a reconstruction, with the angles held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from scipy.ndimage import rotate as _ndrotate
from scipy.optimize import least_squares as _lsq

from scipy.ndimage import gaussian_filter as _gaussian_filter
from scipy.ndimage import map_coordinates as _map_coordinates

from .refine import _ncc_over_shifts, _parabolic, shift_projections

__all__ = ["AlignmentResult", "Geometry", "common_line_shifts",
           "cross_correlation_shifts",
           "centre_of_mass_shifts", "align_projections",
           "fit_geometry", "rotate_projections",
           "out_of_plane_signature"]


@dataclass
class Geometry:
    """What the sinusoidal part of the misalignment says about the setup.

    Under a rotation about y, a centroid at ``(x0, y0, z0)`` projects to
    ``x0 cos(theta) + z0 sin(theta) + c`` on the detector.  Those coefficients
    are not error -- they are the object's position relative to the rotation
    axis, and removing them is exactly a rigid 3D translation, not an
    approximation.

    Motion along the tilt axis is a different matter, and two separate errors
    put it there.  An in-plane inclination ``psi`` turns every image by the
    same angle.  An out-of-plane tip ``alpha`` -- the axis leaning toward the
    beam -- turns image j by ``alpha sin(theta_j)`` and tilts the object by
    ``alpha (cos(theta_j) - 1)`` on top of it, both of which move the centroid
    axially.  Either is a geometry error, and a plain common-line alignment
    shifts both away without reporting them.

    They are read by fitting the geometry, not by matching harmonics.  A first
    version separated them by harmonic order -- psi first, alpha second -- and
    counted only one of alpha's two contributions, reading an injected 3
    degrees as 1.4.  Predicting where the centroid actually lands costs no more
    and needs no approximation.
    """

    swing: float             #: centroid excursion the estimates rest on, px
    x0: float                #: centroid offset across the tilt axis, px
    z0: float                #: centroid height along the beam, px
    centre: float            #: where the rotation axis lands on the detector
    tilt_axis_deg: float     #: in-plane inclination, degrees
    out_of_plane_deg: float  #: axis tipped toward the beam, degrees
    perp_residual: float     #: rms of the fit residual across the axis, px
    axial_residual: float    #: rms of the fit residual along the axis, px

    @property
    def trustworthy(self) -> bool:
        """Whether the object swings far enough for the angles to mean
        anything.  Both are read from how that excursion divides between
        the detector axes; without one there is nothing to divide."""
        return self.swing >= 3.0

    def summary(self) -> str:
        if not self.trustworthy:
            return ("centroid swing %.2f px -- too small to read the tilt "
                    "axis from; offset the specimen from the rotation axis"
                    % self.swing)
        return "\n".join([
            "centroid off-axis    : x0 %.2f px, z0 %.2f px  (radius %.2f px)"
            % (self.x0, self.z0, np.hypot(self.x0, self.z0)),
            "rotation axis on det : %.2f px" % self.centre,
            "tilt axis, in plane  : %.3f deg" % self.tilt_axis_deg,
            "tilt axis, toward beam: %.3f deg" % self.out_of_plane_deg,
            "unmodelled scatter   : %.3f px perp, %.3f px axial"
            % (self.perp_residual, self.axial_residual),
        ])


def _axis(alpha_deg: float) -> np.ndarray:
    """The stage's rotation axis, tipped ``alpha`` from y toward the beam."""
    a = np.deg2rad(alpha_deg)
    return np.array([0.0, np.cos(a), np.sin(a)])


def _turn(axis: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Rotation matrices about ``axis``, one per angle.  Returns (n, 3, 3)."""
    t = np.deg2rad(np.asarray(deg, dtype=np.float64))[:, None, None]
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def _centroid_track(alpha, psi, x0, z0, cx, cy, theta):
    """Where the centroid lands on the detector under this geometry."""
    pts = _turn(_axis(alpha), theta) @ np.array([x0, 0.0, z0])
    b = np.deg2rad(psi)
    px, py = pts[:, 0], pts[:, 1]
    return (px * np.cos(b) - py * np.sin(b) + cx,
            px * np.sin(b) + py * np.cos(b) + cy)


def fit_geometry(projections: np.ndarray, theta: np.ndarray,
                 mask_frac: float = 0.0) -> Geometry:
    """Read the setup's geometry off the centroid track, before correcting it.

    Must be run on the raw stack.  Once the common line has shifted every
    projection onto a reference profile the axial evidence is gone, and with it
    the only sign of either tilt-axis error.

    The four parameters are fitted to where the centroid actually lands, which
    is cheap to predict exactly.  Separating them by harmonic order looks
    tempting -- an in-plane inclination enters at the first, an out-of-plane
    tip at the second -- but the tip contributes to the second harmonic twice
    over, once by turning each image and once by tilting the object, and a
    version that counted only the first read an injected 3 degrees as 1.4.
    """
    p = np.asarray(projections, dtype=np.float64)
    if mask_frac:
        p = np.where(p >= mask_frac * p.max(axis=(0, 1), keepdims=True), p, 0.0)
    total = p.sum(axis=(0, 1))
    total[total == 0] = np.inf
    x = np.arange(p.shape[0], dtype=np.float64)[:, None, None]
    y = np.arange(p.shape[1], dtype=np.float64)[None, :, None]
    com_x = (x * p).sum(axis=(0, 1)) / total
    com_y = (y * p).sum(axis=(0, 1)) / total
    th = np.asarray(theta, dtype=np.float64)

    def residual(par):
        gx, gy = _centroid_track(*par, th)
        return np.concatenate([gx - com_x, gy - com_y])

    guess = [0.0, 0.0, 0.0, 0.0, com_x.mean(), com_y.mean()]
    par = _lsq(residual, guess, x_scale=[1.0, 1.0, 10.0, 10.0, 10.0, 10.0]).x
    gx, gy = _centroid_track(*par, th)
    # Both angles are read from how the centroid's excursion divides between
    # the detector axes.  A specimen on the rotation axis has no excursion, and
    # then the fit explains the data equally well with any angles at all: a
    # centred test with 2 and 3 degrees injected returned -0.2 and -10.9, at a
    # residual of 0.025 px.  The residual cannot catch this -- there is nothing
    # left unexplained -- so the swing is reported and callers check it.
    swing = float(np.hypot(par[2], par[3]))
    return Geometry(swing=swing,
                    x0=float(par[2]), z0=float(par[3]), centre=float(par[4]),
                    tilt_axis_deg=float(par[1]),
                    out_of_plane_deg=float(par[0]),
                    perp_residual=float(np.sqrt(np.mean((com_x - gx) ** 2))),
                    axial_residual=float(np.sqrt(np.mean((com_y - gy) ** 2))))


@dataclass
class AlignmentResult:
    projections: np.ndarray          #: the aligned stack
    perp: np.ndarray                 #: (n_proj,) shift applied along axis 0
    axial: np.ndarray                #: (n_proj,) shift applied along axis 1
    residual_com: float = 0.0        #: rms of the centre-of-mass fit residual
    geometry: Optional[Geometry] = None  #: read off the raw stack
    tilt_axis_applied: float = 0.0   #: in-plane rotation applied, degrees

    def summary(self) -> str:
        return "\n".join([
            "perpendicular shift  : mean %.3f px, max %.3f px"
            % (np.abs(self.perp).mean(), np.abs(self.perp).max()),
            "axial shift          : mean %.3f px, max %.3f px"
            % (np.abs(self.axial).mean(), np.abs(self.axial).max()),
            "centre-of-mass fit   : %.3f px rms residual" % self.residual_com,
            "tilt axis corrected  : %.3f deg" % self.tilt_axis_applied,
        ] + ([self.geometry.summary()] if self.geometry is not None else []))


def _xcorr_1d(reference: np.ndarray, curves: np.ndarray,
              max_shift: Optional[float] = None) -> np.ndarray:
    """Sub-pixel displacement of each column of ``curves`` from ``reference``.

    Zero-padded so the correlation is linear rather than circular; a periodic
    correlation would wrap a large shift around and report a small one.
    ``max_shift`` restricts the peak search to displacements no larger than
    that -- for a crystal, whose profile repeats every lattice plane, the
    search must stay within half a period of an already-coarse alignment or
    it can pick the neighbouring plane.
    """
    n = reference.size
    pad = 2 * n
    fr = np.fft.rfft(reference - reference.mean(), n=pad)
    fc = np.fft.rfft(curves - curves.mean(axis=0, keepdims=True), n=pad, axis=0)
    corr = np.fft.irfft(fc * np.conj(fr)[:, None], n=pad, axis=0)
    if max_shift is not None:
        lag = np.arange(pad)
        lag = np.where(lag > pad / 2, lag - pad, lag)
        corr = np.where((np.abs(lag) <= max_shift)[:, None], corr, -np.inf)
    k = np.argmax(corr, axis=0)
    j = np.arange(curves.shape[1])
    d = k + _parabolic(corr[(k - 1) % pad, j], corr[k, j],
                       corr[(k + 1) % pad, j])
    return np.where(d > pad / 2, d - pad, d)


def _stretch_perp(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale ``image`` by ``factor`` along axis 0 about the centre."""
    if abs(factor - 1.0) < 1e-6:
        return image
    n0, n1 = image.shape
    c = (n0 - 1) / 2.0
    src = c + (np.arange(n0, dtype=np.float64) - c) / factor
    return _map_coordinates(image, np.meshgrid(src, np.arange(n1),
                                               indexing="ij"),
                            order=1, mode="constant", cval=0.0)


def cross_correlation_shifts(projections: np.ndarray, theta: np.ndarray,
                             reference: Optional[int] = None,
                             stretch: bool = True, smooth_px: float = 6.0,
                             taper_px: int = 16) -> Tuple[np.ndarray,
                                                          np.ndarray]:
    """Coarse translation, by correlating each view with its neighbour.

    The centre of mass and the common line are exact in expectation but
    sensitive to everything that is not the object; before either can be
    trusted the bulk drift -- here up to ten pixels -- has to come off.
    Neighbouring views 2-3 degrees apart look almost alike, so correlating
    each with the one next to it that is already aligned finds its
    displacement from the whole image, not one moment of it.

    Starting at ``reference`` (the view nearest 0 degrees by default) and
    walking outward, view k is matched against aligned view k-1 (or k+1),
    and the shift that brings it on is recorded.  Two things keep the match
    honest.  A tilted view is foreshortened across the tilt axis by
    cos(theta), so with ``stretch`` the neighbour is rescaled by
    cos(theta_k)/cos(theta_neighbour) first -- 10% at 70 degrees, enough to
    smear the correlation peak.  And the images are edge-tapered and
    smoothed hard.

    The smoothing is not cosmetic.  A crystal's projection is a lattice of
    columns, and its autocorrelation has a peak at every lattice vector that
    is nearly as tall as the true one -- the particle's envelope is all that
    tells them apart.  Measured on a 5 nm CoPd particle with a 6.2 px
    projected period, correlating lightly smoothed images (1 px) locked onto
    a neighbouring lattice peak in a quarter of the views: 8.6 px rms error
    along the axis, the right period to the pixel.  Smoothing by 6 px
    removes the fringes, leaves the envelope, and brings the chain to 0.5 px
    rms noiseless and 0.36/0.54 px (perp/axial) at dose 30 on a film.

    The errors of a chain add up as a random walk, which is why this is only
    the first stage: what it leaves is small enough for the centre of mass
    and the common line to work on.  The frame is the reference view's own,
    whatever its drift was; that is a global offset the later stages set.

    Returns ``(perp, axial)``: the shifts to apply (as in
    :func:`shift_projections`) to bring each view into the common frame.
    """
    p = np.asarray(projections, dtype=np.float64)
    n0, n1, m = p.shape
    th = np.deg2rad(np.asarray(theta, dtype=np.float64))
    if reference is None:
        reference = int(np.argmin(np.abs(th)))

    # smoothing and an edge taper, applied once to every view
    if smooth_px:
        p = _gaussian_filter(p, (smooth_px, smooth_px, 0))
    w0 = np.ones(n0)
    w1 = np.ones(n1)
    if taper_px:
        r = 0.5 - 0.5 * np.cos(np.pi * np.arange(taper_px) / taper_px)
        w0[:taper_px], w0[-taper_px:] = r, r[::-1]
        w1[:taper_px], w1[-taper_px:] = r, r[::-1]
    win = (w0[:, None] * w1[None, :])[..., None]
    p = p * win

    perp = np.zeros(m)
    axial = np.zeros(m)
    aligned = np.empty_like(p)
    aligned[..., reference] = p[..., reference]
    order = ([(k, k + 1) for k in range(reference - 1, -1, -1)]
             + [(k, k - 1) for k in range(reference + 1, m)])
    for k, nb in order:
        ref_img = aligned[..., nb]
        if stretch:
            ref_img = _stretch_perp(ref_img, np.cos(th[k]) / np.cos(th[nb]))
        _, dy, dx = _ncc_over_shifts(p[..., k:k + 1], ref_img[..., None])
        perp[k], axial[k] = dy[0], dx[0]
        aligned[..., k] = shift_projections(p[..., k:k + 1], dy, dx)[..., 0]
    return perp, axial


def common_line_shifts(projections: np.ndarray,
                       reference: Optional[int] = None,
                       max_shift: Optional[float] = None) -> np.ndarray:
    """Displacement along the tilt axis, from the profile every view shares.

    ``max_shift`` bounds the search (see :func:`_xcorr_1d`); use it once a
    coarse alignment has run, with less than half the lattice period.
    """
    curves = projections.sum(axis=0).astype(np.float64)
    if reference is None:
        reference = projections.shape[2] // 2
    return -_xcorr_1d(curves[:, reference], curves, max_shift)


def centre_of_mass_shifts(projections: np.ndarray, theta: np.ndarray,
                          mask_frac: float = 0.0,
                          fit_sinusoid: bool = True,
                          mask: Optional[np.ndarray] = None,
                          recentre: bool = False
                          ) -> Tuple[np.ndarray, float]:
    """Displacement perpendicular to the tilt axis, from the first moment.

    ``mask_frac`` ignores pixels below that fraction of each projection's
    maximum.  This is the part that actually fights noise: the estimator's
    variance goes as the sum of x^2 over every pixel included, so dropping
    empty background removes most of the leverage.

    ``fit_sinusoid`` puts the measured moments on ``a cos(theta) +
    b sin(theta) + c`` and corrects each projection onto the fit.  That keeps
    the motion an off-axis object genuinely has, instead of flattening it, and
    the fit residual is returned as a diagnostic -- if it is large the model is
    wrong, not the data.

    ``mask`` (bool, same shape) restricts the moment to each view's particle
    footprint, e.g. the one :func:`subtract_background` returns.  On a
    film-backed series this is the larger effect: what the background
    estimate leaves of the film sits mostly far from the centre, where the
    moment's lever arm is longest -- measured at dose 300, 0.18 px of
    perpendicular error from the residue alone, dose-independent.

    ``recentre`` puts the fitted constant on the rotation axis, (n-1)/2,
    instead of leaving it where the data put it.  The sinusoid's cos and sin
    terms are the object's position -- a 3D translation, harmless -- but the
    constant is where the rotation axis sits on the detector, and after a
    drifting acquisition that is the mean drift, not the centre.  Left there
    the reconstruction rotates about the wrong axis.  Measured with +-10 px
    uniform drift and no noise at all: 0.28 px rms perpendicular error, U
    shaped in theta (the constant seen through the cos/sin gauge), gone with
    ``recentre``.  Off by default only so that earlier studies reproduce.
    """
    p = np.asarray(projections, dtype=np.float64)
    if mask is not None:
        p = np.where(mask, p, 0.0)
    if mask_frac:
        p = np.where(p >= mask_frac * p.max(axis=(0, 1), keepdims=True), p, 0.0)
    x = np.arange(p.shape[0], dtype=np.float64)[:, None, None]
    total = p.sum(axis=(0, 1))
    total[total == 0] = np.inf
    com = (x * p).sum(axis=(0, 1)) / total

    if not fit_sinusoid:
        return (p.shape[0] - 1) / 2.0 - com, 0.0
    th = np.deg2rad(np.asarray(theta, dtype=np.float64))
    A = np.column_stack([np.cos(th), np.sin(th), np.ones_like(th)])
    coef, *_ = np.linalg.lstsq(A, com, rcond=None)
    if recentre:
        coef[2] = (p.shape[0] - 1) / 2.0
    fitted = A @ coef
    return fitted - com, float(np.sqrt(np.mean((com - fitted) ** 2)))


def align_projections(projections: np.ndarray, theta: np.ndarray,
                      reference: Optional[int] = None, mask_frac: float = 0.0,
                      fit_sinusoid: bool = True, rounds: int = 2,
                      correct_tilt_axis: bool = True,
                      min_tilt_axis_deg: float = 0.2,
                      max_shift: Optional[float] = None,
                      mask: Optional[np.ndarray] = None,
                      recentre: bool = False) -> AlignmentResult:
    """Common-line then centre-of-mass, repeated a couple of times.

    The two are not quite independent -- shifting along one axis changes the
    other's estimate slightly -- so a second pass is cheap insurance.  Neither
    step looks at the angles except through the sinusoid fit, so nothing here
    can be absorbed into an angle error.
    """
    geom = fit_geometry(projections, theta, mask_frac)
    # Read the inclination before the common line shifts it away, and correct
    # it before the shifts, since the two axes are only separable once the
    # tilt axis is on y.  The floor guards against acting on an angle that is
    # mostly noise: the estimate divides by the centroid's swing, so an object
    # near the rotation axis has almost none and the ratio is meaningless.
    psi = geom.tilt_axis_deg if (correct_tilt_axis and geom.trustworthy) else 0.0
    if abs(psi) < min_tilt_axis_deg:
        psi = 0.0
    # negated: rotate_projections turns by -psi, and the estimate is the
    # inclination present, so undoing it needs the opposite sign.  Measured
    # both ways -- the wrong one adds a second inclination and leaves 7% of
    # the density, the right one restores 99%.
    out = rotate_projections(projections, -psi) if psi else np.array(
        projections, copy=True)
    perp = np.zeros(projections.shape[2])
    axial = np.zeros(projections.shape[2])
    residual = None
    for _ in range(max(1, rounds)):
        d = common_line_shifts(out, reference, max_shift)
        out = shift_projections(out, np.zeros_like(d), d)
        axial += d
        # the footprint moves with the view, so it gets the same rotation and
        # the shifts applied so far (re-thresholded at one half after the
        # sub-pixel interpolation)
        m = None
        if mask is not None:
            m = mask.astype(np.float32)
            if psi:
                m = rotate_projections(m, -psi, order=1)
            m = shift_projections(m, perp, axial) > 0.5
        d, res = centre_of_mass_shifts(out, theta, mask_frac, fit_sinusoid, m,
                                       recentre)
        # the diagnostic is the fit on the data as it arrived; measuring it
        # again after correcting to the fit only reports how well the last
        # correction was applied, which is always well
        residual = res if residual is None else residual
        out = shift_projections(out, d, np.zeros_like(d))
        perp += d
    return AlignmentResult(projections=out, perp=perp, axial=axial,
                           residual_com=float(residual or 0.0),
                           geometry=geom, tilt_axis_applied=psi)


def rotate_projections(projections: np.ndarray, psi_deg: float,
                       order: int = 3) -> np.ndarray:
    """Rotate every projection in plane, to put the tilt axis back on y.

    An in-plane inclination needs no general 3D projector, and it is worth
    being precise about why.  The single-axis operator is cheap because a
    rotation about y leaves each y slice independent: one ``(n, n**2)`` matrix
    serves every column.  Incline the axis and the rotation mixes y with x and
    z, the slices stop being independent, and the operator becomes
    ``(n**2, n**3)`` -- 512 times larger at n = 512, which is the end of that
    approach.

    But an in-plane inclination is the ideal geometry composed with one fixed
    rotation about the beam, the *same* for every projection.  On the detector
    that is a 2D rotation of each recorded image, and undoing it restores the
    ideal geometry exactly; what is reconstructed afterwards differs from the
    truth by a rigid rotation, which is not error.

    Out-of-plane inclination -- the axis tipped toward or away from the beam --
    is a different problem.  It changes the effective tilt angles rather than
    the images, so no image rotation fixes it, and this function does not
    claim to.
    """
    if not psi_deg:
        return np.array(projections, copy=True)
    # axes (0, 1) are the detector; axis 2 indexes the tilt series
    return _ndrotate(projections, -psi_deg, axes=(0, 1), reshape=False,
                     order=order, mode="constant", cval=0.0).astype(
                         projections.dtype, copy=False)


def out_of_plane_signature(projections: np.ndarray, theta: np.ndarray,
                           reference: Optional[int] = None) -> np.ndarray:
    """Per-view evidence that the tilt axis is tipped toward the beam.

    A tipped axis is the error an in-plane rotation cannot reach, because it
    changes the effective orientations rather than the recorded images.  Turn
    the stage by theta about an axis tipped by alpha and the orientation
    reached differs from a clean theta about y by ``2 alpha sin(theta / 2)`` --
    zero at zero tilt, largest at the extremes.

    That shape is what makes it detectable.  Summing a view across x gives the
    object's profile along the tilt axis, the same curve for every view only
    while the axis really is y; tip it and the curves disagree, by more at
    higher tilt and not at all in the middle.  Measured on a 2.5 nm particle,
    the edge views' mismatch rose 16-fold at 1 degree and 204-fold at 3, while
    the central view stayed exactly matched.

    Returns one mismatch per projection -- ``1 - correlation`` against the
    reference profile after its best shift.  A U-shaped result, low in the
    middle and rising at both ends, is the signature; a flat one is not.
    Uniform elevation means something else, noise most likely.

    Detection is worth having on its own.  Only the beam-axis part of the
    leftover can be undone by rotating each image, roughly 40% of it at high
    tilt; the rest needs the general 3D operator this module avoids.  But
    neither released reconstructor notices this error at all, and a
    reconstruction that is quietly wrong is worse than one known to be.
    """
    curves = projections.sum(axis=0).astype(np.float64)
    n_proj = curves.shape[1]
    if reference is None:
        reference = n_proj // 2
    ref = curves[:, reference]
    shifts = common_line_shifts(projections, reference)
    grid = np.arange(ref.size, dtype=np.float64)
    out = np.empty(n_proj)
    for j in range(n_proj):
        aligned = np.interp(grid + shifts[j], grid, curves[:, j])
        out[j] = 1.0 - float(np.corrcoef(ref, aligned)[0, 1])
    return out
