"""RESIRE: real-space iterative reconstruction for single-axis tomography.

A Python implementation of the algorithm in Pham, Yuan, Rana et al.,
"Accurate real space iterative reconstruction (RESIRE) algorithm for
tomography", Sci. Rep. 13, 5624 (2023).  The reference release is MATLAB with
CUDA kernels; this one is CPU NumPy/SciPy and covers the single-tilt-axis case
that the reference's own CPU path covers.
"""

from .align import (AlignmentResult, Geometry, align_projections,
                    centre_of_mass_shifts, common_line_shifts,
                    cross_correlation_shifts,
                    fit_geometry, out_of_plane_signature,
                    rotate_projections)
from .background import (BackgroundResult, laplace_fill, particle_mask,
                         subtract_background)
from .project import SingleAxisProjector
from .score import Pose, fit_pose, sample_atoms
from .refine import (RefinementResult, refine_shifts, refine_theta,
                     score_angles, shift_projections)
from .reconstruct import (ReconstructionConfig, ReconstructionResult,
                          reconstruct)

__all__ = ["SingleAxisProjector", "ReconstructionConfig",
           "ReconstructionResult", "reconstruct",
           "RefinementResult", "refine_theta", "refine_shifts",
           "score_angles", "shift_projections",
           "AlignmentResult", "align_projections",
           "centre_of_mass_shifts", "common_line_shifts",
           "cross_correlation_shifts",
           "Geometry", "fit_geometry", "rotate_projections",
           "out_of_plane_signature",
           "Pose", "fit_pose", "sample_atoms",
           "BackgroundResult", "subtract_background", "particle_mask",
           "laplace_fill"]
__version__ = "0.1.0"
