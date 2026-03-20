"""
Dihedral angle computation and target angle matrix construction.

The dihedral angle between adjacent faces i and j satisfies:

    mink(v_i, v_j) = cos(pi - theta_ij)

where theta_ij is in radians.  Equivalently:

    theta_ij (degrees) = 180 - (180/pi) * arccos(mink(v_i, v_j))

Non-adjacent pairs are marked with the sentinel value -10.
The diagonal (i == i) gives 180, corresponding to mink(v_i, v_i) = 1.
"""

import logging

import numpy as np

from .linalg import mink
from .polyhedron import GeomPolyhedron

logger = logging.getLogger(__name__)


def compute_angles(geom: GeomPolyhedron) -> np.ndarray:
    """
    Compute the N x N matrix of dihedral angles in degrees.

    Non-adjacent pairs → -10.
    Self-pairs (diagonal) → 180.

    Uses safe arccos: inner products are clamped to [-1, 1] before the
    arccos call.  A debug warning is logged when clamping is needed, as
    this indicates floating-point drift that the re-normalization step
    in the Newton solver should be preventing.
    """
    V = geom.face_vectors
    A = geom.adjacency
    N = geom.num_faces

    result = np.full((N, N), -10.0)

    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                ip = mink(V[i], V[j])
                clamped = np.clip(ip, -1.0, 1.0)
                if abs(ip - clamped) > 1e-6:
                    logger.debug(
                        "Clamping inner product at (%d,%d): %.8f → %.8f", i, j, ip, clamped
                    )
                result[i, j] = 180.0 - (180.0 / np.pi) * np.arccos(clamped)

    return result


def right_angle_matrix(adjacency: np.ndarray) -> np.ndarray:
    """
    Build the target angles matrix for a right-angled polyhedron.

    All adjacent pairs get 90°; diagonal gets 180°; non-adjacent get -10.
    """
    N = adjacency.shape[0]
    result = np.full((N, N), -10.0)
    np.fill_diagonal(result, 180.0)
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] == 1:
                result[i, j] = 90.0
                result[j, i] = 90.0
    return result


def canonical_angle_matrix(adjacency: np.ndarray, angle_deg: float = 72.0) -> np.ndarray:
    """
    Build a uniform angle matrix (all adjacent pairs get the same angle).

    Used for the canonical starting position (72°) of the primitive element.
    """
    N = adjacency.shape[0]
    result = np.full((N, N), -10.0)
    np.fill_diagonal(result, 180.0)
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] == 1:
                result[i, j] = angle_deg
                result[j, i] = angle_deg
    return result


def interpolate_angles(
    angles_start: np.ndarray,
    angles_end: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Linearly interpolate between two angle matrices at homotopy parameter t ∈ [0, 1].

    Rules (matching Roeder's intermediate_angles.m):
    - Diagonal entries (180°) stay at 180°.
    - Non-adjacent entries (-10) stay at -10.
    - Adjacent entries are linearly interpolated: (1-t)*a1 + t*a2.

    If an entry is adjacent at the end but not the start (new edge from a
    Whitehead move), it is treated as -10 at the start.  This case is
    handled by the Whitehead move controller rather than here.
    """
    N = angles_start.shape[0]
    result = np.full((N, N), -10.0)

    for i in range(N):
        for j in range(N):
            a1 = angles_start[i, j]
            a2 = angles_end[i, j]

            if abs(a1 - 180.0) < 0.1:
                # Self-pair: keep at 180
                result[i, j] = 180.0
            elif a1 >= 0.0 and a1 < 178.0:
                # Valid adjacent angle (including newly-adjacent edges with a1≈0): interpolate
                result[i, j] = (1.0 - t) * a1 + t * a2
            # else: non-adjacent (-10) → stays at -10

    return result


def target_inner_product(angle_deg: float) -> float:
    """
    Convert a target dihedral angle in degrees to the required Minkowski inner product.

        mink(v_i, v_j) = cos(pi - pi * angle_deg / 180)

    For a right angle (90°): cos(pi/2) = 0.
    For a 72° angle: cos(108° in rad) ≈ -0.309.
    """
    return float(np.cos(np.pi - np.pi * angle_deg / 180.0))
