"""
Core linear algebra for hyperbolic 3-space.

Convention: Minkowski inner product with signature (-,+,+,+).
    <a, b>_M = -a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

Face vectors are *spacelike*: <v,v>_M = +1.
Vertex vectors are *timelike*, future-pointing: <v,v>_M = -1, v[0] > 0.

This matches Roeder's MATLAB convention (his inner() function returns the same value).
"""

import logging

import numpy as np
from scipy.linalg import null_space as _scipy_null_space

from .exceptions import IdealVertexError, NumericalError

logger = logging.getLogger(__name__)

# Tolerances
_SPACELIKE_TOL = 1e-8   # mink(v,v) must exceed this to normalize as spacelike
_TIMELIKE_TOL = 1e-8    # -mink(v,v) must exceed this to normalize as timelike
_KLEIN_BALL_TOL = 1e-10 # Klein point norm must be < 1 - this tolerance

# The Minkowski metric diagonal: (-1, +1, +1, +1)
_ETA = np.array([-1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Inner products
# ---------------------------------------------------------------------------

def mink(a: np.ndarray, b: np.ndarray) -> float:
    """
    Minkowski inner product <a, b>_M = -a0*b0 + a1*b1 + a2*b2 + a3*b3.

    Both a and b should be 1-D arrays of length 4.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(-a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])


def mink_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Batched Minkowski inner products.

    A: (M, 4), B: (N, 4)  ->  result[i, j] = <A[i], B[j]>_M,  shape (M, N).
    """
    # <A_i, B_j> = sum_k eta_k * A_i[k] * B_j[k]  =  (A * eta) @ B^T
    return (A * _ETA) @ B.T


def mink_norms(V: np.ndarray) -> np.ndarray:
    """
    Self Minkowski inner products of each row of V.

    V: (N, 4)  ->  result[i] = <V[i], V[i]>_M,  shape (N,).
    """
    return np.sum(V * (V * _ETA), axis=1)


# ---------------------------------------------------------------------------
# Null space
# ---------------------------------------------------------------------------

def null_space(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """
    SVD-based null space of A.

    Returns a matrix whose columns span the null space of A.
    Uses scipy.linalg.null_space with explicit rcond tolerance.
    """
    return _scipy_null_space(A, rcond=rcond)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_spacelike(v: np.ndarray) -> np.ndarray:
    """
    Normalize v so that mink(v, v) = +1 (unit spacelike vector).

    Raises NumericalError if v is not sufficiently spacelike.
    """
    n = mink(v, v)
    if n < _SPACELIKE_TOL:
        raise NumericalError(
            f"Cannot normalize as spacelike: mink(v,v) = {n:.6e}. "
            "Vector may be timelike, null, or have drifted off the constraint surface."
        )
    return v / np.sqrt(n)


def normalize_timelike(v: np.ndarray, face_triple=None) -> np.ndarray:
    """
    Normalize v so that mink(v, v) = -1 and v[0] > 0 (future-pointing timelike).

    Raises IdealVertexError if v is not sufficiently timelike.
    """
    n = mink(v, v)
    if n > -_TIMELIKE_TOL:
        raise IdealVertexError(face_triple=face_triple, mink_norm=n)
    v = v / np.sqrt(-n)
    if v[0] < 0:
        v = -v
    return v


def renormalize_spacelike_batch(V: np.ndarray) -> np.ndarray:
    """
    Re-project each row of V onto the spacelike unit hyperboloid (mink norm = +1).

    Used after Newton steps to correct floating-point drift.  Vectors that have
    drifted to non-positive Minkowski norm are clamped — a warning is logged,
    as this indicates significant numerical trouble.

    V: (N, 4)  ->  (N, 4) normalized
    """
    norms = mink_norms(V)  # shape (N,)
    bad = norms <= 0
    if np.any(bad):
        logger.warning(
            "Face vectors at indices %s have non-positive Minkowski norm (%.3e). "
            "Newton convergence may be compromised.",
            np.where(bad)[0].tolist(),
            float(norms[bad].min()),
        )
    safe_norms = np.maximum(norms, 1e-30)
    return V / np.sqrt(safe_norms)[:, np.newaxis]


# ---------------------------------------------------------------------------
# Vertex computation
# ---------------------------------------------------------------------------

def solve_for_vertex(
    face_vectors: np.ndarray,
    i: int,
    j: int,
    k: int,
    rcond: float = 1e-10,
) -> np.ndarray:
    """
    Compute the vertex at the intersection of faces i, j, k.

    A vertex in H^3 is Minkowski-orthogonal to each of its three incident face
    normals.  Concretely, if v is the vertex and n_i is face i's normal vector,
    then <v, n_i>_M = 0.  Written out: -v[0]*n_i[0] + v[1]*n_i[1] + ... = 0,
    which is the null space of the matrix A where A[row] = n_i with the first
    column sign-flipped.

    Returns a future-pointing timelike unit vector (mink norm = -1).

    Raises IdealVertexError if the vertex is at infinity (all three faces share
    a common ideal point).
    """
    A = face_vectors[[i, j, k], :].copy()
    A[:, 0] *= -1  # sign flip: row dot product becomes Minkowski inner product

    ns = null_space(A, rcond=rcond)
    if ns.shape[1] == 0:
        raise NumericalError(
            f"No null vector found for faces ({i}, {j}, {k}). "
            "The three face normals may not span a codimension-1 subspace."
        )
    v = ns[:, 0]
    return normalize_timelike(v, face_triple=(i, j, k))


# ---------------------------------------------------------------------------
# Coordinate projections
# ---------------------------------------------------------------------------

def to_klein(v: np.ndarray) -> np.ndarray:
    """
    Project a hyperboloid point to Klein (projective) model coordinates.

        K = v[1:] / v[0]

    The Klein model represents H^3 as the open unit ball {K : |K| < 1}.
    Geomview's OFF format uses these coordinates.

    Raises NumericalError if v is ideal (v[0] ≈ 0) or outside the ball.
    """
    if abs(v[0]) < 1e-12:
        raise NumericalError(
            f"Vertex is ideal (v[0] = {v[0]:.3e} ≈ 0); cannot project to Klein model."
        )
    K = v[1:] / v[0]
    norm_sq = float(np.dot(K, K))
    if norm_sq >= 1.0 - _KLEIN_BALL_TOL:
        raise NumericalError(
            f"Klein point is outside or on the boundary of the unit ball: |K|² = {norm_sq:.8f}. "
            "The vertex may be ideal or the polyhedron non-compact."
        )
    return K


def to_poincare(v: np.ndarray) -> np.ndarray:
    """
    Project a hyperboloid point to Poincaré ball model coordinates.

        P = K / (1 + sqrt(1 - |K|²))

    where K = to_klein(v).  Both the Klein and Poincaré models represent H^3
    as the open unit ball, but with different metrics.  Geodesics are straight
    lines in Klein and circular arcs in Poincaré.
    """
    K = to_klein(v)
    norm_sq = float(np.dot(K, K))
    denom = 1.0 + np.sqrt(max(0.0, 1.0 - norm_sq))
    return K / denom
