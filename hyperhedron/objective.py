"""
Newton objective function and Jacobian for hyperbolic polyhedron construction.

The system being solved is:

    f(V) = 0,   V ∈ R^{4N}  (face vectors flattened row-major)

where f has 4N equations:

  Equations 0–2:   Gauge-fixing (pin 3 components of a reference face).
  Equations 3–5:   Incidence gauge (3 orthogonality conditions at a reference vertex).
  Equations 6–4N:  Angle constraints: mink(v_i, v_j) = cos(π - θ_ij) for each
                   adjacent pair (i ≤ j), including self-pairs (i == j) which
                   enforce the unit-spacelike normalization mink(v_i, v_i) = 1.

The system is square (4N × 4N) for any simple polyhedron satisfying Euler's
formula: E = 3N - 6 off-diagonal adjacent pairs plus N self-pairs gives
N + E + 6 = N + (3N-6) + 6 = 4N equations.

Gauge-fixing explanation
------------------------
The 6 gauge conditions remove the 6 degrees of freedom of the isometry group
Isom(H^3) ≅ SO(3,1).  The first three equations are identically zero at every
point (they measure how much the gauge face moves, which is nothing, since we
pin it).  Their purpose is structural: the corresponding rows of the Jacobian
have 1s in the positions of those components, which zeroes out the Newton
update in those directions.  The last three gauge equations enforce that two
reference vertices stay fixed under the solve.

This is a faithful port of Roeder's f.m / newton.m, converted to 0-based
indexing and with NaN-safe acos added in angles.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .linalg import mink, solve_for_vertex
from .polyhedron import GeomPolyhedron

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gauge data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaugeInfo:
    """
    Fixed reference data for the Newton gauge-fixing conditions.

    face_i, face_j, face_k: 0-indexed face indices of the reference vertex.
    vertex1: timelike unit vector at the intersection of (face_i, face_j, face_k).
    vertex2: timelike unit vector at a second vertex sharing edge (face_i, face_j).

    These are computed *once* from the initial face vectors and held fixed
    throughout the Newton iteration.  Changing gauge mid-solve would invalidate
    the Jacobian structure.
    """

    face_i: int
    face_j: int
    face_k: int
    vertex1: np.ndarray   # shape (4,)
    vertex2: np.ndarray   # shape (4,)

    class Config:
        arbitrary_types_allowed = True


def make_gauge_info(
    face_vectors: np.ndarray,
    adjacency: np.ndarray,
    vertices: np.ndarray,
    vertex_idx: int,
) -> GaugeInfo:
    """
    Compute GaugeInfo anchored at the vertex given by vertex_idx.

    vertex_idx: index into the (M, 3) vertices array.

    The second vertex (vertex2) is chosen as a neighbor of vertex_idx that
    also involves both face_i and face_j (i.e., shares the edge (face_i, face_j)).

    Raises ValueError if no suitable second vertex can be found.
    """
    fi, fj, fk = (
        int(vertices[vertex_idx, 0]),
        int(vertices[vertex_idx, 1]),
        int(vertices[vertex_idx, 2]),
    )
    vertex1 = solve_for_vertex(face_vectors, fi, fj, fk)

    # Find a second vertex sharing the edge (face_i, face_j)
    num_vertices = vertices.shape[0]
    vertex2 = None
    for n in range(num_vertices):
        if n == vertex_idx:
            continue
        fi2, fj2, fk2 = int(vertices[n, 0]), int(vertices[n, 1]), int(vertices[n, 2])
        shared = {fi2, fj2, fk2} & {fi, fj}
        if len(shared) == 2:  # shares both face_i and face_j
            vertex2 = solve_for_vertex(face_vectors, fi2, fj2, fk2)
            break

    if vertex2 is None:
        raise ValueError(
            f"Cannot find a second vertex for gauge info: no vertex other than "
            f"{vertex_idx} shares edge ({fi}, {fj})."
        )

    return GaugeInfo(face_i=fi, face_j=fj, face_k=fk, vertex1=vertex1, vertex2=vertex2)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_residual(
    face_vectors: np.ndarray,
    target_angles: np.ndarray,
    gauge: GaugeInfo,
) -> np.ndarray:
    """
    Compute the (4N,) residual vector f(V).

    face_vectors: (N, 4) current iterate.
    target_angles: (N, N) target dihedral angles in degrees.
        Diagonal must be 180, non-adjacent -10, adjacent entries in (0, 180).
    gauge: fixed GaugeInfo computed from the *initial* face vectors.

    The flat vector ordering is row-major: index 4*i + k corresponds to
    face i, component k.  This matches MATLAB's reshape(v_iter', 4*N, 1).
    """
    N = face_vectors.shape[0]
    f = np.zeros(4 * N)

    fi, fj, fk = gauge.face_i, gauge.face_j, gauge.face_k
    v1, v2 = gauge.vertex1, gauge.vertex2

    # --- Gauge rows 0–2: pin components 0, 2, 3 of face_i ---
    # These are identically 0 at every point in the domain (by construction).
    # Their role is structural: the Jacobian rows 0–2 freeze those components.
    f[0] = 0.0
    f[1] = 0.0
    f[2] = 0.0

    # --- Gauge rows 3–5: incidence orthogonality ---
    # face_j must be Minkowski-orthogonal to both reference vertices;
    # face_k must be Minkowski-orthogonal to the first reference vertex.
    f[3] = mink(face_vectors[fj], v1)
    f[4] = mink(face_vectors[fj], v2)
    f[5] = mink(face_vectors[fk], v1)

    # --- Angle equations (rows 6 … 4N-1) ---
    count = 6
    for i in range(N):
        for j in range(i, N):
            angle_ij = target_angles[i, j]
            if angle_ij > -1.0:
                target_ip = np.cos(np.pi - np.pi / 180.0 * angle_ij)
                actual_ip = mink(face_vectors[i], face_vectors[j])
                f[count] = actual_ip - target_ip
                count += 1

    if count != 4 * N:
        raise ValueError(
            f"System is not square: produced {count} equations for {4 * N} unknowns. "
            f"Check that the polyhedron is simple (Euler: V - E + F = 2) and "
            f"that the adjacency diagonal is all 1s."
        )

    return f


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def compute_jacobian(
    face_vectors: np.ndarray,
    target_angles: np.ndarray,
    gauge: GaugeInfo,
) -> np.ndarray:
    """
    Compute the (4N, 4N) Jacobian matrix J = df/dV analytically.

    This is a faithful port of Roeder's newton.m Jacobian, converted to
    0-based indexing.  The gradient of mink(a, b) with respect to a is:

        d/da mink(a, b) = (-b[0], b[1], b[2], b[3])

    because mink(a, b) = -a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3].

    Test this Jacobian against finite differences via tests/test_objective.py
    before trusting it for a new polyhedron type.
    """
    N = face_vectors.shape[0]
    V = face_vectors   # shape (N, 4)
    J = np.zeros((4 * N, 4 * N))

    fi, fj, fk = gauge.face_i, gauge.face_j, gauge.face_k
    v1, v2 = gauge.vertex1, gauge.vertex2

    # --- Gauge rows 0–2: d/dV[face_i] of the three pinned components ---
    # df_0 / d(face_i[0]) = 1,  df_1 / d(face_i[2]) = 1,  df_2 / d(face_i[3]) = 1
    J[0, 4 * fi + 0] = 1.0
    J[1, 4 * fi + 2] = 1.0
    J[2, 4 * fi + 3] = 1.0

    # --- Gauge row 3: d/d(face_j) [mink(face_j, v1)] = (-v1[0], v1[1], v1[2], v1[3]) ---
    J[3, 4 * fj + 0] = -v1[0]
    J[3, 4 * fj + 1] =  v1[1]
    J[3, 4 * fj + 2] =  v1[2]
    J[3, 4 * fj + 3] =  v1[3]

    # --- Gauge row 4: d/d(face_j) [mink(face_j, v2)] ---
    J[4, 4 * fj + 0] = -v2[0]
    J[4, 4 * fj + 1] =  v2[1]
    J[4, 4 * fj + 2] =  v2[2]
    J[4, 4 * fj + 3] =  v2[3]

    # --- Gauge row 5: d/d(face_k) [mink(face_k, v1)] ---
    J[5, 4 * fk + 0] = -v1[0]
    J[5, 4 * fk + 1] =  v1[1]
    J[5, 4 * fk + 2] =  v1[2]
    J[5, 4 * fk + 3] =  v1[3]

    # --- Angle rows ---
    count = 6
    for i in range(N):
        for j in range(i, N):
            angle_ij = target_angles[i, j]
            if angle_ij > -1.0:
                if i == j:
                    # d/dv_i [mink(v_i, v_i)] = 2 * (-v_i[0], v_i[1], v_i[2], v_i[3])
                    J[count, 4 * i + 0] = -2.0 * V[i, 0]
                    J[count, 4 * i + 1] =  2.0 * V[i, 1]
                    J[count, 4 * i + 2] =  2.0 * V[i, 2]
                    J[count, 4 * i + 3] =  2.0 * V[i, 3]
                else:
                    # d/dv_i [mink(v_i, v_j)] = (-v_j[0], v_j[1], v_j[2], v_j[3])
                    J[count, 4 * i + 0] = -V[j, 0]
                    J[count, 4 * i + 1] =  V[j, 1]
                    J[count, 4 * i + 2] =  V[j, 2]
                    J[count, 4 * i + 3] =  V[j, 3]
                    # d/dv_j [mink(v_i, v_j)] = (-v_i[0], v_i[1], v_i[2], v_i[3])
                    J[count, 4 * j + 0] = -V[i, 0]
                    J[count, 4 * j + 1] =  V[i, 1]
                    J[count, 4 * j + 2] =  V[i, 2]
                    J[count, 4 * j + 3] =  V[i, 3]
                count += 1

    return J
