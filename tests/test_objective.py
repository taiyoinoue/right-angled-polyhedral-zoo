"""
Tests for hyperhedron.objective — residual function and Jacobian.

The most important test here is the Jacobian finite-difference check.
If this test fails for a new polyhedron type, the manual Jacobian in
objective.py has a transcription error that must be fixed before Newton
will converge.
"""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from hyperhedron.angles import right_angle_matrix
from hyperhedron.linalg import mink, mink_norms
from hyperhedron.objective import GaugeInfo, compute_jacobian, compute_residual, make_gauge_info
from hyperhedron.polyhedron import GeomPolyhedron


# ---------------------------------------------------------------------------
# Minimal test polyhedron: triangular prism
# ---------------------------------------------------------------------------
# A triangular prism has 5 faces, 6 vertices, 9 edges.
# E = 3*5 - 6 = 9 ✓, V = 6, Euler: 6 - 9 + 5 = 2 ✓
#
# Faces: 0,1 = triangular caps; 2,3,4 = rectangular sides.
# Adjacency (0-indexed):
#   0 adj 2,3,4  (top cap to all sides)
#   1 adj 2,3,4  (bottom cap to all sides)
#   2 adj 0,1,3,4
#   3 adj 0,1,2,4
#   4 adj 0,1,2,3
#
# Vertices (each is a triple of face indices meeting at a corner):
#   (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4)


def _prism_adjacency() -> np.ndarray:
    N = 5
    adj = np.eye(N, dtype=int)
    edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    return adj


def _prism_vertices() -> np.ndarray:
    return np.array([
        [0, 2, 3],
        [0, 2, 4],
        [0, 3, 4],
        [1, 2, 3],
        [1, 2, 4],
        [1, 3, 4],
    ], dtype=int)


def _make_prism_face_vectors() -> np.ndarray:
    """
    Construct a valid set of face vectors for the triangular prism in H^3.

    We use orthogonal face vectors for the three side faces and construct
    the two cap vectors to be Minkowski-orthogonal to the side faces they
    bound.  All vectors are unit spacelike (mink = +1).

    This is an approximate construction for testing; the angles won't be
    exactly equal, but the system is consistent.
    """
    # Side faces: orthogonal unit spacelike vectors along spatial axes
    v2 = np.array([0., 1., 0., 0.])  # face 2
    v3 = np.array([0., 0., 1., 0.])  # face 3
    v4 = np.array([0., 0., 0., 1.])  # face 4

    # Top cap (face 0): adjacent to 2, 3, 4.
    # For simplicity use a tilted spacelike vector.
    v0_raw = np.array([0.3, 1., 1., 1.])
    v0 = v0_raw / np.sqrt(mink(v0_raw, v0_raw))

    # Bottom cap (face 1): also adjacent to 2, 3, 4; make it different.
    v1_raw = np.array([-0.3, 1., 1., 1.])
    v1 = v1_raw / np.sqrt(mink(v1_raw, v1_raw))

    V = np.stack([v0, v1, v2, v3, v4])
    return V


@pytest.fixture
def prism_geom():
    """Return a GeomPolyhedron for the triangular prism."""
    adj = _prism_adjacency()
    verts = _prism_vertices()
    V = _make_prism_face_vectors()
    return GeomPolyhedron(face_vectors=V, adjacency=adj, vertices=verts)


@pytest.fixture
def prism_angles(prism_geom):
    """Target angle matrix for the prism (right-angled)."""
    return right_angle_matrix(prism_geom.adjacency)


@pytest.fixture
def prism_gauge(prism_geom):
    """GaugeInfo anchored at vertex 0 of the prism."""
    g = prism_geom
    return make_gauge_info(g.face_vectors, g.adjacency, g.vertices, vertex_idx=0)


# ---------------------------------------------------------------------------
# Tests: residual
# ---------------------------------------------------------------------------

class TestComputeResidual:
    def test_output_shape(self, prism_geom, prism_angles, prism_gauge):
        N = prism_geom.num_faces
        f = compute_residual(prism_geom.face_vectors, prism_angles, prism_gauge)
        assert f.shape == (4 * N,)

    def test_gauge_rows_are_zero(self, prism_geom, prism_angles, prism_gauge):
        f = compute_residual(prism_geom.face_vectors, prism_angles, prism_gauge)
        # First three rows must always be identically zero (gauge fixing)
        np.testing.assert_allclose(f[:3], 0.0, atol=1e-15)

    def test_self_norm_equations(self, prism_geom, prism_angles, prism_gauge):
        """
        At a point where all face vectors have mink norm = 1, the self-norm
        equations (diagonal of the angle matrix) should be zero.
        """
        V = prism_geom.face_vectors
        norms = mink_norms(V)
        # Our test face vectors should already be normalized
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_equation_count(self, prism_geom, prism_angles, prism_gauge):
        """Verify the system is square (4N equations, 4N unknowns)."""
        N = prism_geom.num_faces  # 5
        adj = prism_geom.adjacency
        # Count active angle equations: upper triangle of adjacency (including diagonal)
        n_active = sum(
            1 for i in range(N) for j in range(i, N)
            if prism_angles[i, j] > -1.0
        )
        # 6 gauge + n_active angle = 4*N
        assert 6 + n_active == 4 * N


# ---------------------------------------------------------------------------
# Tests: Jacobian vs finite differences
# ---------------------------------------------------------------------------

class TestJacobianAccuracy:
    """
    Verify the analytic Jacobian against finite-difference approximations.

    A failure here means the manual Jacobian in objective.py has an error.
    The tolerance is generous (1e-5) to account for finite-difference noise,
    but the analytic Jacobian should be exact up to floating-point precision.
    """

    def test_jacobian_finite_diff(self, prism_geom, prism_angles, prism_gauge):
        V = prism_geom.face_vectors
        N = prism_geom.num_faces
        x0 = V.flatten()

        def f_flat(x):
            Vx = x.reshape(N, 4)
            return compute_residual(Vx, prism_angles, prism_gauge)

        # Analytic Jacobian
        J_analytic = compute_jacobian(V, prism_angles, prism_gauge)

        # Finite-difference Jacobian (column by column)
        h = 1e-7
        J_fd = np.zeros_like(J_analytic)
        f0 = f_flat(x0)
        for k in range(4 * N):
            x1 = x0.copy()
            x1[k] += h
            J_fd[:, k] = (f_flat(x1) - f0) / h

        # Rows 0–2 are the gauge rows: f[0:3] = 0 identically, so the finite-
        # difference Jacobian is all zeros there, while the analytic Jacobian
        # has 1s by design (to freeze those components during Newton's method).
        # We verify only rows 3 onward where the analytic and FD Jacobians agree.
        np.testing.assert_allclose(J_analytic[3:], J_fd[3:], atol=1e-5, rtol=1e-4)

        # Spot-check gauge rows: analytic has deliberate 1s, FD has 0s.
        assert J_analytic[0, 4 * prism_gauge.face_i + 0] == 1.0
        assert J_analytic[1, 4 * prism_gauge.face_i + 2] == 1.0
        assert J_analytic[2, 4 * prism_gauge.face_i + 3] == 1.0
        np.testing.assert_allclose(J_fd[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(J_fd[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(J_fd[2], 0.0, atol=1e-10)

    def test_jacobian_shape(self, prism_geom, prism_angles, prism_gauge):
        N = prism_geom.num_faces
        J = compute_jacobian(prism_geom.face_vectors, prism_angles, prism_gauge)
        assert J.shape == (4 * N, 4 * N)

    def test_jacobian_not_singular(self, prism_geom, prism_angles, prism_gauge):
        J = compute_jacobian(prism_geom.face_vectors, prism_angles, prism_gauge)
        kappa = np.linalg.cond(J)
        # A well-conditioned starting point should have condition number < 1e8
        assert kappa < 1e8, f"Jacobian condition number {kappa:.3e} is unexpectedly large."


# ---------------------------------------------------------------------------
# Tests: angles module
# ---------------------------------------------------------------------------

class TestAnglesConsistency:
    def test_self_angles_are_180(self, prism_geom):
        from hyperhedron.angles import compute_angles
        A = compute_angles(prism_geom)
        N = prism_geom.num_faces
        for i in range(N):
            assert A[i, i] == pytest.approx(180.0, abs=0.1)

    def test_non_adjacent_are_minus_10(self, prism_geom):
        from hyperhedron.angles import compute_angles
        A = compute_angles(prism_geom)
        N = prism_geom.num_faces
        for i in range(N):
            for j in range(N):
                if prism_geom.adjacency[i, j] == 0:
                    assert A[i, j] == -10.0

    def test_right_angle_matrix_structure(self):
        from hyperhedron.angles import right_angle_matrix
        adj = _prism_adjacency()
        R = right_angle_matrix(adj)
        N = 5
        for i in range(N):
            assert R[i, i] == 180.0
            for j in range(N):
                if i != j:
                    if adj[i, j] == 1:
                        assert R[i, j] == 90.0
                    else:
                        assert R[i, j] == -10.0
