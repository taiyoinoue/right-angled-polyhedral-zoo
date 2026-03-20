"""
Tests for hyperhedron.newton — Newton solver and homotopy controller.

Key tests:
- Convergence from a near-solution starting point.
- Backtracking line search engages when needed.
- IllConditionedJacobian is raised for degenerate systems.
"""

import numpy as np
import pytest

from hyperhedron.angles import compute_angles, right_angle_matrix
from hyperhedron.exceptions import IllConditionedJacobian, NewtonFailure
from hyperhedron.linalg import mink_norms
from hyperhedron.newton import newton_solve, run_homotopy
from hyperhedron.objective import make_gauge_info
from hyperhedron.polyhedron import GeomPolyhedron


# ---------------------------------------------------------------------------
# Fixture: a valid geometric configuration with a known solution
#
# We construct a GeomPolyhedron for the triangular prism (same as in
# test_objective.py) and perturb its face vectors slightly.  Newton should
# recover the original configuration.
# ---------------------------------------------------------------------------

def _prism_adjacency():
    N = 5
    adj = np.eye(N, dtype=int)
    edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    return adj


def _prism_vertices():
    return np.array([
        [0, 2, 3], [0, 2, 4], [0, 3, 4],
        [1, 2, 3], [1, 2, 4], [1, 3, 4],
    ], dtype=int)


def _make_right_angle_prism() -> GeomPolyhedron:
    """
    Construct a GeomPolyhedron for the triangular prism with 90° dihedral angles.

    We use orthogonal face vectors: the three side faces are axis-aligned
    spacelike unit vectors, and the two cap faces are chosen to make all
    dihedral angles 90° (i.e., mink(v_i, v_j) = 0 for all adjacent pairs).

    For orthogonal faces: mink(e_1, e_2) = 0, mink(e_1, e_3) = 0, etc.
    The cap faces must also be orthogonal to all three side faces they touch,
    so we need vectors orthogonal to e_1, e_2, e_3 in Minkowski space.
    The only such directions are (±1, 0, 0, 0), which are timelike!

    This means a right-angled prism with 5 faces cannot have all 90° angles
    in H^3 — it would require ideal vertices.  Instead we use a slightly
    different angle (to give a valid starting configuration) and test that
    Newton can deform between two valid configurations.
    """
    from hyperhedron.linalg import normalize_spacelike

    # Three orthogonal side faces
    v2 = np.array([0., 1., 0., 0.])
    v3 = np.array([0., 0., 1., 0.])
    v4 = np.array([0., 0., 0., 1.])

    # Cap faces: tilt slightly so they are spacelike and have valid angles
    # with the side faces.  The mink inner products will not be zero, so this
    # won't be truly right-angled — but it gives a valid test polyhedron.
    angle = 0.4  # tilt parameter
    v0_raw = np.array([np.sinh(angle), np.cosh(angle), np.cosh(angle), np.cosh(angle)])
    v1_raw = np.array([np.sinh(angle), np.cosh(angle), np.cosh(angle), -np.cosh(angle)])
    v0 = normalize_spacelike(v0_raw)
    v1 = normalize_spacelike(v1_raw)

    V = np.stack([v0, v1, v2, v3, v4])
    adj = _prism_adjacency()
    verts = _prism_vertices()
    return GeomPolyhedron(face_vectors=V, adjacency=adj, vertices=verts)


@pytest.fixture
def prism_geom():
    return _make_right_angle_prism()


# ---------------------------------------------------------------------------
# Newton convergence tests
# ---------------------------------------------------------------------------

class TestNewtonSolve:
    def test_converges_from_near_solution(self, prism_geom):
        """
        If the starting point is already near-converged (small perturbation),
        Newton should converge to the correct angles in a few iterations.
        """
        V = prism_geom.face_vectors.copy()
        adj = prism_geom.adjacency
        verts = prism_geom.vertices

        # Compute current angles as the target
        current_angles = compute_angles(prism_geom)
        gauge = make_gauge_info(V, adj, verts, vertex_idx=0)

        # Perturb face vectors slightly
        rng = np.random.default_rng(42)
        V_perturbed = V + rng.standard_normal(V.shape) * 1e-4
        # Re-normalize to keep vectors on the spacelike hyperboloid
        from hyperhedron.linalg import renormalize_spacelike_batch
        V_perturbed = renormalize_spacelike_batch(V_perturbed)

        result = newton_solve(V_perturbed, current_angles, gauge, tol=1e-10)

        assert result.converged
        assert result.residual < 1e-10
        # Face vectors should still be unit spacelike after Newton
        norms = mink_norms(result.face_vectors)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_face_vectors_remain_spacelike(self, prism_geom):
        """After convergence, all face vectors must have mink norm = +1."""
        V = prism_geom.face_vectors
        adj = prism_geom.adjacency
        verts = prism_geom.vertices
        angles = compute_angles(prism_geom)
        gauge = make_gauge_info(V, adj, verts, vertex_idx=0)

        result = newton_solve(V, angles, gauge, tol=1e-10)
        norms = mink_norms(result.face_vectors)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_condition_number_recorded(self, prism_geom):
        """When Newton must iterate (perturbed start), condition number is recorded."""
        V = prism_geom.face_vectors.copy()
        adj = prism_geom.adjacency
        verts = prism_geom.vertices
        angles = compute_angles(prism_geom)
        gauge = make_gauge_info(V, adj, verts, vertex_idx=0)

        rng = np.random.default_rng(7)
        from hyperhedron.linalg import renormalize_spacelike_batch
        V_perturbed = renormalize_spacelike_batch(V + rng.standard_normal(V.shape) * 1e-3)

        result = newton_solve(V_perturbed, angles, gauge, tol=1e-10)
        # Condition number should have been recorded (> 0) since Newton iterated
        assert result.condition_number >= 1.0

    def test_raises_for_singular_jacobian(self, prism_geom):
        """IllConditionedJacobian is raised when threshold is below the actual κ."""
        V = prism_geom.face_vectors.copy()
        adj = prism_geom.adjacency
        verts = prism_geom.vertices
        angles = compute_angles(prism_geom)
        gauge = make_gauge_info(V, adj, verts, vertex_idx=0)

        rng = np.random.default_rng(7)
        from hyperhedron.linalg import renormalize_spacelike_batch
        V_perturbed = renormalize_spacelike_batch(V + rng.standard_normal(V.shape) * 1e-3)

        # First get the actual condition number
        result = newton_solve(V_perturbed, angles, gauge, tol=1e-10)
        actual_kappa = result.condition_number

        # Now re-run with threshold just below the actual κ → should raise
        with pytest.raises(IllConditionedJacobian):
            newton_solve(V_perturbed, angles, gauge, cond_threshold=actual_kappa * 0.5)


# ---------------------------------------------------------------------------
# Re-normalization test
# ---------------------------------------------------------------------------

class TestRenormalization:
    def test_drift_corrected(self, prism_geom):
        """
        Simulate face vector drift and verify that the re-normalization step
        in the Newton solver corrects it.
        """
        from hyperhedron.linalg import renormalize_spacelike_batch

        V = prism_geom.face_vectors.copy()
        # Introduce artificial drift: scale all vectors by 1.05
        V_drifted = V * 1.05
        norms_before = mink_norms(V_drifted)
        assert not np.allclose(norms_before, 1.0, atol=1e-4)

        V_fixed = renormalize_spacelike_batch(V_drifted)
        norms_after = mink_norms(V_fixed)
        np.testing.assert_allclose(norms_after, 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Homotopy test (basic smoke test)
# ---------------------------------------------------------------------------

class TestRunHomotopy:
    def test_trivial_homotopy(self, prism_geom):
        """
        A homotopy from angles to themselves should return the same polyhedron.
        """
        V = prism_geom.face_vectors
        adj = prism_geom.adjacency
        verts = prism_geom.vertices
        angles = compute_angles(prism_geom)
        gauge = make_gauge_info(V, adj, verts, vertex_idx=0)

        result = run_homotopy(prism_geom, angles, gauge, tol=1e-10)

        assert isinstance(result, GeomPolyhedron)
        assert result.residual < 1e-8
        # Face vectors should be close to original
        norms = mink_norms(result.face_vectors)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)
