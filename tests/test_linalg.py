"""
Tests for hyperhedron.linalg — Minkowski geometry primitives.

These tests verify the mathematical foundation that all other modules depend on.
"""

import numpy as np
import pytest

from hyperhedron.exceptions import IdealVertexError, NumericalError
from hyperhedron.linalg import (
    mink,
    mink_matrix,
    mink_norms,
    normalize_spacelike,
    normalize_timelike,
    renormalize_spacelike_batch,
    solve_for_vertex,
    to_klein,
    to_poincare,
)


# ---------------------------------------------------------------------------
# Minkowski inner product
# ---------------------------------------------------------------------------

class TestMinkInnerProduct:
    def test_basis_vectors(self):
        e0 = np.array([1., 0., 0., 0.])
        e1 = np.array([0., 1., 0., 0.])
        e2 = np.array([0., 0., 1., 0.])
        e3 = np.array([0., 0., 0., 1.])
        assert mink(e0, e0) == pytest.approx(-1.0)
        assert mink(e1, e1) == pytest.approx(+1.0)
        assert mink(e2, e2) == pytest.approx(+1.0)
        assert mink(e3, e3) == pytest.approx(+1.0)
        assert mink(e0, e1) == pytest.approx(0.0)

    def test_symmetry(self):
        a = np.array([2., 1., 3., 1.])
        b = np.array([1., 2., 1., 4.])
        assert mink(a, b) == pytest.approx(mink(b, a))

    def test_known_value(self):
        # <(1,0,0,0), (2,1,1,1)> = -1*2 + 0 + 0 + 0 = -2
        a = np.array([1., 0., 0., 0.])
        b = np.array([2., 1., 1., 1.])
        assert mink(a, b) == pytest.approx(-2.0)

    def test_spacelike_face_vector(self):
        # A face vector must satisfy mink(v, v) = +1
        v = np.array([0., 1., 0., 0.])
        assert mink(v, v) == pytest.approx(1.0)

    def test_timelike_vertex_vector(self):
        # A vertex vector must satisfy mink(v, v) = -1
        v = np.array([1., 0., 0., 0.])
        assert mink(v, v) == pytest.approx(-1.0)


class TestMinkMatrix:
    def test_single_pair(self):
        A = np.array([[1., 0., 0., 0.]])
        B = np.array([[2., 1., 1., 1.]])
        result = mink_matrix(A, B)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(-2.0)

    def test_batch(self):
        # A = [e0, e1], B = [e0, e1]
        A = np.eye(4)[:2]   # (2, 4)
        B = np.eye(4)[:2]   # (2, 4)
        result = mink_matrix(A, B)
        # <e0, e0> = -1, <e0, e1> = 0, <e1, e0> = 0, <e1, e1> = +1
        expected = np.array([[-1., 0.], [0., 1.]])
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_consistent_with_mink(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 4))
        B = rng.standard_normal((7, 4))
        matrix = mink_matrix(A, B)
        for i in range(5):
            for j in range(7):
                assert matrix[i, j] == pytest.approx(mink(A[i], B[j]))


class TestMinkNorms:
    def test_spacelike(self):
        V = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.]])
        norms = mink_norms(V)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-15)

    def test_timelike(self):
        V = np.array([[1., 0., 0., 0.]])
        norms = mink_norms(V)
        np.testing.assert_allclose(norms, [-1.0], atol=1e-15)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalizeSpacelike:
    def test_basic(self):
        v = np.array([0., 3., 0., 0.])
        nv = normalize_spacelike(v)
        assert mink(nv, nv) == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_allclose(nv, [0., 1., 0., 0.], atol=1e-14)

    def test_mixed(self):
        v = np.array([1., 2., 2., 0.])  # mink = -1 + 4 + 4 = 7
        nv = normalize_spacelike(v)
        assert mink(nv, nv) == pytest.approx(1.0, abs=1e-14)

    def test_raises_on_timelike(self):
        v = np.array([2., 0., 0., 0.])  # mink = -4
        with pytest.raises(NumericalError):
            normalize_spacelike(v)

    def test_raises_on_null(self):
        v = np.array([1., 1., 0., 0.])  # mink = -1 + 1 = 0
        with pytest.raises(NumericalError):
            normalize_spacelike(v)


class TestNormalizeTimelike:
    def test_basic(self):
        v = np.array([3., 0., 0., 0.])  # mink = -9
        nv = normalize_timelike(v)
        assert mink(nv, nv) == pytest.approx(-1.0, abs=1e-14)
        assert nv[0] > 0

    def test_future_pointing(self):
        v = np.array([-3., 0., 0., 0.])   # past-pointing
        nv = normalize_timelike(v)
        assert nv[0] > 0                   # flipped to future-pointing

    def test_mixed_timelike(self):
        v = np.array([2., 1., 0., 0.])  # mink = -4 + 1 = -3
        nv = normalize_timelike(v)
        assert mink(nv, nv) == pytest.approx(-1.0, abs=1e-14)

    def test_raises_on_spacelike(self):
        v = np.array([0., 2., 0., 0.])  # mink = +4
        with pytest.raises(IdealVertexError):
            normalize_timelike(v)

    def test_raises_on_null(self):
        v = np.array([1., 1., 0., 0.])  # mink = 0
        with pytest.raises(IdealVertexError):
            normalize_timelike(v)


class TestRenormalizeSpacelikeBatch:
    def test_already_normalized(self):
        V = np.array([[0., 1., 0., 0.], [0., 0., 0., 1.]])
        Vn = renormalize_spacelike_batch(V)
        norms = mink_norms(Vn)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-14)

    def test_scales_correctly(self):
        V = np.array([[0., 3., 0., 0.], [0., 0., 2., 0.]])
        Vn = renormalize_spacelike_batch(V)
        norms = mink_norms(Vn)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-14)


# ---------------------------------------------------------------------------
# Vertex computation
# ---------------------------------------------------------------------------

class TestSolveForVertex:
    def _axis_faces(self):
        """Three axis-aligned face normals; their intersection is the origin vertex."""
        # Face normals along spatial axes: spacelike, unit
        return np.array([
            [0., 1., 0., 0.],   # face 0: normal e_1
            [0., 0., 1., 0.],   # face 1: normal e_2
            [0., 0., 0., 1.],   # face 2: normal e_3
        ])

    def test_axis_vertex_is_origin(self):
        # null space of diag(-1,1,1,1) restricted to e1,e2,e3 rows should be e0
        face_vectors = self._axis_faces()
        v = solve_for_vertex(face_vectors, 0, 1, 2)
        # v should be (±1, 0, 0, 0), normalized to mink=-1, future-pointing
        np.testing.assert_allclose(np.abs(v), [1., 0., 0., 0.], atol=1e-12)
        assert v[0] > 0

    def test_vertex_is_timelike(self):
        face_vectors = self._axis_faces()
        v = solve_for_vertex(face_vectors, 0, 1, 2)
        assert mink(v, v) == pytest.approx(-1.0, abs=1e-12)

    def test_vertex_orthogonal_to_faces(self):
        face_vectors = self._axis_faces()
        v = solve_for_vertex(face_vectors, 0, 1, 2)
        for i in range(3):
            assert mink(face_vectors[i], v) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Coordinate projections
# ---------------------------------------------------------------------------

class TestKleinProjection:
    def test_along_x_axis(self):
        r = 0.5
        v = np.array([np.cosh(r), np.sinh(r), 0., 0.])
        K = to_klein(v)
        np.testing.assert_allclose(K, [np.tanh(r), 0., 0.], atol=1e-14)

    def test_inside_unit_ball(self):
        for r in [0.1, 0.5, 1.0, 2.0]:
            v = np.array([np.cosh(r), np.sinh(r), 0., 0.])
            K = to_klein(v)
            assert np.linalg.norm(K) < 1.0

    def test_raises_for_ideal_point(self):
        v = np.array([1e-15, 1., 0., 0.])
        with pytest.raises(NumericalError):
            to_klein(v)


class TestPoincareProjection:
    def test_inside_unit_ball(self):
        for r in [0.1, 0.5, 1.0, 2.0]:
            v = np.array([np.cosh(r), np.sinh(r), 0., 0.])
            P = to_poincare(v)
            assert np.linalg.norm(P) < 1.0

    def test_poincare_smaller_than_klein(self):
        # Poincaré coordinates are always closer to origin than Klein coords
        r = 1.0
        v = np.array([np.cosh(r), np.sinh(r), 0., 0.])
        K = to_klein(v)
        P = to_poincare(v)
        assert np.linalg.norm(P) < np.linalg.norm(K)

    def test_consistency_with_klein(self):
        r = 0.7
        v = np.array([np.cosh(r), 0., np.sinh(r), 0.])
        K = to_klein(v)
        P = to_poincare(v)
        # P = K / (1 + sqrt(1 - |K|^2))
        norm_sq = np.dot(K, K)
        expected = K / (1.0 + np.sqrt(1.0 - norm_sq))
        np.testing.assert_allclose(P, expected, atol=1e-14)
