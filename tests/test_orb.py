"""
Tests for the .orb file reader (hyperhedron/io/orb.py).

Skipped if 1.orb is not present in the project root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ORB_FILE = Path(__file__).parent.parent / "1.orb"
pytestmark = pytest.mark.skipif(
    not ORB_FILE.exists(),
    reason="1.orb not found",
)


@pytest.fixture(scope="module")
def dodecahedron_comb():
    from hyperhedron.io.orb import read_orb
    return read_orb(ORB_FILE)


class TestOrbReader:
    def test_num_faces(self, dodecahedron_comb):
        # Dodecahedron has 12 pentagonal faces
        assert dodecahedron_comb.num_faces == 12

    def test_num_vertices(self, dodecahedron_comb):
        # Dodecahedron has 20 vertices
        assert dodecahedron_comb.num_vertices == 20

    def test_euler_characteristic(self, dodecahedron_comb):
        # V - E + F = 2  (sphere)
        F = dodecahedron_comb.num_faces
        V = dodecahedron_comb.num_vertices
        # Count edges from adjacency (off-diagonal 1s, divided by 2)
        E = (dodecahedron_comb.adjacency.sum() - F) // 2
        assert V - E + F == 2

    def test_every_face_has_five_neighbours(self, dodecahedron_comb):
        # Dodecahedron: each face is a pentagon, adjacent to exactly 5 others
        adj = dodecahedron_comb.adjacency
        face_degrees = adj.sum(axis=1) - 1   # subtract self-loop (diagonal=1)
        assert (face_degrees == 5).all(), f"Face degrees: {face_degrees}"

    def test_adjacency_is_symmetric(self, dodecahedron_comb):
        adj = dodecahedron_comb.adjacency
        assert np.array_equal(adj, adj.T)

    def test_diagonal_is_ones(self, dodecahedron_comb):
        adj = dodecahedron_comb.adjacency
        assert np.all(np.diag(adj) == 1)

    def test_each_vertex_in_exactly_three_faces(self, dodecahedron_comb):
        # In a simple polyhedron, every vertex is the meeting of exactly 3 faces
        assert dodecahedron_comb.vertices.shape == (20, 3)

    def test_vertex_face_indices_in_range(self, dodecahedron_comb):
        N = dodecahedron_comb.num_faces
        verts = dodecahedron_comb.vertices
        assert verts.min() >= 0
        assert verts.max() < N

    def test_vertex_face_triples_sorted(self, dodecahedron_comb):
        # Each triple (i, j, k) must satisfy i < j < k
        verts = dodecahedron_comb.vertices
        assert np.all(verts[:, 0] < verts[:, 1])
        assert np.all(verts[:, 1] < verts[:, 2])

    def test_validate_passes(self, dodecahedron_comb):
        # CombPolyhedron.validate() raises ValueError on inconsistent data
        dodecahedron_comb.validate()

    def test_construct_polyhedron(self, dodecahedron_comb):
        """Full pipeline: orb → CombPolyhedron → right-angled geometric realisation."""
        from hyperhedron.make import construct_polyhedron
        geom = construct_polyhedron(dodecahedron_comb)
        assert geom.residual < 1e-6, f"|f| = {geom.residual:.3e}"
        # All dihedral angles should be close to 90°
        from hyperhedron.angles import compute_angles
        angles = compute_angles(geom)
        import numpy as np
        adj = geom.adjacency
        for i in range(geom.num_faces):
            for j in range(i + 1, geom.num_faces):
                if adj[i, j] == 1:
                    assert abs(angles[i, j] - 90.0) < 0.1, (
                        f"angle({i},{j}) = {angles[i,j]:.4f}° (expected 90°)"
                    )
