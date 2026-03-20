"""
Tests for hyperhedron.combinatorial.

We test with small, known polyhedra:
  - Tetrahedron (4 faces, simplest case)
  - Triangular prism (5 faces)
  - Cube (6 faces)
  - Dodecahedron (12 faces, from Roeder's dodec_combinatorics.m)
"""

import numpy as np
import pytest

from hyperhedron.combinatorial import (
    WhiteheadMove,
    comb_algorithm,
    comb_whitehead,
    find_v_inf,
    interior_to_cycle,
    is_3prismatic,
)
from hyperhedron.exceptions import CycleOrderingError, InvalidGraphError
from hyperhedron.polyhedron import CombPolyhedron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tetrahedron() -> CombPolyhedron:
    """Tetrahedron: 4 faces, all mutually adjacent. V=4, E=6, F=4."""
    N = 4
    adj = np.ones((N, N), dtype=int)
    verts = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)
    return CombPolyhedron(adjacency=adj, vertices=verts)


def _cube() -> CombPolyhedron:
    """Cube: 6 faces. Opposite faces are NOT adjacent."""
    adj = np.array([
        [1,1,1,1,0,1],  # face 0 (top) adj to 1,2,3,5 but not 4 (bottom)
        [1,1,1,0,1,1],  # face 1
        [1,1,1,1,1,0],
        [1,0,1,1,1,1],
        [0,1,1,1,1,1],
        [1,1,0,1,1,1],
    ], dtype=int)
    # Build vertices from triangles in the adjacency graph
    N = 6
    verts = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i,j]:
                for k in range(j+1, N):
                    if adj[i,k] and adj[j,k]:
                        verts.append([i,j,k])
    return CombPolyhedron(adjacency=adj, vertices=np.array(verts, dtype=int))


def _dodecahedron() -> CombPolyhedron:
    """Dodecahedron from Roeder's dodec_combinatorics.m (converted to 0-indexed)."""
    adj = np.array([
        [1,0,1,0,0,0,1,1,0,1,0,1],
        [0,1,0,1,1,1,0,0,1,0,1,0],
        [1,0,1,0,0,1,1,0,0,1,1,0],
        [0,1,0,1,0,1,1,0,1,0,0,1],
        [0,1,0,0,1,0,0,1,1,1,1,0],
        [0,1,1,1,0,1,1,0,0,0,1,0],
        [1,0,1,1,0,1,1,0,0,0,0,1],
        [1,0,0,0,1,0,0,1,1,1,0,1],
        [0,1,0,1,1,0,0,1,1,0,0,1],
        [1,0,1,0,1,0,0,1,0,1,1,0],
        [0,1,1,0,1,1,0,0,0,1,1,0],
        [1,0,0,1,0,0,1,1,1,0,0,1],
    ], dtype=int)
    # Vertices from MATLAB (converted from 1-indexed to 0-indexed)
    verts_1idx = [
        [3,6,11],[2,4,6],[2,4,9],[1,3,10],[1,8,12],[4,7,12],
        [1,7,12],[4,6,7],[3,6,7],[1,3,7],[5,10,11],[5,8,10],
        [5,8,9],[2,6,11],[2,5,11],[2,5,9],[8,9,12],[4,9,12],
        [3,10,11],[1,8,10],
    ]
    verts = np.array([[a-1, b-1, c-1] for a,b,c in verts_1idx], dtype=int)
    return CombPolyhedron(adjacency=adj, vertices=verts)


# ---------------------------------------------------------------------------
# is_3prismatic
# ---------------------------------------------------------------------------

class TestIs3Prismatic:
    def test_tetrahedron_not_3prismatic(self):
        assert not is_3prismatic(_tetrahedron())

    def test_cube_not_3prismatic(self):
        assert not is_3prismatic(_cube())

    def test_dodecahedron_not_3prismatic(self):
        assert not is_3prismatic(_dodecahedron())


# ---------------------------------------------------------------------------
# find_v_inf
# ---------------------------------------------------------------------------

class TestFindVInf:
    def test_returns_three_parts(self):
        comb = _cube()
        v_inf, cycle, interior = find_v_inf(comb)
        assert isinstance(v_inf, int)
        assert 0 <= v_inf < comb.num_faces

    def test_cycle_plus_interior_plus_vinf_covers_all_faces(self):
        comb = _cube()
        v_inf, cycle, interior = find_v_inf(comb)
        all_faces = set(range(comb.num_faces))
        assert set(cycle) | set(interior) | {v_inf} == all_faces

    def test_cycle_is_ordered(self):
        """Adjacent cycle members must be adjacent in the face graph."""
        comb = _cube()
        v_inf, cycle, interior = find_v_inf(comb)
        n = len(cycle)
        for i in range(n):
            assert comb.adjacency[cycle[i], cycle[(i+1) % n]] == 1, \
                f"cycle[{i}]={cycle[i]} and cycle[{(i+1)%n}]={cycle[(i+1)%n]} not adjacent"

    def test_dodecahedron_interior_count(self):
        comb = _dodecahedron()
        v_inf, cycle, interior = find_v_inf(comb)
        # v_inf has degree 3, so cycle has 3 elements; 12 - 1 - 3 = 8 interior
        assert len(interior) == comb.num_faces - 1 - len(cycle)


# ---------------------------------------------------------------------------
# interior_to_cycle
# ---------------------------------------------------------------------------

class TestInteriorToCycle:
    def test_shape(self):
        comb = _cube()
        v_inf, cycle, interior = find_v_inf(comb)
        M = interior_to_cycle(comb, cycle, interior)
        assert M.shape == (len(cycle), len(interior))

    def test_values_match_adjacency(self):
        comb = _cube()
        v_inf, cycle, interior = find_v_inf(comb)
        M = interior_to_cycle(comb, cycle, interior)
        for j, cf in enumerate(cycle):
            for i, inf in enumerate(interior):
                assert M[j, i] == comb.adjacency[cf, inf]


# ---------------------------------------------------------------------------
# comb_whitehead
# ---------------------------------------------------------------------------

class TestCombWhitehead:
    def test_edge_removed_and_new_edge_added(self):
        comb = _cube()
        # Pick an existing edge
        adj = comb.adjacency
        N = comb.num_faces
        i, j = next((a, b) for a in range(N) for b in range(a+1, N) if adj[a,b])
        new_comb, k, l = comb_whitehead(i, j, comb)
        assert new_comb.adjacency[i, j] == 0
        assert new_comb.adjacency[k, l] == 1

    def test_symmetry(self):
        comb = _cube()
        adj = comb.adjacency
        N = comb.num_faces
        i, j = next((a, b) for a in range(N) for b in range(a+1, N) if adj[a,b])
        new_comb, k, l = comb_whitehead(i, j, comb)
        assert new_comb.adjacency[j, i] == 0
        assert new_comb.adjacency[l, k] == 1

    def test_vertex_count_preserved(self):
        comb = _cube()
        adj = comb.adjacency
        N = comb.num_faces
        i, j = next((a, b) for a in range(N) for b in range(a+1, N) if adj[a,b])
        new_comb, k, l = comb_whitehead(i, j, comb)
        assert new_comb.num_vertices == comb.num_vertices

    def test_raises_for_non_edge(self):
        comb = _cube()
        # Find a non-adjacent pair (opposite faces on a cube)
        adj = comb.adjacency
        N = comb.num_faces
        ni, nj = next((a,b) for a in range(N) for b in range(a+1,N) if adj[a,b]==0 and a!=b)
        with pytest.raises(InvalidGraphError):
            comb_whitehead(ni, nj, comb)


# ---------------------------------------------------------------------------
# comb_algorithm  (high-level smoke tests)
# ---------------------------------------------------------------------------

class TestCombAlgorithm:
    def test_dodecahedron_terminates(self):
        result = comb_algorithm(_dodecahedron())
        assert len(result.forward_moves) > 0
        assert len(result.forward_moves) == len(result.inverse_moves)

    def test_permutation_is_bijection(self):
        comb = _dodecahedron()
        result = comb_algorithm(comb)
        P = result.permutation
        N = comb.num_faces
        assert set(P) == set(range(N)), "Permutation must be a bijection on face indices"

    def test_permutation_length(self):
        comb = _dodecahedron()
        result = comb_algorithm(comb)
        assert len(result.permutation) == comb.num_faces
