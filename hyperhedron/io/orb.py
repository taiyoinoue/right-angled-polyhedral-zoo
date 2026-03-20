"""
Orb (.orb) file reader.

The .orb format is produced by the Orb software for hyperbolic orbifolds.
It lists tetrahedra that triangulate the orbifold; the second column of each
row, when non-zero, gives an edge index of the underlying polyhedron's
1-skeleton, and the fourth and fifth columns give the two vertex indices
connected by that edge.

Ports nothing directly — this is a new I/O module for the Python pipeline.

Usage
-----
    from hyperhedron.io.orb import read_orb
    comb = read_orb("1.orb")
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from ..exceptions import InvalidGraphError
from ..polyhedron import CombPolyhedron


def read_orb(path: str | Path) -> CombPolyhedron:
    """
    Parse a .orb file and return a CombPolyhedron.

    Extracts the polyhedron's 1-skeleton (vertices + edges) from the
    tetrahedra listing, uses a planar embedding to recover the faces,
    then builds the face-adjacency matrix and vertex triples.

    Parameters
    ----------
    path:
        Path to the .orb file.

    Returns
    -------
    CombPolyhedron with 0-indexed face labels.
    """
    edges = _parse_1skeleton(Path(path))
    return _skeleton_to_comb(edges)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_1skeleton(path: Path) -> list[tuple[int, int]]:
    """
    Extract the polyhedron's 1-skeleton from a .orb file.

    Returns a list of (v1, v2) pairs (1-indexed vertex IDs).
    Each edge index appears in multiple tetrahedra; only the first
    occurrence is kept.
    """
    edge_map: dict[int, tuple[int, int]] = {}

    for raw in path.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            edge_idx = int(parts[1])
            v1       = int(parts[3])
            v2       = int(parts[4])
        except ValueError:
            # Coordinate data section or header — parts[1] is a float
            continue

        if edge_idx == 0 or v1 <= 0 or v2 <= 0:
            # Not a 1-skeleton edge (interior face or ideal vertex)
            continue

        if edge_idx not in edge_map:
            edge_map[edge_idx] = (v1, v2)

    if not edge_map:
        raise InvalidGraphError(f"No 1-skeleton edges found in {path}")

    return list(edge_map.values())


# ---------------------------------------------------------------------------
# 1-skeleton → CombPolyhedron
# ---------------------------------------------------------------------------

def _skeleton_to_comb(edges: list[tuple[int, int]]) -> CombPolyhedron:
    """
    Build a CombPolyhedron from the polyhedron's 1-skeleton edge list.

    Uses networkx's planar embedding to recover the faces, then constructs
    the N×N face-adjacency matrix and M×3 vertex-triple array.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise InvalidGraphError(
            "The 1-skeleton graph is not planar; cannot recover faces."
        )

    faces = _enumerate_faces(embedding)   # list of vertex-index lists
    N = len(faces)

    # Map: polyhedron-vertex → sorted list of face indices containing it
    vertex_to_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(faces):
        for v in face:
            vertex_to_faces.setdefault(v, []).append(fi)

    # Map: edge (frozenset) → list of the two face indices sharing it
    edge_to_faces: dict[frozenset, list[int]] = {}
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            key = frozenset([face[i], face[(i + 1) % n]])
            edge_to_faces.setdefault(key, []).append(fi)

    # Face-adjacency matrix
    adjacency = np.eye(N, dtype=int)
    for face_list in edge_to_faces.values():
        if len(face_list) == 2:
            fi, fj = face_list
            adjacency[fi, fj] = adjacency[fj, fi] = 1

    # Vertex triples (0-indexed face labels, sorted i < j < k)
    vert_list = []
    for v in sorted(vertex_to_faces):
        fi_list = sorted(vertex_to_faces[v])
        if len(fi_list) != 3:
            raise InvalidGraphError(
                f"Vertex {v} belongs to {len(fi_list)} faces; "
                "expected exactly 3 (polyhedron must be simple and 3-connected)."
            )
        vert_list.append(fi_list)

    vertices = np.array(vert_list, dtype=int)
    comb = CombPolyhedron(adjacency=adjacency, vertices=vertices)

    try:
        comb.validate()
    except ValueError as e:
        raise InvalidGraphError(str(e)) from e

    return comb


# ---------------------------------------------------------------------------
# Planar-face enumeration (helper)
# ---------------------------------------------------------------------------

def _enumerate_faces(embedding: nx.PlanarEmbedding) -> list[list[int]]:
    """
    Return every face of a planar embedding as an ordered list of vertex indices.

    Traverses every directed half-edge exactly once; the face containing
    half-edge (u→v) is the cycle returned by embedding.traverse_face(u, v).
    """
    seen: set[tuple[int, int]] = set()
    faces: list[list[int]] = []

    for u in embedding:
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in seen:
                continue
            face = embedding.traverse_face(u, v)
            n = len(face)
            for i in range(n):
                seen.add((face[i], face[(i + 1) % n]))
            faces.append(face)

    return faces
