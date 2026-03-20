"""
Geomview OFF file reader and writer for hyperbolic polyhedra.

Ports write_poly.m.  The OFF format stores vertices in Klein (projective)
coordinates.  Geomview interprets these as projective coordinates in the
Poincaré ball model.

Centering: the polyhedron is centred at the hyperbolic origin by applying
a Lorentz boost that maps the vertex centroid to (1,0,0,0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..linalg import mink, null_space, solve_for_vertex, to_klein
from ..polyhedron import GeomPolyhedron

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compute vertex positions from face vectors
# ---------------------------------------------------------------------------

def compute_vertex_positions(geom: GeomPolyhedron) -> np.ndarray:
    """
    Compute the (M, 4) array of hyperboloid vertex positions.

    Each vertex is the null-space intersection of its three face normals,
    normalised to mink norm = -1, future-pointing.
    """
    M = geom.num_vertices
    positions = np.zeros((M, 4))
    for v in range(M):
        i, j, k = int(geom.vertices[v, 0]), int(geom.vertices[v, 1]), int(geom.vertices[v, 2])
        positions[v] = solve_for_vertex(geom.face_vectors, i, j, k)
    return positions


# ---------------------------------------------------------------------------
# Centering isometry  (Lorentz boost to hyperbolic origin)
# ---------------------------------------------------------------------------

def _centroid(vertex_positions: np.ndarray) -> np.ndarray:
    """Unit timelike vector toward the centroid of the vertices."""
    total = np.sum(vertex_positions, axis=0)
    n = mink(total, total)
    return total / np.sqrt(abs(n))


def _centering_matrix(cent: np.ndarray) -> np.ndarray:
    """
    Build the 4×4 Lorentz matrix that maps cent → (1,0,0,0).

    Implements the Minkowski Gram-Schmidt procedure from write_poly.m.
    """
    # Row 0: mink-dual of centroid
    row0 = np.array([-cent[0], cent[1], cent[2], cent[3]])

    # v1: a spacelike vector orthogonal to cent
    T = cent.reshape(1, 4)
    ns = null_space(T)
    v1 = ns[:, 0]
    v1 = v1 / np.sqrt(mink(v1, v1))

    # v2: spacelike, orthogonal to both cent and v1
    T2 = np.stack([cent, np.array([-v1[0], v1[1], v1[2], v1[3]])])
    ns2 = null_space(T2)
    v2 = ns2[:, 0]
    v2 = v2 / np.sqrt(mink(v2, v2))

    # v3: spacelike, orthogonal to cent, v1, v2
    T3 = np.stack([
        cent,
        np.array([-v1[0], v1[1], v1[2], v1[3]]),
        np.array([-v2[0], v2[1], v2[2], v2[3]]),
    ])
    ns3 = null_space(T3)
    v3 = ns3[:, 0]
    v3 = v3 / np.sqrt(mink(v3, v3))

    A = np.stack([row0, v1, v2, v3])
    return A


def center_vertex_positions(vertex_positions: np.ndarray) -> np.ndarray:
    """Apply the centering isometry and return (M,4) centred positions."""
    cent = _centroid(vertex_positions)
    A = _centering_matrix(cent)
    return (A @ vertex_positions.T).T


# ---------------------------------------------------------------------------
# Face traversal (for ordered vertex lists per face)
# ---------------------------------------------------------------------------

def _vertex_in_face(v: int, face: int, vertices: np.ndarray) -> bool:
    return any(int(vertices[v, k]) == face for k in range(3))


def _share_two_faces(vi: int, vj: int, vertices: np.ndarray) -> bool:
    si = set(int(vertices[vi, k]) for k in range(3))
    sj = set(int(vertices[vj, k]) for k in range(3))
    return len(si & sj) >= 2


def _ordered_vertices_for_face(face: int, geom: GeomPolyhedron) -> list[int]:
    """
    Return the vertex indices around face in traversal order.

    Starting at any vertex belonging to the face, walks to adjacent vertices
    (those sharing 2 faces with the current one) that also belong to the face.
    """
    M = geom.num_vertices
    verts = geom.vertices

    # Find a starting vertex
    start = next(v for v in range(M) if _vertex_in_face(v, face, verts))
    order = [start]
    prev = start
    current = start

    while True:
        found = False
        for v in range(M):
            if (
                v != current
                and v != prev
                and _vertex_in_face(v, face, verts)
                and _share_two_faces(v, current, verts)
            ):
                if v == order[0] and len(order) >= 3:
                    return order    # completed the cycle
                if v not in order:
                    order.append(v)
                    prev, current = current, v
                    found = True
                    break
        if not found:
            break

    return order


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_off(geom: GeomPolyhedron, path: str | Path) -> None:
    """
    Write a GeomPolyhedron to a Geomview OFF file.

    Vertex coordinates are written in Klein (projective) model coordinates
    after applying the centering isometry.
    """
    path = Path(path)
    if path.suffix.lower() != ".off":
        logger.warning("File %s does not end in .off; Geomview may not open it.", path)

    N = geom.num_faces
    M = geom.num_vertices

    # Compute and centre vertex positions
    hyperboloid_verts = compute_vertex_positions(geom)
    centred = center_vertex_positions(hyperboloid_verts)

    # Project to Klein coordinates
    klein = np.zeros((M, 3))
    for v in range(M):
        klein[v] = to_klein(centred[v])

    # Build vertex adjacency (for face traversal)
    face_orders = [_ordered_vertices_for_face(f, geom) for f in range(N)]

    E_approx = N + M - 2   # Euler: V - E + F = 2 → E = V + F - 2
    with open(path, "w") as fid:
        fid.write(f"{M} {N} {E_approx}\n")
        for v in range(M):
            fid.write("\n")
            fid.write(" ".join(f"{x:.8f}" for x in klein[v]))
        fid.write("\n")
        for f, vorder in enumerate(face_orders):
            # Format: num_vertices v0 v1 ... vn-1 color_index
            fid.write("\n")
            k = len(vorder)
            fid.write(f"{k} " + " ".join(str(v) for v in vorder) + f" {f}")
        fid.write("\n")

    logger.info("Wrote %s: %d vertices, %d faces.", path, M, N)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_off(path: str | Path, adjacency: np.ndarray, vertices: np.ndarray) -> GeomPolyhedron:
    """
    Load a GeomPolyhedron from a Geomview OFF file.

    Because OFF stores vertices (not face normals), this function recovers
    the face normal vectors from the vertex positions.

    Parameters
    ----------
    path:       Path to the .off file.
    adjacency:  (N,N) int adjacency matrix (must be known independently).
    vertices:   (M,3) int vertex-face incidence (must be known independently).
    """
    from ..linalg import normalize_spacelike

    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

    # First non-empty line: V F E counts
    header = lines[0].split()
    num_verts = int(header[0])
    num_faces = int(header[1])

    # Read vertex Klein coordinates
    klein_coords = []
    idx = 1
    for _ in range(num_verts):
        while idx < len(lines) and not lines[idx]:
            idx += 1
        coords = [float(x) for x in lines[idx].split()]
        klein_coords.append(coords)
        idx += 1
    klein = np.array(klein_coords)   # (V, 3)

    # Convert Klein → hyperboloid
    # Klein point K satisfies |K|<1; hyperboloid: v0 = 1/sqrt(1-|K|^2), v1:3 = K/sqrt(...)
    hyp_verts = np.zeros((num_verts, 4))
    for v in range(num_verts):
        K = klein[v]
        norm_sq = np.dot(K, K)
        denom = np.sqrt(max(1.0 - norm_sq, 1e-30))
        hyp_verts[v, 0] = 1.0 / denom
        hyp_verts[v, 1:] = K / denom

    # Recover face normals from vertex triples
    N = num_faces
    face_vectors = np.zeros((N, 4))
    center = np.sum(hyp_verts, axis=0)

    for fi in range(N):
        # Find 3 vertices belonging to this face
        verts_of_face = [v for v in range(num_verts) if fi in set(int(x) for x in vertices[v])]
        i0, i1, i2 = verts_of_face[:3]
        A = hyp_verts[[i0, i1, i2], :].copy()
        A[:, 0] *= -1
        ns = null_space(A)
        normal = ns[:, 0]
        norm_sq = mink(normal, normal)
        normal = normal / np.sqrt(abs(norm_sq))
        if mink(normal, center) > 0:
            normal = -normal
        face_vectors[fi] = normal

    return GeomPolyhedron(
        face_vectors=face_vectors,
        adjacency=adjacency,
        vertices=vertices,
    )
