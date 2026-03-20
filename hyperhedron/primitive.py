"""
Construction of the primitive geometric element (double prism).

Ports prism.m and prim.m.

The primitive for a polyhedron with N faces is an (N-2)-sided prism
subjected to two Whitehead moves, giving the canonical starting
configuration for the homotopy.  All dihedral angles are set to 72°
(canonical position) via Newton's method.
"""

from __future__ import annotations

import logging

import numpy as np

from .angles import canonical_angle_matrix, compute_angles
from .linalg import mink, normalize_spacelike, null_space, solve_for_vertex
from .newton import run_homotopy
from .objective import make_gauge_info
from .polyhedron import GeomPolyhedron

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prism construction (port of prism.m)
# ---------------------------------------------------------------------------

def _mink_centroid(vertex_vectors: np.ndarray) -> np.ndarray:
    """Unit timelike vector pointing toward the centroid of vertex positions."""
    total = np.sum(vertex_vectors, axis=0)
    n = mink(total, total)
    return total / np.sqrt(abs(n))


def build_prism(n_sides: int) -> GeomPolyhedron:
    """
    Build an n_sides-sided prism as a starting geometric object.

    The prism has n_sides rectangular faces and 2 cap faces, giving
    n_sides + 2 total faces and 2*n_sides vertices.

    Returns a GeomPolyhedron in canonical 72° position.
    """
    n = n_sides
    num_vertices = 2 * n
    num_faces = n + 2

    # --- Vertex coordinates (in Poincaré-ball-like space, then lifted to hyperboloid) ---
    V_raw = np.zeros((num_vertices, 4))
    for i in range(n):
        V_raw[i, 1] = 0.95 * np.cos(i * 2 * np.pi / n)
        V_raw[i, 2] = 0.95 * np.sin(i * 2 * np.pi / n)
        V_raw[i, 3] = -0.2

    for i in range(n):
        V_raw[n + i, 1] = 0.95 * np.cos(i * 2 * np.pi / n)
        V_raw[n + i, 2] = 0.95 * np.sin(i * 2 * np.pi / n)
        V_raw[n + i, 3] = 0.2

    # Lift to hyperboloid (port of MATLAB's conversion)
    V = V_raw.copy()
    for i in range(num_vertices):
        r_sq = V[i, 1] ** 2 + V[i, 2] ** 2 + V[i, 3] ** 2
        V[i, :] /= (1.0 - r_sq)
        V[i, 0] = np.sqrt(1.0 + V[i, 1] ** 2 + V[i, 2] ** 2 + V[i, 3] ** 2)

    # --- Face definitions (0-indexed vertex lists) ---
    # Top cap (face 0): vertices 0 .. n-1
    # Bottom cap (face 1): vertices n .. 2n-1
    # Rectangular face i+2 (i=0..n-1): vertices i, (i+1)%n, n+(i+1)%n, n+i

    face_vertex_lists = []
    # top cap
    face_vertex_lists.append(list(range(n)))
    # bottom cap
    face_vertex_lists.append(list(range(n, 2 * n)))
    # rectangular sides
    for i in range(n):
        j = (i + 1) % n
        face_vertex_lists.append([i, j, n + j, n + i])

    # --- Vertex-face incidence ---
    # vertex k belongs to exactly 3 faces
    vert_list = []
    for k in range(num_vertices):
        faces_of_k = [f for f, fvl in enumerate(face_vertex_lists) if k in fvl]
        assert len(faces_of_k) == 3, f"Vertex {k} belongs to {len(faces_of_k)} faces"
        vert_list.append(sorted(faces_of_k))
    vertices_arr = np.array(vert_list, dtype=int)

    # --- Face normal vectors (via null space) ---
    center_v = _mink_centroid(V)

    face_vectors = np.zeros((num_faces, 4))
    for fi, fvl in enumerate(face_vertex_lists):
        # Pick any 3 vertices from this face
        A = V[fvl[:3], :].copy()
        A[:, 0] *= -1    # sign flip for Minkowski orthogonality
        ns = null_space(A)
        normal = ns[:, 0]
        norm_sq = mink(normal, normal)
        if norm_sq <= 0:
            raise ValueError(f"Face {fi} normal is not spacelike: mink={norm_sq:.4f}")
        normal = normal / np.sqrt(norm_sq)
        # Ensure normal points "inward" (mink product with vertex centroid < 0)
        if mink(center_v, normal) > 0:
            normal = -normal
        face_vectors[fi] = normal

    # --- Adjacency matrix ---
    adjacency = np.eye(num_faces, dtype=int)
    for f1 in range(num_faces):
        for f2 in range(f1 + 1, num_faces):
            shared = len(set(face_vertex_lists[f1]) & set(face_vertex_lists[f2]))
            if shared >= 2:
                adjacency[f1, f2] = adjacency[f2, f1] = 1

    geom = GeomPolyhedron(
        face_vectors=face_vectors,
        adjacency=adjacency,
        vertices=vertices_arr,
    )

    # Put in 72° canonical position if large enough
    if num_faces > 5:
        geom = _to_canonical(geom)

    return geom


def _to_canonical(geom: GeomPolyhedron, angle_deg: float = 72.0) -> GeomPolyhedron:
    """Deform all dihedral angles to angle_deg via Newton + homotopy.

    Uses a relaxed condition number threshold (1e20) to match the MATLAB
    canonical.m behaviour, which does not check the condition number.
    """
    target = canonical_angle_matrix(geom.adjacency, angle_deg)
    gauge = make_gauge_info(geom.face_vectors, geom.adjacency, geom.vertices, 0)
    return run_homotopy(geom, target, gauge, cond_threshold=1e20)


# ---------------------------------------------------------------------------
# Primitive construction  (port of prim.m)
# ---------------------------------------------------------------------------

def build_primitive(num_faces: int) -> GeomPolyhedron:
    """
    Build the primitive geometric element for a polyhedron with num_faces faces.

    The primitive is a double prism (an (num_faces-2)-sided prism subjected
    to two Whitehead moves) in canonical 72° position.

    This is the starting point for the geometric homotopy in construct_polyhedron().
    """
    from .whitehead import geometric_whitehead   # avoid circular import

    n_sides = num_faces - 2
    if n_sides < 3:
        raise ValueError(
            f"Cannot build primitive for {num_faces} faces; need at least 5."
        )

    poly = build_prism(n_sides)
    N = poly.num_faces   # = n_sides + 2

    logger.debug("Primitive: starting with %d-sided prism (%d faces)", n_sides, N)

    # --- First Whitehead move on edge (0, 2) (MATLAB prim.m: faces 1 and 3, 1-indexed) ---
    # Don't canonicalize yet — apply both moves first, then bring to 72° once.
    logger.debug("Primitive: first Whitehead move (0, 2)")
    poly = geometric_whitehead(0, 2, poly, canonicalize=False)

    # --- Second Whitehead move on edge (N-1, 1) (MATLAB prim.m: faces n and 2, 1-indexed) ---
    logger.debug("Primitive: second Whitehead move (%d, 1)", N - 1)
    poly = geometric_whitehead(N - 1, 1, poly, canonicalize=False)

    # --- Bring to 72° canonical position ---
    logger.debug("Primitive: canonical step")
    poly = _to_canonical(poly)

    logger.debug("Primitive built: %d faces, residual=%.3e", num_faces, poly.residual)
    return poly


def _pick_gauge_vertex(
    geom: GeomPolyhedron, face_i: int, face_j: int
) -> int:
    """
    Return the index of a vertex NOT at the edge (face_i, face_j).

    Used to choose a stable gauge vertex for Newton during a Whitehead move.
    """
    for v, triple in enumerate(geom.vertices):
        fset = set(int(x) for x in triple)
        if not (face_i in fset and face_j in fset):
            return v
    raise ValueError(
        f"No vertex found that avoids edge ({face_i}, {face_j})."
    )
