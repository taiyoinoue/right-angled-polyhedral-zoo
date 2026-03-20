"""
Geometric Whitehead moves.

A Whitehead move replaces edge (face_i, face_j) with edge (face_k, face_l),
changing the combinatorial type while deforming the geometry continuously.

The move proceeds in three Newton/homotopy stages:
  1. Pre-move: approach the topology boundary by setting angle(i,j) → small.
  2. Flip:     update adjacency; set angle(k,l) → same small value.
  3. Post-move: deform to canonical 72° position.

Ports whitehead.m and canonical_choose_vertex.m.
"""

from __future__ import annotations

import logging

import numpy as np

from .angles import canonical_angle_matrix, compute_angles
from .combinatorial import comb_whitehead
from .newton import run_homotopy, newton_solve
from .objective import make_gauge_info
from .polyhedron import GeomPolyhedron

logger = logging.getLogger(__name__)

# Angle used to approach the topology-change boundary
_NEAR_ZERO_ANGLE = 5.0
# Angle used for "almost right-angle" during pre-move softening
_SOFT_ANGLE = 89.0
# Canonical angle for inter-step positioning
_CANONICAL_ANGLE = 72.0


def _normalize_face_orientations(geom: GeomPolyhedron) -> GeomPolyhedron:
    """
    Ensure all face normals point "inward" (mink product with center-of-normals < 0).
    Mirrors the orientation fix in whitehead.m.
    """
    V = geom.face_vectors.copy()
    center = np.sum(V, axis=0)
    from .linalg import mink
    for n in range(geom.num_faces):
        if mink(V[n], center) > 0:
            V[n] = -V[n]
    return geom.replace_vectors(V)


def combinatorial_whitehead_geom(
    face_i: int,
    face_j: int,
    geom: GeomPolyhedron,
    gauge_vertex_idx: int,
) -> GeomPolyhedron:
    """
    Perform only the combinatorial part of a Whitehead move (no Newton solve).

    Updates adjacency and vertex list; face vectors are carried over unchanged.
    Used during primitive construction where the geometry is re-solved immediately
    after by a separate Newton call.
    """
    comb = geom.to_comb()
    new_comb, k, l = comb_whitehead(face_i, face_j, comb)
    return GeomPolyhedron(
        face_vectors=geom.face_vectors.copy(),
        adjacency=new_comb.adjacency,
        vertices=new_comb.vertices,
        residual=geom.residual,
        condition_number=geom.condition_number,
    )


def geometric_whitehead(
    face_i: int,
    face_j: int,
    geom: GeomPolyhedron,
    gauge_vertex_idx: int | None = None,
    canonicalize: bool = True,
) -> GeomPolyhedron:
    """
    Full geometric Whitehead move on edge (face_i, face_j).

    Deforms the polyhedron geometry continuously through the topology change,
    then returns a new GeomPolyhedron in canonical 72° position.

    Parameters
    ----------
    face_i, face_j:
        0-indexed face labels of the edge to flip.
    geom:
        Current geometric polyhedron.
    gauge_vertex_idx:
        Index into geom.vertices to use for gauge fixing.  If None, a vertex
        not on the edge (face_i, face_j) is chosen automatically.

    Returns
    -------
    GeomPolyhedron with the new combinatorial type, in 72° canonical position.
    """
    from .primitive import _pick_gauge_vertex

    geom = _normalize_face_orientations(geom)

    comb = geom.to_comb()
    _, k, l = comb_whitehead(face_i, face_j, comb)   # find k, l without modifying geom

    if gauge_vertex_idx is None:
        gauge_vertex_idx = _pick_gauge_vertex(geom, face_i, face_j)

    N = geom.num_faces
    angles = compute_angles(geom)

    # ------------------------------------------------------------------
    # Stage 1: pre-move deformation
    # Soften the angles around the edge being flipped, setting angle(i,j) → 5°.
    # ------------------------------------------------------------------
    pre_angles = angles.copy()
    pre_angles[face_i, face_j] = pre_angles[face_j, face_i] = _NEAR_ZERO_ANGLE
    for pair in [(face_i, k), (face_i, l), (face_j, k), (face_j, l)]:
        pi, pj = pair
        if geom.adjacency[pi, pj] == 1:
            pre_angles[pi, pj] = pre_angles[pj, pi] = _SOFT_ANGLE

    gauge = make_gauge_info(
        geom.face_vectors, geom.adjacency, geom.vertices, gauge_vertex_idx
    )
    poly_pre = run_homotopy(geom, pre_angles, gauge, cond_threshold=1e20)
    logger.debug("Whitehead (%d,%d): pre-move done, |f|=%.3e", face_i, face_j, poly_pre.residual)

    # ------------------------------------------------------------------
    # Stage 2: topology flip
    # Update adjacency, then solve for the new edge at a small angle.
    # We reuse the Stage 1 gauge: the face vectors haven't changed yet,
    # so gauge's vertex1/vertex2 are still valid intersection points.
    # ------------------------------------------------------------------
    new_comb, _, _ = comb_whitehead(face_i, face_j, poly_pre.to_comb())
    poly_flipped = GeomPolyhedron(
        face_vectors=poly_pre.face_vectors,
        adjacency=new_comb.adjacency,
        vertices=new_comb.vertices,
    )

    # flip_angles: keep pre-move angles for unchanged pairs,
    # drop old edge (face_i,face_j), add new edge (k,l) at 5°.
    flip_angles = pre_angles.copy()
    flip_angles[face_i, face_j] = flip_angles[face_j, face_i] = -10.0  # excluded (no longer adj)
    flip_angles[k, l] = flip_angles[l, k] = _NEAR_ZERO_ANGLE

    # Mini-homotopy to lock in the topology change.
    # The new edge (k,l) had ultraparallel face vectors (initial_angles[k,l] ≈ 0),
    # which interpolate_angles now handles (a1 >= 0 condition).
    # Reuse Stage 1 gauge (face vectors haven't changed, so vertex1/vertex2 are still valid).
    poly_post = run_homotopy(
        poly_flipped, flip_angles, gauge, h_init=1.0, h_max=1.0,
        cond_threshold=1e20,
    )
    logger.debug(
        "Whitehead (%d,%d): topology flip done, |f|=%.3e", face_i, face_j, poly_post.residual
    )

    # ------------------------------------------------------------------
    # Stage 3: canonical position (72° for all angles)   [optional]
    # ------------------------------------------------------------------
    if not canonicalize:
        return poly_post

    target_canonical = canonical_angle_matrix(poly_post.adjacency, _CANONICAL_ANGLE)
    gauge_canon = make_gauge_info(
        poly_post.face_vectors,
        poly_post.adjacency,
        poly_post.vertices,
        gauge_vertex_idx,
    )
    poly_final = run_homotopy(poly_post, target_canonical, gauge_canon, cond_threshold=1e20)
    logger.debug(
        "Whitehead (%d,%d): canonical done, |f|=%.3e", face_i, face_j, poly_final.residual
    )

    return poly_final
