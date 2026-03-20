"""
Top-level polyhedron construction pipeline.

Ports make.m: given a combinatorial polyhedron description and a target
dihedral angle matrix, constructs the geometric realisation in H^3.

Usage
-----
    from hyperhedron.make import construct_polyhedron
    from hyperhedron.io.gxl import read_gxl
    from hyperhedron.angles import right_angle_matrix

    comb = read_gxl("example1.gxl")
    target = right_angle_matrix(comb.adjacency)
    geom = construct_polyhedron(comb, target)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .angles import canonical_angle_matrix, compute_angles, right_angle_matrix
from .combinatorial import comb_algorithm
from .newton import run_homotopy
from .objective import make_gauge_info
from .polyhedron import CombPolyhedron, GeomPolyhedron
from .primitive import build_primitive
from .whitehead import geometric_whitehead

logger = logging.getLogger(__name__)


def construct_polyhedron(
    comb: CombPolyhedron,
    target_angles: np.ndarray | None = None,
    *,
    output_dir: str | Path | None = None,
    tol: float = 1e-10,
) -> GeomPolyhedron:
    """
    Construct the hyperbolic polyhedron with the given combinatorial type.

    Parameters
    ----------
    comb:
        Combinatorial description (adjacency + vertex-face incidence).
    target_angles:
        (N,N) target dihedral angles in degrees.  Defaults to right-angles (90°).
    output_dir:
        If given, write intermediate OFF files poly1.off, poly2.off, ... here.
    tol:
        Newton convergence tolerance.

    Returns
    -------
    GeomPolyhedron at the target angles.
    """
    # target_angles will be recomputed from current.adjacency after Whitehead moves,
    # because the face labels are permuted by the combinatorial algorithm.
    # Only use comb.adjacency if the user supplied an explicit target.
    user_supplied_target = target_angles is not None
    if not user_supplied_target:
        # Placeholder — will be replaced after Whitehead reconstruction (Step 4).
        target_angles = None

    output_dir = Path(output_dir) if output_dir else None

    # --- Step 1: Combinatorial algorithm ---
    logger.info("Running combinatorial algorithm on %d-face polyhedron.", comb.num_faces)
    result = comb_algorithm(comb)
    n_moves = len(result.inverse_moves)
    logger.info("%d Whitehead moves required.", n_moves)

    # --- Step 2: Build primitive ---
    logger.info("Building primitive element (%d faces).", comb.num_faces)
    current = build_primitive(comb.num_faces)

    # --- Step 3: Apply inverse Whitehead moves in reverse order ---
    for step, move in enumerate(reversed(result.inverse_moves)):
        fi, fj = move.face_i, move.face_j
        logger.info(
            "Applying inverse Whitehead move %d/%d: edge (%d, %d).",
            step + 1, n_moves, fi, fj,
        )
        current = geometric_whitehead(fi, fj, current)

        if output_dir is not None:
            from .io.off import write_off
            out_path = output_dir / f"poly{step + 1}.off"
            write_off(current, out_path)

    # --- Step 4: Deform from 72° canonical to target angles ---
    # If no user target was supplied, build a right-angle target from the
    # CURRENT adjacency (post-Whitehead), since the face labels may be permuted.
    if not user_supplied_target:
        target_angles = right_angle_matrix(current.adjacency)

    logger.info("Deforming to target angles.")
    gauge = make_gauge_info(
        current.face_vectors, current.adjacency, current.vertices, 0
    )
    final = run_homotopy(current, target_angles, gauge, tol=tol, cond_threshold=1e20)
    logger.info(
        "Construction complete: |f|=%.3e, κ=%.3e.",
        final.residual, final.condition_number,
    )
    return final
