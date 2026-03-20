"""
hyperhedron — construct and visualize compact hyperbolic polyhedra.

Ported from Roeder's MATLAB implementation (2006) with numerical stability
improvements.  See CLAUDE.md and the plan document for architecture details.
"""

from .polyhedron import CombPolyhedron, GeomPolyhedron
from .angles import compute_angles, right_angle_matrix
from .linalg import mink, solve_for_vertex, to_klein, to_poincare
from .objective import GaugeInfo, make_gauge_info
from .newton import newton_solve, run_homotopy

__all__ = [
    "CombPolyhedron",
    "GeomPolyhedron",
    "compute_angles",
    "right_angle_matrix",
    "mink",
    "solve_for_vertex",
    "to_klein",
    "to_poincare",
    "GaugeInfo",
    "make_gauge_info",
    "newton_solve",
    "run_homotopy",
]
