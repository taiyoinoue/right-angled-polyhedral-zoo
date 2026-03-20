"""Typed exceptions for hyperhedron numerical operations."""

from __future__ import annotations

from typing import Optional, Tuple


class HyperhedronError(Exception):
    """Base exception for all hyperhedron errors."""


class NumericalError(HyperhedronError):
    """Raised when a numerical operation fails (e.g., normalizing a null vector)."""


class IdealVertexError(HyperhedronError):
    """
    Raised when a vertex is ideal (at infinity in hyperbolic space).

    This occurs when three face normal vectors are nearly coplanar in Minkowski
    space, so their intersection point lies on the boundary of H^3 rather than
    in the interior.
    """

    def __init__(self, face_triple: Optional[Tuple[int, int, int]] = None, mink_norm: Optional[float] = None):
        self.face_triple = face_triple
        self.mink_norm = mink_norm
        msg = "Vertex is ideal (at infinity)"
        if face_triple is not None:
            msg += f" for face triple {face_triple}"
        if mink_norm is not None:
            msg += f": mink(v,v) = {mink_norm:.6e} (expected < 0)"
        super().__init__(msg)


class IllConditionedJacobian(HyperhedronError):
    """Raised when the Newton Jacobian is too ill-conditioned to solve reliably."""

    def __init__(self, kappa: float):
        self.kappa = kappa
        super().__init__(f"Jacobian condition number {kappa:.3e} exceeds threshold.")


class LineSearchFailure(HyperhedronError):
    """Raised when the Armijo backtracking line search finds no descent step."""


class NewtonFailure(HyperhedronError):
    """Raised when Newton's method fails to converge to the requested tolerance."""


class HomotopyFailure(HyperhedronError):
    """Raised when the homotopy path cannot be followed (step size collapses)."""


class CycleOrderingError(HyperhedronError):
    """Raised when face cycle ordering is topologically inconsistent."""


class InvalidGraphError(HyperhedronError):
    """Raised when the input graph is not a valid simple polyhedron skeleton."""
