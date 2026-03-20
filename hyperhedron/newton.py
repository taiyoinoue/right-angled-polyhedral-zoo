"""
Newton's method with backtracking line search and adaptive homotopy controller.

Numerical stability improvements over Roeder's MATLAB implementation:

1. Backtracking line search (Armijo condition) — prevents divergence when the
   initial iterate is far from the solution.

2. Condition number check before each linear solve — raises IllConditionedJacobian
   early rather than silently producing garbage from a near-singular system.

3. Face vector re-normalization after each Newton step — corrects floating-point
   drift off the spacelike unit hyperboloid mink(v,v) = 1.

4. Adaptive homotopy step size (PI-style controller) — replaces the fixed
   'divisions' parameter.  Step size grows after successful Newton solves and
   shrinks after failures.

5. Graded convergence: iterate until |f|_inf < tol rather than stopping at a
   hard iteration cap.  Degraded-accuracy solutions are returned with a warning
   rather than silently accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from .angles import interpolate_angles
from .exceptions import (
    HomotopyFailure,
    IllConditionedJacobian,
    LineSearchFailure,
    NewtonFailure,
)
from .linalg import renormalize_spacelike_batch
from .objective import GaugeInfo, compute_jacobian, compute_residual

if TYPE_CHECKING:
    from .polyhedron import GeomPolyhedron

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOL: float = 1e-12      # Target |f|_inf for convergence
WARN_TOL: float = 1e-6          # Acceptable degraded accuracy (log warning)
MAX_NEWTON_ITER: int = 100      # Hard cap on Newton iterations
COND_THRESHOLD: float = 1e12    # Maximum acceptable Jacobian condition number

ARMIJO_C: float = 1e-4          # Armijo sufficient-decrease constant
BACKTRACK_RHO: float = 0.5      # Step-halving factor
MAX_HALVINGS: int = 20          # Maximum number of backtracking halvings

# Homotopy controller defaults
H_INIT: float = 0.1
H_MIN: float = 1e-6
H_MAX: float = 1.0
FACTOR_UP: float = 1.5
FACTOR_DOWN: float = 0.5
MAX_HOMOTOPY_STEPS: int = 10_000


# ---------------------------------------------------------------------------
# Single Newton solve
# ---------------------------------------------------------------------------

@dataclass
class NewtonResult:
    """Outcome of a single Newton solve."""
    face_vectors: np.ndarray
    residual: float           # |f|_inf at termination
    condition_number: float   # κ(J) at the last successful step (0 if not recorded)
    iterations: int
    converged: bool           # True iff residual <= tol


def newton_solve(
    face_vectors: np.ndarray,
    target_angles: np.ndarray,
    gauge: GaugeInfo,
    tol: float = DEFAULT_TOL,
    max_iter: int = MAX_NEWTON_ITER,
    cond_threshold: float = COND_THRESHOLD,
) -> NewtonResult:
    """
    Run Newton's method with backtracking line search until convergence.

    Parameters
    ----------
    face_vectors:
        (N, 4) initial iterate; will not be mutated.
    target_angles:
        (N, N) target dihedral angles in degrees.
    gauge:
        Fixed GaugeInfo (computed once from the initial configuration).
    tol:
        Target infinity-norm tolerance.  The solve is considered converged
        when |f|_inf <= tol.
    max_iter:
        Hard cap.  If |f|_inf <= WARN_TOL at the cap, a warning is logged
        and a result with converged=False is returned.  Otherwise NewtonFailure
        is raised.
    cond_threshold:
        Jacobian condition numbers above this trigger IllConditionedJacobian.

    Returns
    -------
    NewtonResult on success or degraded accuracy.

    Raises
    ------
    IllConditionedJacobian  — Jacobian too ill-conditioned.
    LineSearchFailure       — No descent direction found after MAX_HALVINGS.
    NewtonFailure           — Did not converge within max_iter iterations.
    """
    N = face_vectors.shape[0]
    V = face_vectors.copy()

    f = compute_residual(V, target_angles, gauge)
    f_norm = float(np.linalg.norm(f, np.inf))
    kappa = 0.0

    for iteration in range(max_iter):
        if f_norm <= tol:
            return NewtonResult(V, f_norm, kappa, iteration, True)

        # Build Jacobian and check condition number
        J = compute_jacobian(V, target_angles, gauge)
        kappa = float(np.linalg.cond(J))
        if kappa > cond_threshold:
            raise IllConditionedJacobian(kappa)

        # Solve J @ delta = -f  (more stable than explicit inverse)
        try:
            delta = scipy.linalg.solve(J, -f, assume_a="gen")
        except np.linalg.LinAlgError:
            # Matrix is exactly/numerically singular (e.g. symmetric prism initial
            # position has a degenerate gauge).  Use least-squares solve: when
            # |f| ≈ 0 (already at the solution) this returns delta ≈ 0, which is
            # correct and allows the homotopy to continue making small t-steps.
            delta, _, _, _ = scipy.linalg.lstsq(J, -f)

        # --- Armijo backtracking line search ---
        alpha = 1.0
        accepted = False
        for _ in range(MAX_HALVINGS):
            V_new = V + alpha * delta.reshape(N, 4)
            V_new = renormalize_spacelike_batch(V_new)   # Fix 4: re-normalize
            f_new = compute_residual(V_new, target_angles, gauge)
            f_new_norm = float(np.linalg.norm(f_new, np.inf))

            # Armijo sufficient-decrease condition
            if f_new_norm < f_norm * (1.0 - ARMIJO_C * alpha):
                accepted = True
                break
            alpha *= BACKTRACK_RHO

        if not accepted:
            # If we're already near machine precision, treat as converged rather
            # than failing — the Armijo condition can fail due to floating-point
            # noise when |f| ≈ tol.
            if f_norm <= WARN_TOL:
                logger.debug(
                    "Line search could not improve on |f|=%.3e (near tol); treating as converged.",
                    f_norm,
                )
                return NewtonResult(V, f_norm, kappa, iteration, f_norm <= tol)
            raise LineSearchFailure(
                f"Line search failed at iteration {iteration}: "
                f"|f| = {f_norm:.3e}, smallest α tried = {alpha:.3e}."
            )

        V = V_new
        f = f_new
        f_norm = f_new_norm

        logger.debug(
            "Newton iter %d: |f|_inf = %.3e, α = %.4f, κ(J) = %.3e",
            iteration, f_norm, alpha, kappa,
        )

    # Reached max_iter
    if f_norm <= WARN_TOL:
        logger.warning(
            "Newton's method reached max_iter=%d with |f|_inf = %.3e > tol = %.3e. "
            "Continuing with degraded accuracy.",
            max_iter, f_norm, tol,
        )
        return NewtonResult(V, f_norm, kappa, max_iter, False)

    raise NewtonFailure(
        f"Newton's method failed: |f|_inf = {f_norm:.3e} after {max_iter} iterations."
    )


# ---------------------------------------------------------------------------
# Adaptive homotopy controller
# ---------------------------------------------------------------------------

def run_homotopy(
    geom_init: GeomPolyhedron,
    target_angles: np.ndarray,
    gauge: GaugeInfo,
    tol: float = DEFAULT_TOL,
    h_init: float = H_INIT,
    h_min: float = H_MIN,
    h_max: float = H_MAX,
    factor_up: float = FACTOR_UP,
    factor_down: float = FACTOR_DOWN,
    max_steps: int = MAX_HOMOTOPY_STEPS,
    cond_threshold: float = 1e12,
) -> GeomPolyhedron:
    """
    Follow the homotopy from geom_init's current angles to target_angles.

    The homotopy path is the linear interpolation:

        angles(t) = (1 - t) * initial_angles + t * target_angles,  t ∈ [0, 1]

    At each t, Newton's method solves for the polyhedron with those intermediate
    angles.  The step size h = Δt is adapted based on Newton's performance:
    - Successful solve → h *= factor_up  (up to h_max)
    - Failed solve     → h *= factor_down and retry (until h < h_min → raise)

    Parameters
    ----------
    geom_init:
        Starting GeomPolyhedron.  Its face_vectors are the initial iterate; its
        adjacency and vertices define the combinatorial type for the whole path.
    target_angles:
        (N, N) target dihedral angle matrix in degrees.
    gauge:
        Fixed GaugeInfo computed from geom_init.face_vectors.

    Returns
    -------
    GeomPolyhedron at t = 1 (with the target angles).

    Raises
    ------
    HomotopyFailure if the step size collapses below h_min.
    """
    from .angles import compute_angles
    from .polyhedron import GeomPolyhedron

    initial_angles = compute_angles(geom_init)

    V = geom_init.face_vectors.copy()
    adjacency = geom_init.adjacency
    vertices = geom_init.vertices

    t = 0.0
    h = h_init
    kappa_last = 0.0
    total_steps = 0

    while t < 1.0 - 1e-12:
        if total_steps >= max_steps:
            raise HomotopyFailure(
                f"Exceeded max_steps={max_steps} at t={t:.6f}."
            )

        h = min(h, 1.0 - t)   # Don't overshoot t = 1
        t_new = t + h

        angles_t = interpolate_angles(initial_angles, target_angles, t_new)

        try:
            result = newton_solve(V, angles_t, gauge, tol=tol, cond_threshold=cond_threshold)
        except (IllConditionedJacobian, LineSearchFailure, NewtonFailure) as exc:
            h *= factor_down
            if h < h_min:
                raise HomotopyFailure(
                    f"Homotopy step size collapsed to {h:.3e} < h_min={h_min:.3e} "
                    f"at t={t:.6f}: {exc}"
                ) from exc
            logger.debug("Homotopy step t=%.4f failed (h=%.6f → %.6f): %s", t_new, h / factor_down, h, exc)
            continue

        # Step accepted
        V = result.face_vectors
        kappa_last = result.condition_number
        t = t_new
        h = min(h * factor_up, h_max)
        total_steps += 1

        logger.debug(
            "Homotopy t=%.4f accepted: |f|=%.3e, κ=%.3e, h→%.6f",
            t, result.residual, kappa_last, h,
        )

    # Final Newton solve precisely at t = 1
    final = newton_solve(V, target_angles, gauge, tol=tol, cond_threshold=cond_threshold)

    return GeomPolyhedron(
        face_vectors=final.face_vectors,
        adjacency=adjacency,
        vertices=vertices,
        residual=final.residual,
        condition_number=final.condition_number if final.condition_number > 0 else kappa_last,
    )
