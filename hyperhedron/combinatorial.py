"""
Combinatorial algorithm for hyperbolic polyhedra.

Ports comb_algorithm.m, cond_cycle.m, final_comb.m, find_v_inf.m,
interior_to_cycle.m, comb_whitehead.m, and permutation.m.

All face/vertex indices are 0-based throughout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .exceptions import CycleOrderingError, InvalidGraphError
from .polyhedron import CombPolyhedron

logger = logging.getLogger(__name__)


@dataclass
class WhiteheadMove:
    face_i: int
    face_j: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.face_i, self.face_j)


@dataclass
class CombResult:
    """Output of comb_algorithm."""
    forward_moves: list[WhiteheadMove]   # moves that reduce poly → primitive
    inverse_moves: list[WhiteheadMove]   # their inverses (apply to primitive to rebuild)
    permutation: np.ndarray              # P[i] = primitive face index for reduced face i


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _cyc(j: int, n: int) -> int:
    """Wrap index j into [0, n) — equivalent to MATLAB's fix_index converted to 0-based."""
    return j % n


def is_3prismatic(comb: CombPolyhedron) -> bool:
    """Return True if the polyhedron has a prismatic 3-circuit (invalid for Andreev)."""
    A = comb.adjacency.astype(float)
    A3 = A @ A @ A
    for i in range(comb.num_faces):
        deg = int(np.sum(A[i])) - 1   # subtract self-loop
        if A3[i, i] - 5 * deg - 1 > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# find_v_inf
# ---------------------------------------------------------------------------

def find_v_inf(comb: CombPolyhedron) -> tuple[int, list[int], list[int]]:
    """
    Find v_inf (highest-degree face), its ordered cycle, and interior faces.

    Returns (v_inf, cycle, interior), all 0-indexed.
    The cycle is ordered cyclically using the face-adjacency graph.
    """
    A = comb.adjacency
    N = comb.num_faces

    degrees = np.sum(A, axis=1) - 1          # exclude self-loop
    v_inf = int(np.argmax(degrees))

    unordered = [j for j in range(N) if A[v_inf, j] == 1 and j != v_inf]
    if len(unordered) < 3:
        raise CycleOrderingError(
            f"v_inf={v_inf} has only {len(unordered)} adjacent faces; need ≥ 3."
        )

    # The faces adjacent to v_inf form a simple cycle in the dual graph.
    G = nx.Graph()
    G.add_nodes_from(unordered)
    unordered_set = set(unordered)
    for i in unordered:
        for j in unordered:
            if i < j and A[i, j] == 1:
                G.add_edge(i, j)

    for node in G.nodes():
        if G.degree(node) != 2:
            raise CycleOrderingError(
                f"Face {node} has degree {G.degree(node)} in the cycle subgraph "
                f"(expected 2). The polyhedron may not be simple."
            )

    # Walk the cycle
    start = unordered[0]
    cycle = [start]
    visited = {start}
    current = start
    while len(cycle) < len(unordered):
        moved = False
        for nbr in G.neighbors(current):
            if nbr not in visited:
                cycle.append(nbr)
                visited.add(nbr)
                current = nbr
                moved = True
                break
        if not moved:
            break

    inf_and_cycle = set(cycle) | {v_inf}
    interior = [f for f in range(N) if f not in inf_and_cycle]

    return v_inf, cycle, interior


# ---------------------------------------------------------------------------
# interior_to_cycle
# ---------------------------------------------------------------------------

def interior_to_cycle(
    comb: CombPolyhedron, cycle: list[int], interior: list[int]
) -> np.ndarray:
    """M[j, i] = 1 if cycle[j] is adjacent to interior[i]."""
    A = comb.adjacency
    M = np.zeros((len(cycle), len(interior)), dtype=int)
    for j, cf in enumerate(cycle):
        for i, inf in enumerate(interior):
            M[j, i] = int(A[cf, inf])
    return M


# ---------------------------------------------------------------------------
# comb_whitehead
# ---------------------------------------------------------------------------

def comb_whitehead(
    face_i: int, face_j: int, comb: CombPolyhedron
) -> tuple[CombPolyhedron, int, int]:
    """
    Combinatorial Whitehead move on edge (face_i, face_j).

    Finds the two vertices v1=(face_i, face_j, k) and v2=(face_i, face_j, l),
    then replaces edge (i,j) with edge (k,l).

    Returns (new_comb, k, l).
    """
    A = comb.adjacency.copy()
    verts = comb.vertices.copy()
    M = comb.num_vertices

    if A[face_i, face_j] != 1:
        raise InvalidGraphError(
            f"No edge ({face_i}, {face_j}); cannot perform Whitehead move."
        )

    k = l = first_v = second_v = None
    for v in range(M):
        fset = set(int(x) for x in verts[v])
        if face_i in fset and face_j in fset:
            other = list(fset - {face_i, face_j})[0]
            if first_v is None:
                k, first_v = other, v
            else:
                l, second_v = other, v
                break

    if first_v is None or second_v is None:
        raise InvalidGraphError(
            f"Could not find two vertices for edge ({face_i}, {face_j})."
        )

    verts_new = verts.copy()
    verts_new[first_v]  = sorted([face_i, k, l])
    verts_new[second_v] = sorted([face_j, k, l])

    A_new = A.copy()
    A_new[face_i, face_j] = A_new[face_j, face_i] = 0
    A_new[k, l] = A_new[l, k] = 1

    new_comb = CombPolyhedron(adjacency=A_new, vertices=verts_new)
    if is_3prismatic(new_comb):
        logger.warning(
            "Whitehead move (%d,%d)→(%d,%d) produced a 3-prismatic polyhedron.",
            face_i, face_j, k, l,
        )
    return new_comb, k, l


# ---------------------------------------------------------------------------
# Arc helpers used by cond_cycle
# ---------------------------------------------------------------------------

def _find_arcs(M_col: np.ndarray) -> list[tuple[int, int]]:
    """
    Find arcs (consecutive 1-runs) in a *cyclic* binary vector.

    Returns list of (start_idx, length), handling wrap-around correctly.
    """
    n = len(M_col)
    total = int(np.sum(M_col))
    if total == 0:
        return []
    if total == n:
        return [(0, n)]

    # Work on doubled array; each arc appears at most twice — keep first occurrence.
    M2 = np.concatenate([M_col, M_col])
    arcs = []
    seen_starts: set[int] = set()
    j = 0
    while j < 2 * n:
        if M2[j] == 1:
            prev = (j - 1) % (2 * n)
            if M2[prev] == 0:                    # arc start
                real_start = j % n
                if real_start not in seen_starts:
                    length = 0
                    k = j
                    while k < 2 * n and M2[k] == 1 and length < n:
                        length += 1
                        k += 1
                    arcs.append((real_start, length))
                    seen_starts.add(real_start)
        j += 1
    return arcs


def _max_arc(arcs: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (start, length) of the longest arc; (0, 0) if none."""
    if not arcs:
        return 0, 0
    return max(arcs, key=lambda a: a[1])


# ---------------------------------------------------------------------------
# cond_cycle  (main reduction step)
# ---------------------------------------------------------------------------

def cond_cycle(
    comb: CombPolyhedron,
    M: np.ndarray,
    cycle: list[int],
    interior: list[int],
    v_inf: int,                    # kept for API compatibility; not used internally
) -> tuple[CombPolyhedron, list[WhiteheadMove], list[WhiteheadMove]]:
    """
    One reduction step of the combinatorial algorithm.

    Performs Whitehead moves to increase the cycle length by 1,
    reducing the number of interior vertices toward 2.

    Returns (new_comb, forward_moves, inverse_moves).
    """
    comb_t = comb
    fwd: list[WhiteheadMove] = []
    inv: list[WhiteheadMove] = []

    n_int = len(interior)
    n_cyc = len(cycle)

    # Interior-interior adjacency matrix
    M_int = np.zeros((n_int, n_int), dtype=int)
    for i in range(n_int):
        for j in range(n_int):
            M_int[i, j] = int(comb.adjacency[interior[i], interior[j]])

    # int_class[i] = #interior neighbors (excl. self)
    int_class = np.array([int(np.sum(M_int[i])) - 1 for i in range(n_int)])
    # num_connected[i] = #cycle neighbors
    num_connected = np.array([int(np.sum(M[:, i])) for i in range(n_int)])

    # Arc info for each interior vertex
    all_arcs = [_find_arcs(M[:, i]) for i in range(n_int)]
    N_arcs  = np.array([len(a) for a in all_arcs])
    max_arcs = [_max_arc(a) for a in all_arcs]   # (start_idx, length) in cycle

    # --- Case selection ---
    interior_preferred = -1
    best_connected = n_cyc + 1
    for i in range(n_int):
        if int_class[i] > 1 and max_arcs[i][1] > 1:
            if num_connected[i] < best_connected:
                best_connected = num_connected[i]
                interior_preferred = i

    if interior_preferred >= 0:
        whichcase = 2
    else:
        end_point = end_point_preferred = -1
        for i in range(n_int):
            if int_class[i] == 1:
                end_point = i
                if max_arcs[i][1] > 3:
                    end_point_preferred = i
        whichcase = 1 if end_point_preferred >= 0 else 3

    def _do_move(fi, fj):
        nonlocal comb_t
        new_comb, k, l = comb_whitehead(fi, fj, comb_t)
        comb_t = new_comb
        fwd.append(WhiteheadMove(fi, fj))
        inv.append(WhiteheadMove(k, l))
        return interior_to_cycle(comb_t, cycle, interior)

    # -----------------------------------------------------------------------
    if whichcase == 1:
        logger.debug("cond_cycle: Case 1")
        ep = end_point_preferred
        arc_start, arc_len = max_arcs[ep]
        outer1_idx = arc_start
        outer2_idx = _cyc(arc_start + arc_len - 1, n_cyc)
        outer1 = cycle[outer1_idx]
        outer2 = cycle[outer2_idx]

        # Find the other interior vertex connected to ep
        inner_idx = next(
            i for i in range(n_int) if M_int[i, ep] == 1 and i != ep
        )
        inner = interior[inner_idx]

        M = _do_move(outer1, interior[ep])

        # Eliminate extra connections from cycle to inner
        i_idx = _cyc(outer1_idx + 2, n_cyc)
        while i_idx != outer1_idx:
            if M[i_idx, inner_idx] == 1:
                M = _do_move(cycle[i_idx], inner)
            i_idx = _cyc(i_idx + 1, n_cyc)

        # Increase cycle length
        next_idx = _cyc(outer1_idx + 1, n_cyc)
        M = _do_move(outer1, cycle[next_idx])

    # -----------------------------------------------------------------------
    elif whichcase == 2:
        logger.debug("cond_cycle: Case 2")
        ip = interior_preferred
        arc_start, arc_len = max_arcs[ip]

        # Eliminate connections outside the max arc
        i_start = _cyc(arc_start + arc_len, n_cyc)
        i_end   = arc_start
        i_idx = i_start
        while i_idx != i_end:
            if M[i_idx, ip] == 1:
                M = _do_move(cycle[i_idx], interior[ip])
            i_idx = _cyc(i_idx + 1, n_cyc)

        # Trim max arc down to exactly 2 connections
        M = interior_to_cycle(comb_t, cycle, interior)
        cur_arcs = _find_arcs(M[:, ip])
        cur_start, cur_len = _max_arc(cur_arcs)
        num_remove = cur_len - 2
        for step in range(num_remove):
            idx = _cyc(cur_start + step, n_cyc)
            M = _do_move(cycle[idx], interior[ip])

        # Increase cycle length between the 2 remaining connections
        M = interior_to_cycle(comb_t, cycle, interior)
        cur_arcs = _find_arcs(M[:, ip])
        cur_start, _ = _max_arc(cur_arcs)
        c1_idx = cur_start
        c2_idx = _cyc(cur_start + 1, n_cyc)
        M = _do_move(cycle[c1_idx], cycle[c2_idx])

    # -----------------------------------------------------------------------
    else:
        logger.debug("cond_cycle: Case 3")
        ep = end_point

        # Walk the interior chain from ep to the first junction
        line_idx = [ep]
        while int_class[line_idx[-1]] < 3:
            prev = line_idx[-2] if len(line_idx) >= 2 else -1
            nxt = next(
                (i for i in range(n_int)
                 if M_int[i, line_idx[-1]] == 1 and i != line_idx[-1] and i != prev),
                None,
            )
            if nxt is None:
                break
            line_idx.append(nxt)

        arc_start, arc_len = max_arcs[ep]
        outer1_idx = arc_start
        outer2_idx = _cyc(arc_start + arc_len - 1, n_cyc)
        outer1 = cycle[outer1_idx]
        outer2 = cycle[outer2_idx]
        tip = line_idx[-1]
        tip_face = interior[tip]

        # Eliminate connections from tip to cycle except outer1 and outer2
        for c_idx in range(n_cyc):
            if M[c_idx, tip] == 1 and c_idx != outer1_idx and c_idx != outer2_idx:
                M = _do_move(tip_face, cycle[c_idx])

        # Move outer2 connection so tip is connected to 1 cyclic component only
        M = _do_move(tip_face, outer2)

        # Open up: walk the chain in reverse, each time moving outer1
        for step in range(len(line_idx) - 2, -1, -1):
            M = _do_move(interior[line_idx[step]], outer1)

        # Increase cycle length
        next_idx = _cyc(outer1_idx + 1, n_cyc)
        M = _do_move(outer1, cycle[next_idx])

    return comb_t, fwd, inv


# ---------------------------------------------------------------------------
# final_comb
# ---------------------------------------------------------------------------

def final_comb(
    comb: CombPolyhedron,
    M: np.ndarray,
    cycle: list[int],
    interior: list[int],
    v_inf: int,
) -> tuple[CombPolyhedron, list[WhiteheadMove], list[WhiteheadMove]]:
    """
    Final adjustment: ensure interior[1] is connected to exactly 3 consecutive
    cycle vertices.  Requires exactly 2 interior faces; no-op otherwise.
    """
    if len(interior) < 2:
        return comb, [], []

    comb_t = comb
    fwd: list[WhiteheadMove] = []
    inv: list[WhiteheadMove] = []
    n_cyc = len(cycle)

    def _do_move(fi, fj):
        nonlocal comb_t
        new_comb, k, l = comb_whitehead(fi, fj, comb_t)
        comb_t = new_comb
        fwd.append(WhiteheadMove(fi, fj))
        inv.append(WhiteheadMove(k, l))
        return interior_to_cycle(comb_t, cycle, interior)

    # Find where interior[1]'s connections start in the cycle
    col1 = M[:, 1]
    # Find first position where col1 transitions to 1 from a 0
    i = 0
    while i < n_cyc and not (col1[i] == 1 and col1[_cyc(i - 1, n_cyc)] == 0):
        i += 1

    num = int(np.sum(col1))
    j = _cyc(i + num - 1, n_cyc)
    end_cond = _cyc(i + 2, n_cyc)

    while j != end_cond:
        M = _do_move(cycle[j], interior[1])
        j = _cyc(j - 1, n_cyc)

    return comb_t, fwd, inv


# ---------------------------------------------------------------------------
# permutation
# ---------------------------------------------------------------------------

def _compute_permutation(
    comb: CombPolyhedron, v_inf: int, cycle: list[int], interior: list[int]
) -> np.ndarray:
    """
    Compute the permutation P mapping reduced-poly face indices to primitive face indices.

    Returns P as a numpy array of length N where P[i] is the 0-indexed
    primitive face that corresponds to face i of the reduced polyhedron.
    """
    A = comb.adjacency
    N = comb.num_faces
    P = np.full(N, -1, dtype=int)

    n_cyc = len(cycle)
    M = interior_to_cycle(comb, cycle, interior)

    P[v_inf] = 0   # v_inf → primitive face 0

    if len(interior) < 2:
        # Degenerate case: assign remaining faces sequentially
        for idx, f in enumerate(interior):
            P[f] = idx + 1
        for idx, f in enumerate(cycle):
            P[f] = len(interior) + 1 + idx
        return P

    # Which interior vertex has more cycle connections?
    c0 = int(np.sum(M[:, 0]))
    c1 = int(np.sum(M[:, 1]))
    if c0 > c1:
        int1, int2 = interior[0], interior[1]
        int1_col, int2_col = 0, 1
    else:
        int1, int2 = interior[1], interior[0]
        int1_col, int2_col = 1, 0

    P[int1] = 1   # most-connected interior → primitive face 1
    P[int2] = 2   # other interior → primitive face 2

    # Find the cycle vertex NOT adjacent to int1
    special = next(
        ci for ci in range(n_cyc) if A[cycle[ci], int1] == 0
    )

    # Cycle vertices from special+1 to end → primitive faces 3, 4, ...
    for idx in range(special + 1, n_cyc):
        P[cycle[idx]] = idx - special + 2   # 3-based for non-special vertices

    # Cycle vertices from 0 to special → primitive faces N-special-1 ... N-1
    for idx in range(special + 1):
        P[cycle[idx]] = N - special - 1 + idx

    return P


# ---------------------------------------------------------------------------
# comb_algorithm  (top-level)
# ---------------------------------------------------------------------------

def comb_algorithm(comb: CombPolyhedron) -> CombResult:
    """
    Run the combinatorial algorithm on a polyhedron.

    Finds a sequence of Whitehead moves reducing the polyhedron to the
    primitive double-prism form.  Returns the forward moves (reduce) and
    inverse moves (reconstruct from primitive), plus the face-index permutation.
    """
    comb_t = comb
    all_fwd: list[WhiteheadMove] = []
    all_inv: list[WhiteheadMove] = []

    v_inf, cycle, interior = find_v_inf(comb_t)
    M = interior_to_cycle(comb_t, cycle, interior)
    logger.info("Initial: %d interior vertices, cycle length %d", len(interior), len(cycle))

    max_iterations = 10 * comb.num_faces   # safety limit
    itr = 0
    while len(interior) > 2 and itr < max_iterations:
        comb_t, fwd, inv = cond_cycle(comb_t, M, cycle, interior, v_inf)
        all_fwd.extend(fwd)
        all_inv.extend(inv)

        v_inf, cycle, interior = find_v_inf(comb_t)
        M = interior_to_cycle(comb_t, cycle, interior)
        logger.debug("After reduction: %d interior, cycle %d", len(interior), len(cycle))
        itr += 1

    if len(interior) > 2:
        raise InvalidGraphError(
            f"Combinatorial algorithm did not converge after {max_iterations} iterations."
        )

    # Final adjustment
    comb_t, fwd, inv = final_comb(comb_t, M, cycle, interior, v_inf)
    all_fwd.extend(fwd)
    all_inv.extend(inv)

    # Recompute after final_comb
    v_inf, cycle, interior = find_v_inf(comb_t)

    # Compute permutation and re-index all moves
    P = _compute_permutation(comb_t, v_inf, cycle, interior)

    def _remap(move: WhiteheadMove) -> WhiteheadMove:
        return WhiteheadMove(int(P[move.face_i]), int(P[move.face_j]))

    remapped_fwd = [_remap(m) for m in all_fwd]
    remapped_inv = [_remap(m) for m in all_inv]

    logger.info(
        "Combinatorial algorithm: %d Whitehead moves required.", len(all_fwd)
    )

    return CombResult(
        forward_moves=remapped_fwd,
        inverse_moves=remapped_inv,
        permutation=P,
    )
