"""
Core data structures for hyperbolic polyhedra.

A polyhedron is described at two levels:
- CombPolyhedron: purely combinatorial (adjacency, vertex-face incidence).
- GeomPolyhedron: adds the geometric realization (face normal vectors in H^3).

All arrays use 0-based indexing throughout (unlike the original MATLAB code).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CombPolyhedron:
    """
    Combinatorial description of a simple hyperbolic polyhedron.

    adjacency: (N, N) int array.
        adjacency[i, j] = 1 if faces i and j share an edge, or if i == j.
        (The diagonal is 1 by convention, matching Roeder's enter_combinatorics.)

    vertices: (M, 3) int array.
        vertices[k] = (i, j, l) are the 0-indexed face indices of the three
        faces meeting at vertex k.  For a simple polyhedron every vertex has
        degree exactly 3.

    For a valid simple polyhedron: V = M, E = (sum of adjacency - N) / 2,
    F = N, and Euler's formula V - E + F = 2 must hold.
    """

    adjacency: np.ndarray   # (N, N) int
    vertices: np.ndarray    # (M, 3) int

    # ------------------------------------------------------------------
    # Derived counts
    # ------------------------------------------------------------------

    @property
    def num_faces(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_edges(self) -> int:
        # Off-diagonal 1s, each edge counted twice
        return int((np.sum(self.adjacency) - self.num_faces) // 2)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Check Euler's formula and structural constraints.
        Raises ValueError with a descriptive message on failure.
        """
        N = self.num_faces
        M = self.num_vertices
        E = self.num_edges

        if N < 4:
            raise ValueError(f"A polyhedron must have at least 4 faces, got {N}.")

        if self.adjacency.shape != (N, N):
            raise ValueError("adjacency must be square.")

        if not np.array_equal(self.adjacency, self.adjacency.T):
            raise ValueError("adjacency must be symmetric.")

        if not np.all(np.diag(self.adjacency) == 1):
            raise ValueError("adjacency diagonal must be all 1s.")

        euler = M - E + N
        if euler != 2:
            raise ValueError(
                f"Euler formula violated: V({M}) - E({E}) + F({N}) = {euler} ≠ 2."
            )

        # Each vertex must reference 3 distinct faces, all in range
        for k, triple in enumerate(self.vertices):
            if len(set(triple)) != 3:
                raise ValueError(f"Vertex {k} has repeated face indices: {triple}.")
            for fi in triple:
                if fi < 0 or fi >= N:
                    raise ValueError(
                        f"Vertex {k} references face index {fi} out of range [0, {N})."
                    )
            # All pairs in the triple must be adjacent
            i, j, l = int(triple[0]), int(triple[1]), int(triple[2])
            for a, b in [(i, j), (i, l), (j, l)]:
                if self.adjacency[a, b] != 1:
                    raise ValueError(
                        f"Vertex {k} claims faces {a} and {b} meet, but they are not adjacent."
                    )


@dataclass
class GeomPolyhedron:
    """
    Geometric realization of a compact hyperbolic polyhedron.

    face_vectors: (N, 4) float array.
        Each row is a unit spacelike normal vector in the hyperboloid model:
            mink(face_vectors[i], face_vectors[i]) = +1  for all i.
        The dihedral angle between adjacent faces i and j is given by:
            theta_ij = 180 - (180/pi) * arccos(mink(face_vectors[i], face_vectors[j]))

    adjacency, vertices: same as CombPolyhedron.

    residual: infinity-norm of the Newton objective at convergence.
    condition_number: condition number of the Jacobian at the final Newton step.
        0.0 means it was not recorded (e.g., for analytically-constructed primitives).
    """

    face_vectors: np.ndarray   # (N, 4) float
    adjacency: np.ndarray      # (N, N) int
    vertices: np.ndarray       # (M, 3) int
    residual: float = 0.0
    condition_number: float = 0.0

    # ------------------------------------------------------------------
    # Derived counts (mirroring CombPolyhedron)
    # ------------------------------------------------------------------

    @property
    def num_faces(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_edges(self) -> int:
        return int((np.sum(self.adjacency) - self.num_faces) // 2)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_comb(self) -> CombPolyhedron:
        """Strip geometric data, returning only the combinatorial description."""
        return CombPolyhedron(
            adjacency=self.adjacency.copy(),
            vertices=self.vertices.copy(),
        )

    def replace_vectors(self, new_vectors: np.ndarray) -> GeomPolyhedron:
        """Return a new GeomPolyhedron with updated face vectors (immutable update)."""
        return GeomPolyhedron(
            face_vectors=new_vectors,
            adjacency=self.adjacency,
            vertices=self.vertices,
            residual=self.residual,
            condition_number=self.condition_number,
        )
