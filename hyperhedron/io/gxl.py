"""
GXL graph file reader.

GXL (Graph eXchange Language) is an XML format used by JGraphpad to
describe the 1-skeleton (edge graph) of a polyhedron.

Ports gxl_to_comb.m.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..exceptions import InvalidGraphError
from ..polyhedron import CombPolyhedron


def read_gxl(path: str | Path) -> CombPolyhedron:
    """
    Parse a JGraphpad GXL file and return a CombPolyhedron.

    The GXL file encodes the 1-skeleton as a graph: nodes are faces,
    edges are face-adjacency relations.  Node IDs are expected to be
    integers (or strings that parse as integers).
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    # Collect all edges — each <edge from="nodeX" to="nodeY">
    edges = []
    max_id = 0
    for elem in root.iter():
        if elem.tag.endswith("edge"):
            from_id = _parse_node_id(elem.get("from", ""))
            to_id   = _parse_node_id(elem.get("to", ""))
            edges.append((from_id, to_id))
            max_id = max(max_id, from_id, to_id)

    num_faces = max_id + 1   # 0-indexed
    return _build_comb(num_faces, edges)


def _parse_node_id(s: str) -> int:
    """Extract the integer node ID from strings like 'node5' or '5'."""
    s = s.strip().lower()
    for prefix in ("node", "n", "v"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return int(s)


def _build_comb(num_faces: int, edges: list[tuple[int, int]]) -> CombPolyhedron:
    """
    Build a CombPolyhedron from a face count and edge list (face-adjacency graph).

    Uses Euler's formula to determine the expected vertex count:
        V = 2 + E - F
    Then enumerates all triangles in the adjacency graph as vertices.
    """
    adjacency = np.eye(num_faces, dtype=int)
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1

    num_edges = len(edges)
    num_vertices = 2 + num_edges - num_faces    # Euler: V = 2 + E - F

    # Enumerate vertex triples: (i, j, k) with i<j<k all mutually adjacent
    vert_list = []
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            if adjacency[i, j] != 1:
                continue
            for k in range(j + 1, num_faces):
                if adjacency[j, k] == 1 and adjacency[i, k] == 1:
                    vert_list.append([i, j, k])

    if len(vert_list) != num_vertices:
        raise InvalidGraphError(
            f"Expected {num_vertices} vertices (Euler: 2 + {num_edges} - {num_faces}), "
            f"found {len(vert_list)} triangles in the adjacency graph. "
            "The input graph may not be a simple polyhedron skeleton."
        )

    vertices = np.array(vert_list, dtype=int)
    comb = CombPolyhedron(adjacency=adjacency, vertices=vertices)

    try:
        comb.validate()
    except ValueError as e:
        raise InvalidGraphError(str(e)) from e

    return comb
