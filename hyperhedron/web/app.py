"""
FastAPI backend for the Hyperhedron Zoo.

Serves the web interface and computes/caches polyhedron data as JSON
(Poincaré ball vertex coordinates, face topology, and edge lists).

Usage
-----
    hyperhedron-web [--gxl-dir DIR] [--cache-dir DIR] [--host HOST] [--port PORT]

Environment variables
---------------------
    HYPERHEDRON_GXL_DIR   directory containing *.gxl files  (default: ~/hyperhedron)
    HYPERHEDRON_CACHE_DIR directory for cached JSON results (default: /tmp/hyperhedron_web)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent
_STATIC_DIR = _THIS_DIR / "static"

_GXL_DIR = Path(os.environ.get("HYPERHEDRON_GXL_DIR", Path.home() / "hyperhedron"))
_CACHE_DIR = Path(os.environ.get("HYPERHEDRON_CACHE_DIR", "/tmp/hyperhedron_web"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Hyperhedron Zoo", version="1.0.0")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers: geometry → JSON
# ---------------------------------------------------------------------------

def _klein_to_poincare(K: np.ndarray) -> np.ndarray:
    """K = (k1,k2,k3) in Klein model → Poincaré ball coordinates."""
    norm_sq = float(np.dot(K, K))
    denom = 1.0 + np.sqrt(max(1.0 - norm_sq, 0.0))
    return K / denom


def _geom_to_dict(geom, name: str) -> dict:
    from hyperhedron.io.off import (
        compute_vertex_positions,
        center_vertex_positions,
        _ordered_vertices_for_face,
    )

    hyperboloid = compute_vertex_positions(geom)
    centred = center_vertex_positions(hyperboloid)

    M = geom.num_vertices
    N = geom.num_faces

    # Poincaré ball vertex positions
    poincare: list[list[float]] = []
    for v in range(M):
        h = centred[v]
        K = h[1:] / h[0]            # Klein coordinates
        p = _klein_to_poincare(K)
        poincare.append(p.tolist())

    # Ordered vertex lists per face (for rendering)
    faces: list[list[int]] = []
    for f in range(N):
        vorder = _ordered_vertices_for_face(f, geom)
        faces.append(vorder)

    # 3D edges: pairs of vertices that share exactly 2 faces
    verts = geom.vertices
    edges: list[list[int]] = []
    for vi in range(M):
        for vj in range(vi + 1, M):
            si = {int(verts[vi, k]) for k in range(3)}
            sj = {int(verts[vj, k]) for k in range(3)}
            if len(si & sj) == 2:
                edges.append([vi, vj])

    # Face-adjacency edges (face graph), for reference
    adj_edges: list[list[int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            if geom.adjacency[i, j] == 1:
                adj_edges.append([i, j])

    return {
        "name": name,
        "num_faces": N,
        "num_vertices": M,
        "num_edges": len(edges),
        "vertices": poincare,
        "faces": faces,
        "edges": edges,
    }


def _compute_and_cache(name: str) -> dict:
    """Construct the polyhedron and write the result to cache. Returns the dict."""
    from hyperhedron.io.gxl import read_gxl
    from hyperhedron.make import construct_polyhedron

    gxl = _GXL_DIR / f"{name}.gxl"
    if not gxl.exists():
        raise FileNotFoundError(f"No GXL file for '{name}' in {_GXL_DIR}")

    logger.info("Computing %s …", name)
    comb = read_gxl(gxl)
    geom = construct_polyhedron(comb)
    data = _geom_to_dict(geom, name)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{name}.json"
    cache.write_text(json.dumps(data))
    logger.info("Cached %s → %s", name, cache)
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/polyhedra")
def list_polyhedra():
    """Return a list of all available polyhedra with their computation status."""
    if not _GXL_DIR.exists():
        return []
    gxl_files = sorted(_GXL_DIR.glob("*.gxl"))
    result = []
    for f in gxl_files:
        name = f.stem
        cache = _CACHE_DIR / f"{name}.json"
        result.append({"name": name, "computed": cache.exists()})
    return result


@app.get("/api/polyhedra/{name}")
def get_polyhedron(name: str):
    """
    Return polyhedron data as JSON (Poincaré ball vertex coords + topology).

    Computes and caches on first request; subsequent requests are instant.
    """
    cache = _CACHE_DIR / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    try:
        return _compute_and_cache(name)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to construct %s", name)
        raise HTTPException(500, detail=str(e))


@app.delete("/api/polyhedra/{name}/cache")
def delete_cache(name: str):
    """Delete the cached result for a polyhedron (forces recomputation)."""
    cache = _CACHE_DIR / f"{name}.json"
    if cache.exists():
        cache.unlink()
        return {"deleted": True}
    return {"deleted": False}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Hyperhedron Zoo web server")
    parser.add_argument("--gxl-dir", default=str(_GXL_DIR),
                        help="Directory containing *.gxl files")
    parser.add_argument("--cache-dir", default=str(_CACHE_DIR),
                        help="Directory for cached JSON results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload (development mode)")
    args = parser.parse_args()

    # Override module-level paths from CLI args (import module to mutate its namespace)
    import hyperhedron.web.app as _mod
    _mod._GXL_DIR = Path(args.gxl_dir)
    _mod._CACHE_DIR = Path(args.cache_dir)

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "hyperhedron.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
