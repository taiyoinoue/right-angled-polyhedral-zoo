"""
Batch pipeline for processing a database of polyhedra.

Each input is a GXL file encoding the 1-skeleton of a right-angled
hyperbolic polyhedron.  Outputs are Geomview OFF files suitable for
3D visualization.

Usage (programmatic)
--------------------
    from hyperhedron.batch import run_batch
    results = run_batch(gxl_dir="data/graphs/", out_dir="data/off/", workers=4)

Usage (CLI)
-----------
    python -m hyperhedron.batch data/graphs/ data/off/ --workers 4
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolyResult:
    """Outcome of processing one GXL file."""
    name: str               # stem of the input file
    gxl_path: Path
    off_path: Optional[Path] = None
    success: bool = False
    error: str = ""
    elapsed: float = 0.0    # seconds


# ---------------------------------------------------------------------------
# Per-process worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_one(gxl_path: Path, out_dir: Path, tol: float) -> PolyResult:
    """
    Worker function: read one GXL file, construct the polyhedron, write OFF.

    Runs in a subprocess — imports are local to avoid pickling the whole module.
    """
    import traceback
    from hyperhedron.io.gxl import read_gxl
    from hyperhedron.make import construct_polyhedron
    from hyperhedron.io.off import write_off

    name = gxl_path.stem
    off_path = out_dir / f"{name}.off"
    t0 = time.perf_counter()

    try:
        comb = read_gxl(gxl_path)
        geom = construct_polyhedron(comb, tol=tol)
        write_off(geom, off_path)
        elapsed = time.perf_counter() - t0
        return PolyResult(
            name=name,
            gxl_path=gxl_path,
            off_path=off_path,
            success=True,
            elapsed=elapsed,
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return PolyResult(
            name=name,
            gxl_path=gxl_path,
            success=False,
            error=traceback.format_exc(),
            elapsed=elapsed,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BatchSummary:
    results: List[PolyResult] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def failed(self) -> List[PolyResult]:
        return [r for r in self.results if not r.success]

    def print_summary(self) -> None:
        print(f"\nBatch complete: {self.n_success} succeeded, {self.n_failed} failed "
              f"out of {len(self.results)} total.")
        if self.failed:
            print("\nFailed:")
            for r in self.failed:
                print(f"  {r.name}: {r.error.splitlines()[-1]}")


def run_batch(
    gxl_dir: str | Path,
    out_dir: str | Path,
    *,
    workers: int = 4,
    tol: float = 1e-10,
    glob: str = "*.gxl",
) -> BatchSummary:
    """
    Process all GXL files in gxl_dir, writing OFF files to out_dir.

    Parameters
    ----------
    gxl_dir:  Directory containing .gxl input files.
    out_dir:  Directory for .off output files (created if absent).
    workers:  Number of parallel worker processes.
    tol:      Newton convergence tolerance passed to construct_polyhedron.
    glob:     Glob pattern for input files (default "*.gxl").

    Returns
    -------
    BatchSummary with per-file PolyResult objects.
    """
    gxl_dir = Path(gxl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gxl_files = sorted(gxl_dir.glob(glob))
    if not gxl_files:
        logger.warning("No files matching %r found in %s.", glob, gxl_dir)
        return BatchSummary()

    logger.info(
        "Processing %d files from %s → %s (workers=%d).",
        len(gxl_files), gxl_dir, out_dir, workers,
    )

    summary = BatchSummary()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one, path, out_dir, tol): path
            for path in gxl_files
        }
        for future in as_completed(futures):
            result = future.result()
            summary.results.append(result)
            status = "OK" if result.success else "FAIL"
            logger.info("[%s] %s (%.2fs)", status, result.name, result.elapsed)
            if not result.success:
                logger.debug("Error for %s:\n%s", result.name, result.error)

    summary.results.sort(key=lambda r: r.name)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Batch-construct hyperbolic polyhedra from GXL 1-skeleton graphs."
    )
    parser.add_argument("gxl_dir", help="Directory of input .gxl files")
    parser.add_argument("out_dir", help="Directory for output .off files")
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-10,
        help="Newton convergence tolerance (default: 1e-10)",
    )
    parser.add_argument(
        "--glob", default="*.gxl",
        help="Glob pattern for input files (default: '*.gxl')",
    )
    args = parser.parse_args()

    summary = run_batch(
        args.gxl_dir, args.out_dir,
        workers=args.workers, tol=args.tol, glob=args.glob,
    )
    summary.print_summary()
    raise SystemExit(0 if summary.n_failed == 0 else 1)


if __name__ == "__main__":
    _cli()
