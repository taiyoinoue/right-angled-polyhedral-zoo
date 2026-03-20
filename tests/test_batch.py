"""
Smoke tests for hyperhedron.batch.

We copy a few known-good GXL files from the MATLAB examples directory into a
temporary directory and verify that run_batch produces OFF files without errors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hyperhedron.batch import run_batch, _process_one

# Location of the MATLAB GXL examples
_MATLAB_DIR = Path("/home/taiyo/hyperhedron")
_GXL_FILES = [
    _MATLAB_DIR / "Lobell6.gxl",
    _MATLAB_DIR / "Lobell7.gxl",
]

# Skip entire module if the MATLAB dir isn't present
pytestmark = pytest.mark.skipif(
    not _MATLAB_DIR.exists(),
    reason="MATLAB example directory not found",
)


class TestProcessOne:
    def test_lobell6_produces_off(self, tmp_path):
        src = _MATLAB_DIR / "Lobell6.gxl"
        if not src.exists():
            pytest.skip("Lobell6.gxl not found")
        result = _process_one(src, tmp_path, tol=1e-8)
        assert result.success, f"Failed: {result.error}"
        assert result.off_path is not None
        assert result.off_path.exists()
        # OFF file must start with a header line (V F E)
        first_line = result.off_path.read_text().splitlines()[0]
        parts = first_line.split()
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_failed_result_on_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.gxl"
        bad.write_text("<gxl><graph></graph></gxl>")
        result = _process_one(bad, tmp_path, tol=1e-8)
        assert not result.success
        assert result.error != ""


class TestRunBatch:
    def test_batch_on_lobell_files(self, tmp_path):
        # Copy a couple of GXL files into a temp input dir
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        available = [p for p in _GXL_FILES if p.exists()]
        if not available:
            pytest.skip("No Lobell GXL files found")

        for p in available:
            shutil.copy(p, in_dir / p.name)

        summary = run_batch(in_dir, out_dir, workers=2, tol=1e-8)

        assert len(summary.results) == len(available)
        assert summary.n_success == len(available), (
            f"Failed: {[r.error for r in summary.failed]}"
        )
        # Each success should have produced an OFF file
        for result in summary.results:
            if result.success:
                assert result.off_path is not None
                assert result.off_path.exists()

    def test_empty_directory_returns_empty_summary(self, tmp_path):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        out_dir = tmp_path / "out"
        summary = run_batch(in_dir, out_dir, workers=1)
        assert len(summary.results) == 0

    def test_summary_counts(self, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        available = [p for p in _GXL_FILES[:1] if p.exists()]
        if not available:
            pytest.skip("No Lobell GXL files found")

        for p in available:
            shutil.copy(p, in_dir / p.name)

        summary = run_batch(in_dir, out_dir, workers=1, tol=1e-8)
        assert summary.n_success + summary.n_failed == len(summary.results)
