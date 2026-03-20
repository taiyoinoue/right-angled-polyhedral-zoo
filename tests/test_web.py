"""
Integration tests for the web API.

Skipped if the Lobell GXL files are not present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

LOBELL6_GXL = Path("/home/taiyo/hyperhedron/Lobell6.gxl")
pytestmark = pytest.mark.skipif(
    not LOBELL6_GXL.exists(),
    reason="Lobell GXL files not present",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Return a TestClient with the web app, using a fresh cache dir."""
    from fastapi.testclient import TestClient
    import hyperhedron.web.app as mod

    mod._GXL_DIR = LOBELL6_GXL.parent
    mod._CACHE_DIR = tmp_path_factory.mktemp("web_cache")

    from hyperhedron.web.app import app
    return TestClient(app)


class TestListEndpoint:
    def test_list_returns_array(self, client):
        r = client.get("/api/polyhedra")
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)
        assert len(items) > 0

    def test_lobell6_in_list(self, client):
        r = client.get("/api/polyhedra")
        names = {item["name"] for item in r.json()}
        assert "Lobell6" in names

    def test_computed_field_present(self, client):
        r = client.get("/api/polyhedra")
        for item in r.json():
            assert "computed" in item
            assert isinstance(item["computed"], bool)


class TestPolyhedronEndpoint:
    def test_lobell6_returns_200(self, client):
        r = client.get("/api/polyhedra/Lobell6")
        assert r.status_code == 200

    def test_lobell6_shape(self, client):
        data = client.get("/api/polyhedra/Lobell6").json()
        assert data["name"] == "Lobell6"
        assert data["num_faces"] > 0
        assert data["num_vertices"] > 0
        assert data["num_edges"] > 0
        assert len(data["vertices"]) == data["num_vertices"]
        assert len(data["faces"])    == data["num_faces"]
        assert len(data["edges"])    == data["num_edges"]

    def test_vertices_inside_unit_ball(self, client):
        data = client.get("/api/polyhedra/Lobell6").json()
        for i, v in enumerate(data["vertices"]):
            r2 = sum(x * x for x in v)
            assert r2 < 1.0, f"vertex {i} outside unit ball: r={math.sqrt(r2):.4f}"

    def test_second_request_uses_cache(self, client):
        """Second call should be fast (cache hit) and return the same data."""
        r1 = client.get("/api/polyhedra/Lobell6").json()
        r2 = client.get("/api/polyhedra/Lobell6").json()
        assert r1 == r2

    def test_missing_name_returns_404(self, client):
        r = client.get("/api/polyhedra/DoesNotExist99999")
        assert r.status_code == 404

    def test_list_shows_computed_after_fetch(self, client):
        # Lobell6 was already fetched above; list should mark it computed.
        items = {i["name"]: i for i in client.get("/api/polyhedra").json()}
        assert items["Lobell6"]["computed"] is True

    def test_index_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Hyperhedron Zoo" in r.text
