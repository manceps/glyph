"""Tests for Glyph FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "glyph"

    def test_health_includes_version(self):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


class TestEncodeEndpoint:
    def test_encode_png_returns_image(self):
        resp = client.post(
            "/encode",
            json={
                "data_hex": "deadbeefcafebabe",
                "shape": "hexagon",
                "size": 128,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:4] == b"\x89PNG"

    def test_encode_svg_returns_svg(self):
        resp = client.post(
            "/encode/svg",
            json={
                "data_hex": "deadbeefcafebabe",
                "shape": "circle",
                "size": 256,
            },
        )
        assert resp.status_code == 200
        assert "image/svg+xml" in resp.headers["content-type"]
        assert "<svg" in resp.text

    def test_encode_invalid_hex_returns_422(self):
        resp = client.post(
            "/encode",
            json={
                "data_hex": "not_valid_hex",
                "shape": "hexagon",
            },
        )
        assert resp.status_code == 422

    def test_encode_unknown_shape_returns_422(self):
        resp = client.post(
            "/encode",
            json={
                "data_hex": "deadbeef",
                "shape": "star",
            },
        )
        assert resp.status_code == 422

    def test_encode_with_custom_colors(self):
        resp = client.post(
            "/encode/svg",
            json={
                "data_hex": "deadbeef",
                "shape": "shield",
                "colors": ["#FF0000", "#00FF00", "#0000FF"],
                "size": 128,
            },
        )
        assert resp.status_code == 200
        assert "#FF0000" in resp.text

    def test_encode_respects_size_limits(self):
        resp = client.post(
            "/encode",
            json={
                "data_hex": "deadbeef",
                "shape": "hexagon",
                "size": 32,  # below minimum
            },
        )
        assert resp.status_code == 422

    def test_encode_all_shapes(self):
        for shape in [
            "circle",
            "hexagon",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        ]:
            resp = client.post(
                "/encode/svg",
                json={
                    "data_hex": "deadbeef",
                    "shape": shape,
                    "size": 128,
                },
            )
            assert resp.status_code == 200, f"Failed for shape: {shape}"

    def test_encode_large_data(self):
        resp = client.post(
            "/encode/svg",
            json={
                "data_hex": "ab" * 32,  # 32 bytes
                "shape": "hexagon",
                "size": 128,
            },
        )
        assert resp.status_code == 200


class TestDecodeEndpoint:
    def test_decode_accepts_png(self):
        # First generate a PNG
        encode_resp = client.post(
            "/encode",
            json={
                "data_hex": "deadbeefcafebabe",
                "shape": "hexagon",
                "size": 256,
            },
        )
        png_bytes = encode_resp.content

        # Then decode it
        resp = client.post(
            "/decode",
            files={"file": ("glyph.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data_hex" in data
        assert "confidence" in data

    def test_decode_rejects_invalid_type(self):
        resp = client.post(
            "/decode",
            files={"file": ("data.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422


class TestTypedEncodeEndpoint:
    def test_encode_member_png(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "MEMBER",
                "pubkey_hash_prefix": "0102030405060708090a0b0c",
                "tier": 1,
                "shape": "hexagon",
                "size": 128,
                "format": "png",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:4] == b"\x89PNG"

    def test_encode_member_svg(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "MEMBER",
                "pubkey_hash_prefix": "0102030405060708090a0b0c",
                "tier": 2,
                "shape": "hexagon",
                "size": 256,
                "format": "svg",
            },
        )
        assert resp.status_code == 200
        assert "image/svg+xml" in resp.headers["content-type"]
        assert "<svg" in resp.text
        assert "#C0C0C0" in resp.text

    def test_encode_community_png(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "COMMUNITY",
                "community_id_prefix": "aabbccdd",
                "join_requirement": 1,
                "expiry_timestamp": 1700000000,
                "presenter_prefix": "11223344",
                "tier": 1,
                "shape": "circle",
                "size": 128,
                "format": "png",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_encode_community_svg_has_double_ring(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "COMMUNITY",
                "community_id_prefix": "aabbccdd",
                "join_requirement": 0,
                "expiry_timestamp": 1700000000,
                "presenter_prefix": "11223344",
                "tier": 2,
                "shape": "hexagon",
                "size": 256,
                "format": "svg",
            },
        )
        assert resp.status_code == 200
        assert "tier-ring-outer" in resp.text
        assert "tier-ring-inner" in resp.text

    def test_encode_typed_unknown_type_returns_422(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "UNKNOWN",
                "shape": "hexagon",
                "size": 128,
            },
        )
        assert resp.status_code == 422

    def test_encode_member_missing_pubkey_returns_422(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "MEMBER",
                "tier": 0,
                "shape": "hexagon",
                "size": 128,
            },
        )
        assert resp.status_code == 422

    def test_encode_community_missing_fields_returns_422(self):
        resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "COMMUNITY",
                "shape": "hexagon",
                "size": 128,
            },
        )
        assert resp.status_code == 422


class TestDecodeEndpointGlyphType:
    def test_decode_returns_glyph_type_for_member(self):
        """Typed member glyph includes glyph_type in decode response."""
        encode_resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "MEMBER",
                "pubkey_hash_prefix": "0102030405060708090a0b0c",
                "tier": 2,
                "shape": "hexagon",
                "size": 512,
                "format": "png",
            },
        )
        png_bytes = encode_resp.content

        resp = client.post(
            "/decode",
            files={"file": ("glyph.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["glyph_type"] == "MEMBER"
        assert data["glyph_fields"] is not None
        assert data["glyph_fields"]["tier"] == 2
        assert data["glyph_fields"]["pubkey_prefix"] == "0102030405060708090a0b0c"

    def test_decode_returns_glyph_type_for_community(self):
        """Typed community glyph includes glyph_type in decode response."""
        encode_resp = client.post(
            "/encode/typed",
            json={
                "glyph_type": "COMMUNITY",
                "community_id_prefix": "aabbccdd",
                "join_requirement": 1,
                "expiry_timestamp": 1700000000,
                "presenter_prefix": "11223344",
                "shape": "hexagon",
                "size": 512,
                "format": "png",
            },
        )
        png_bytes = encode_resp.content

        resp = client.post(
            "/decode",
            files={"file": ("glyph.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["glyph_type"] == "COMMUNITY"
        assert data["glyph_fields"] is not None
        assert data["glyph_fields"]["community_id_prefix"] == "aabbccdd"
        assert data["glyph_fields"]["join_requirement"] == 1

    def test_decode_returns_legacy_for_old_payloads(self):
        """Old-style payloads should decode as LEGACY type."""
        encode_resp = client.post(
            "/encode",
            json={
                "data_hex": "a3f04b2c9e1d7f8a0011223344556677",  # pragma: allowlist secret
                "shape": "hexagon",
                "size": 512,
            },
        )
        png_bytes = encode_resp.content

        resp = client.post(
            "/decode",
            files={"file": ("glyph.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["glyph_type"] == "LEGACY"
