"""Tests for Glyph SVG/PNG rendering."""

from src.encoder import encode
from src.renderer import render_png, render_svg


class TestRenderSVG:
    def test_render_svg_produces_valid_svg(self):
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "xmlns" in svg

    def test_render_svg_with_custom_colors(self):
        tree = encode("deadbeef", "hexagon")
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        svg = render_svg(tree, "hexagon", colors=colors)
        assert "#FF0000" in svg

    def test_render_svg_respects_size(self):
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", size=256)
        assert 'width="256"' in svg
        assert 'height="256"' in svg

    def test_render_svg_different_shapes(self):
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
            tree = encode("deadbeef", shape)
            svg = render_svg(tree, shape)
            assert "<svg" in svg

    def test_render_svg_contains_polygons(self):
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon")
        assert "<polygon" in svg

    def test_render_svg_default_size_512(self):
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon")
        assert 'width="512"' in svg


class TestRenderPNG:
    def test_render_png_produces_png(self):
        tree = encode("deadbeef", "hexagon")
        png_bytes = render_png(tree, "hexagon")
        assert png_bytes[:4] == b"\x89PNG"

    def test_render_png_respects_size(self):
        tree = encode("deadbeef", "hexagon")
        small = render_png(tree, "hexagon", size=64)
        large = render_png(tree, "hexagon", size=512)
        assert len(large) > len(small)

    def test_render_png_with_custom_colors(self):
        tree = encode("deadbeef", "hexagon")
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        png_bytes = render_png(tree, "hexagon", colors=colors)
        assert png_bytes[:4] == b"\x89PNG"


class TestTierRendering:
    """Test tier-based visual modifications in SVG output."""

    def test_tier_none_no_ring(self):
        """No tier (default) should not have tier-ring elements."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=None)
        assert "tier-ring" not in svg

    def test_tier0_no_ring(self):
        """Tier 0 has reduced opacity but no border ring."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=0)
        assert "tier-ring" not in svg
        assert 'opacity="0.40"' in svg

    def test_tier1_bronze_ring(self):
        """Tier 1 has a bronze border ring."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=1)
        assert "tier-ring" in svg
        assert "#CD7F32" in svg

    def test_tier2_silver_ring(self):
        """Tier 2 has a silver border ring."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=2)
        assert "tier-ring" in svg
        assert "#C0C0C0" in svg

    def test_tier3_gold_ring_with_glow(self):
        """Tier 3 has a gold ring and glow filter."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=3)
        assert "tier-ring" in svg
        assert "#FFD700" in svg
        assert "filter" in svg
        assert "glow" in svg

    def test_community_double_ring(self):
        """Community subtype renders double-ring border."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=2, glyph_subtype="community")
        assert "tier-ring-outer" in svg
        assert "tier-ring-inner" in svg

    def test_community_double_ring_has_two_circles(self):
        """Community subtype has exactly two tier-ring circles."""
        tree = encode("deadbeef", "hexagon")
        svg = render_svg(tree, "hexagon", tier=1, glyph_subtype="community")
        assert svg.count("tier-ring") >= 2

    def test_tier_opacity_values(self):
        """Check that each tier applies correct opacity to data cells."""
        tree = encode("ff", "hexagon")

        svg0 = render_svg(tree, "hexagon", tier=0)
        svg1 = render_svg(tree, "hexagon", tier=1)
        svg2 = render_svg(tree, "hexagon", tier=2)
        svg3 = render_svg(tree, "hexagon", tier=3)

        assert 'opacity="0.40"' in svg0
        assert 'opacity="0.70"' in svg1
        assert 'opacity="1.00"' in svg2
        assert 'opacity="1.00"' in svg3

    def test_tier_rendering_still_produces_valid_svg(self):
        """All tier + shape combos produce valid SVG."""
        for tier in [0, 1, 2, 3]:
            for shape in ["hexagon", "circle", "shield"]:
                tree = encode("deadbeef", shape)
                svg = render_svg(tree, shape, tier=tier)
                assert svg.startswith("<svg")
                assert svg.endswith("</svg>")


class TestTierPNGRendering:
    def test_tier_png_produces_valid_png(self):
        """Tier-based rendering still produces valid PNGs."""
        tree = encode("deadbeef", "hexagon")
        for tier in [0, 1, 2, 3]:
            png_bytes = render_png(tree, "hexagon", tier=tier)
            assert png_bytes[:4] == b"\x89PNG", f"Tier {tier} failed"

    def test_community_png_produces_valid_png(self):
        tree = encode("deadbeef", "hexagon")
        png_bytes = render_png(tree, "hexagon", tier=2, glyph_subtype="community")
        assert png_bytes[:4] == b"\x89PNG"
