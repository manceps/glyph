#!/usr/bin/env python3
"""Basic usage example for Glyph.

Demonstrates encoding hex data into a Glyph and decoding it back.

Usage:
    python examples/basic_usage.py
"""

import hashlib
import sys
import os

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import encode, encode_member_payload
from src.decoder import decode_image, parse_glyph_type
from src.renderer import render_svg, render_png


def example_basic_roundtrip():
    """Encode arbitrary hex data and decode it back."""
    print("=" * 60)
    print("Example 1: Basic Encode/Decode Roundtrip")
    print("=" * 60)

    data_hex = "deadbeefcafebabe0123456789abcdef"
    print(f"  Input data:  {data_hex}")
    print(f"  Data bytes:  {len(data_hex) // 2}")

    # Encode
    tree = encode(data_hex, "hexagon")
    print(f"  Tree nodes:  {tree.total_nodes}")
    print(f"  Data bits:   {tree.bit_length}")

    # Render as PNG
    png_bytes = render_png(tree, "hexagon", size=512)
    print(f"  PNG size:    {len(png_bytes)} bytes")

    # Decode
    result = decode_image(png_bytes)
    print(f"  Decoded:     {result.data_hex}")
    print(f"  Confidence:  {result.confidence}")
    print(f"  Match:       {result.data_hex == data_hex}")
    print()


def example_custom_colors():
    """Encode with custom colors and different shapes."""
    print("=" * 60)
    print("Example 2: Custom Colors and Shapes")
    print("=" * 60)

    data_hex = "abcdef0123456789"
    colors = ["#1A365D", "#2B6CB0", "#4299E1", "#90CDF4", "#EBF8FF"]

    for shape in ["hexagon", "circle", "shield", "diamond"]:
        tree = encode(data_hex, shape)
        svg = render_svg(tree, shape, colors=colors, size=256)
        print(f"  Shape: {shape:10s}  SVG length: {len(svg):5d} chars")

    print()


def example_sha256_identity():
    """Derive a visual identity from a public key hash."""
    print("=" * 60)
    print("Example 3: Visual Identity from Public Key")
    print("=" * 60)

    # Simulate a public key
    fake_pubkey = b"ed25519_public_key_for_user_alice"
    pubkey_hash = hashlib.sha256(fake_pubkey).hexdigest()
    print(f"  Public key hash: {pubkey_hash}")

    # Use first 16 bytes as glyph data
    glyph_data = pubkey_hash[:32]
    print(f"  Glyph data:      {glyph_data}")

    # Deterministically derive shape from hash
    shapes = ["circle", "hexagon", "octagon", "shield",
              "diamond", "pentagon", "heptagon", "triangle"]
    shape_idx = int(pubkey_hash[0:2], 16) % len(shapes)
    shape = shapes[shape_idx]
    print(f"  Derived shape:   {shape}")

    # Encode and verify roundtrip
    tree = encode(glyph_data, shape)
    png = render_png(tree, shape, size=512)
    result = decode_image(png)
    print(f"  Roundtrip OK:    {result.data_hex == glyph_data}")
    print()


def example_typed_payload():
    """Encode a typed MEMBER payload with tier-based visuals."""
    print("=" * 60)
    print("Example 4: Typed MEMBER Payload")
    print("=" * 60)

    pubkey_prefix = bytes(range(1, 13))  # 12 bytes
    tier = 2  # TRUSTED

    # Encode typed payload
    data_hex = encode_member_payload(pubkey_prefix, tier=tier)
    print(f"  Payload hex:  {data_hex}")
    print(f"  Payload size: {len(data_hex) // 2} bytes")

    # Render with tier styling (silver ring for TRUSTED)
    tree = encode(data_hex, "hexagon")
    png = render_png(tree, "hexagon", size=512, tier=tier)
    print(f"  PNG size:     {len(png)} bytes")

    # Decode and parse type
    result = decode_image(png)
    parsed = parse_glyph_type(result.data_hex)
    print(f"  Parsed type:  {parsed['type']}")
    print(f"  Pubkey prefix:{parsed.get('pubkey_prefix', 'N/A')}")
    print(f"  Tier:         {parsed.get('tier', 'N/A')}")
    print()


def example_all_tiers():
    """Show all tier visual configurations."""
    print("=" * 60)
    print("Example 5: All Trust Tiers")
    print("=" * 60)

    tier_names = {0: "NEWCOMER", 1: "MEMBER", 2: "TRUSTED", 3: "STEWARD"}
    pubkey = bytes(range(20, 32))

    for tier in range(4):
        data_hex = encode_member_payload(pubkey, tier=tier)
        tree = encode(data_hex, "hexagon")
        svg = render_svg(tree, "hexagon", size=256, tier=tier)
        has_ring = "tier-ring" in svg
        has_glow = "glow" in svg
        print(
            f"  Tier {tier} ({tier_names[tier]:10s}): "
            f"SVG {len(svg):5d} chars, "
            f"ring={'yes' if has_ring else 'no ':3s}, "
            f"glow={'yes' if has_glow else 'no'}"
        )

    print()


if __name__ == "__main__":
    example_basic_roundtrip()
    example_custom_colors()
    example_sha256_identity()
    example_typed_payload()
    example_all_tiers()
    print("All examples completed successfully.")
