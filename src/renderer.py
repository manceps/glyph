"""SVG and PNG rendering for Glyph visual codes.

Renders a radial-ring topology tree as a shape boundary with
data-encoding cells arranged in concentric rings.

Color assignment:
- Root (depth 0): outer boundary with primary stroke, transparent fill
- Data cells (depth 2): filled with palette color for bit=1,
  white fill for bit=0 (ensures clear contrast for decoding)
- Ring-based color variation: cells in different rings use
  different palette colors for visual richness

Tier-based visual modifications (optional):
- tier=0: opacity=0.4, saturation=50%, no border ring
- tier=1: opacity=0.7, saturation=75%, thin border ring
- tier=2: opacity=1.0, saturation=100%, border ring
- tier=3: opacity=1.0, saturation=100%, border ring + glow

The rendering is deliberately abstract -- these are NOT QR codes.
They look like organic, layered geometric patterns.
"""

from __future__ import annotations

import structlog

from .encoder import TopologyTree, compute_polygon_positions
from .shapes import select_shape

logger = structlog.get_logger(__name__)

# Default fallback colors (cool neutral palette)
DEFAULT_COLORS = ["#2D3748", "#4A5568", "#718096", "#A0AEC0", "#CBD5E0"]

# Tier visual configuration (optional -- applications can define their own)
TIER_CONFIG = {
    0: {"opacity": 0.4, "saturation": 0.50, "ring_color": None, "ring_label": "NEWCOMER"},
    1: {"opacity": 0.7, "saturation": 0.75, "ring_color": "#CD7F32", "ring_label": "MEMBER"},
    2: {"opacity": 1.0, "saturation": 1.00, "ring_color": "#C0C0C0", "ring_label": "TRUSTED"},
    3: {"opacity": 1.0, "saturation": 1.00, "ring_color": "#FFD700", "ring_label": "STEWARD"},
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert (r, g, b) to hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _desaturate(hex_color: str, saturation: float) -> str:
    """Reduce saturation of a hex color toward gray.

    Args:
        hex_color: Input hex color (e.g. "#2D3748").
        saturation: Saturation factor (0.0 = fully gray, 1.0 = original).

    Returns:
        Hex color with reduced saturation.
    """
    if saturation >= 1.0:
        return hex_color
    r, g, b = _hex_to_rgb(hex_color)
    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
    r = int(gray + saturation * (r - gray))
    g = int(gray + saturation * (g - gray))
    b = int(gray + saturation * (b - gray))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return _rgb_to_hex(r, g, b)


def render_svg(
    tree: TopologyTree,
    shape_name: str,
    colors: list[str] | None = None,
    size: int = 512,
    tier: int | None = None,
    glyph_subtype: str | None = None,
) -> str:
    """Render a topology tree as an SVG string.

    Args:
        tree: Encoded topology tree.
        shape_name: Shape template name.
        colors: List of hex color strings.
        size: Output size in pixels (width = height).
        tier: Trust tier (0-3) for visual styling. None for default behavior.
        glyph_subtype: Visual variant ("community", "recovery", "invite"), or None.

    Returns:
        Complete SVG document as a string.
    """
    shape = select_shape(shape_name)
    palette = colors or DEFAULT_COLORS
    polygons = compute_polygon_positions(tree, shape, size)

    # Get tier configuration
    tier_cfg = TIER_CONFIG.get(tier, None) if tier is not None else None
    cell_opacity = tier_cfg["opacity"] if tier_cfg else 0.90
    sat = tier_cfg["saturation"] if tier_cfg else 1.0
    ring_color = tier_cfg["ring_color"] if tier_cfg else None

    # Recovery subtype: dimmed appearance
    if glyph_subtype == "recovery":
        cell_opacity = 0.5
        ring_color = ring_color or "#888888"

    # Invite subtype: slightly brighter colors (boost saturation)
    if glyph_subtype == "invite":
        sat = min(sat * 1.25, 1.0)
        ring_color = ring_color or "#4488CC"

    # Apply saturation to palette
    effective_palette = [_desaturate(c, sat) for c in palette]

    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}">',
    ]

    # Defs for glow effect (tier 3)
    if tier == 3 and ring_color:
        svg_parts.append("  <defs>")
        svg_parts.append('    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">')
        svg_parts.append('      <feGaussianBlur stdDeviation="4" result="blur"/>')
        svg_parts.append(f'      <feFlood flood-color="{ring_color}" flood-opacity="0.3"/>')
        svg_parts.append('      <feComposite in2="blur" operator="in"/>')
        svg_parts.append("      <feMerge>")
        svg_parts.append("        <feMergeNode/>")
        svg_parts.append('        <feMergeNode in="SourceGraphic"/>')
        svg_parts.append("      </feMerge>")
        svg_parts.append("    </filter>")
        svg_parts.append("  </defs>")

    # White background and border
    svg_parts.append(f'  <rect width="{size}" height="{size}" fill="white"/>')
    svg_parts.append(
        f'  <rect x="3" y="3" width="{size - 6}" height="{size - 6}" '
        f'fill="none" stroke="#555555" stroke-width="5"/>'
    )

    # Tier border ring(s) -- rendered BEFORE data cells so they sit behind
    if ring_color is not None:
        cx = size / 2.0
        cy = size / 2.0
        ring_r = size * 0.46  # slightly outside the glyph boundary

        if glyph_subtype == "community":
            # Double-ring for community-type glyphs
            outer_r = ring_r + 2
            inner_r = ring_r - 5
            svg_parts.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{outer_r:.1f}" '
                f'fill="none" stroke="{ring_color}" stroke-width="2.0" '
                f'class="tier-ring tier-ring-outer"/>'
            )
            svg_parts.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{inner_r:.1f}" '
                f'fill="none" stroke="{ring_color}" stroke-width="2.0" '
                f'class="tier-ring tier-ring-inner"/>'
            )
        elif glyph_subtype == "recovery":
            # Dashed ring for recovery-type glyphs
            svg_parts.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{ring_r:.1f}" '
                f'fill="none" stroke="{ring_color}" stroke-width="2.5" '
                f'stroke-dasharray="8,4" '
                f'class="tier-ring tier-ring-recovery"/>'
            )
        elif glyph_subtype == "invite":
            # Dotted ring for invite-type glyphs
            svg_parts.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{ring_r:.1f}" '
                f'fill="none" stroke="{ring_color}" stroke-width="2.5" '
                f'stroke-dasharray="2,4" stroke-linecap="round" '
                f'class="tier-ring tier-ring-invite"/>'
            )
        else:
            # Single ring (default)
            stroke_w = 2.0 if tier == 1 else 3.0
            filter_attr = ' filter="url(#glow)"' if tier == 3 else ""
            svg_parts.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{ring_r:.1f}" '
                f'fill="none" stroke="{ring_color}" stroke-width="{stroke_w}"{filter_attr} '
                f'class="tier-ring"/>'
            )

    for poly in polygons:
        vertices = poly["vertices"]
        depth = poly["depth"]
        bit_value = poly["bit_value"]

        # Build SVG polygon points string
        points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in vertices)

        if depth == 0:
            # Root boundary: outline stroke, transparent fill
            svg_parts.append(
                f'  <polygon points="{points_str}" '
                f'fill="none" '
                f'stroke="{effective_palette[0]}" '
                f'stroke-width="2.5" '
                f'opacity="1.00"/>'
            )
        elif bit_value == 1:
            # Filled data cell: single dark color for reliable camera decoding.
            fill_color = (
                effective_palette[1] if len(effective_palette) > 1 else effective_palette[0]
            )
            stroke_color = effective_palette[0]
            svg_parts.append(
                f'  <polygon points="{points_str}" '
                f'fill="{fill_color}" '
                f'stroke="{stroke_color}" '
                f'stroke-width="0.5" '
                f'opacity="{cell_opacity:.2f}"/>'
            )
        else:
            # Empty data cell: white fill, no stroke (clean for decoding)
            svg_parts.append(
                f'  <polygon points="{points_str}" '
                f'fill="white" '
                f'stroke="none" '
                f'stroke-width="0" '
                f'opacity="1.00"/>'
            )

    svg_parts.append("</svg>")
    svg_content = "\n".join(svg_parts)

    logger.debug(
        "svg_rendered",
        shape=shape_name,
        polygon_count=len(polygons),
        size=size,
        tier=tier,
        glyph_subtype=glyph_subtype,
    )

    return svg_content


def render_png(
    tree: TopologyTree,
    shape_name: str,
    colors: list[str] | None = None,
    size: int = 512,
    tier: int | None = None,
    glyph_subtype: str | None = None,
) -> bytes:
    """Render a topology tree as a PNG image.

    Generates SVG first, then converts to PNG via CairoSVG.

    Args:
        tree: Encoded topology tree.
        shape_name: Shape template name.
        colors: List of hex color strings.
        size: Output size in pixels.
        tier: Trust tier (0-3) for visual styling. None for default behavior.
        glyph_subtype: Visual variant ("community", "recovery", "invite"), or None.

    Returns:
        PNG image bytes.
    """
    import cairosvg

    svg = render_svg(tree, shape_name, colors, size, tier=tier, glyph_subtype=glyph_subtype)
    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=size,
        output_height=size,
    )

    logger.debug("png_rendered", shape=shape_name, size=size, bytes=len(png_bytes))
    return png_bytes
