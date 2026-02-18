"""Radial-ring encoder for Glyph visual codes.

Converts a hex data payload into a TopologyTree of cells arranged in
concentric rings inside the outer shape boundary.

Encoding algorithm:
1. Convert hex data to bits
2. Append CRC-32 for error detection
3. Compute cell positions in concentric rings around the center
4. Assign one bit per cell (filled=1, empty=0)
5. Build tree structure: root -> ring nodes -> cell nodes

The radial layout has deterministic cell positions independent of
data content, making it robustly decodable by sampling image intensity
at each known cell centroid.
"""

from __future__ import annotations

import math

import structlog

from .bits import (
    GLYPH_TYPE_COMMUNITY,
    GLYPH_TYPE_INVITE,
    GLYPH_TYPE_MEMBER,
    GLYPH_TYPE_RECOVERY,
    append_crc,
    hex_to_bits,
)
from .shapes import ShapeTemplate, select_shape
from .tree import TopologyTree, TreeNode

logger = structlog.get_logger(__name__)

# Maximum data size in bytes
MAX_DATA_BYTES = 64

# Layout constants (shared with decoder)
CENTER = 0.5  # Normalized center
OUTER_RADIUS = 0.42  # Outer ring radius (< 0.45 shape radius to stay inside)
INNER_RADIUS = 0.08  # Inner ring radius (small center gap)
CELL_RADIUS_FRACTION = 0.32  # Cell radius as fraction of ring spacing
MIN_CELL_GAP_RATIO = 1.6  # Minimum gap between cell centers / cell diameter


def compute_ring_layout(total_bits: int) -> list[tuple[int, float]]:
    """Compute concentric ring layout for a given number of bits.

    Distributes cells across rings from inside out.  Inner rings have
    fewer cells (less circumference), outer rings have more.

    Args:
        total_bits: Total number of cells needed.

    Returns:
        List of (cells_in_ring, ring_radius) tuples.
    """
    if total_bits <= 0:
        return []

    # Start with a rough estimate of ring count
    # Each ring's capacity ~ 2*pi*r / cell_angular_spacing
    # We want cells to be roughly equally spaced (not overlapping)
    ring_count = max(1, int(math.sqrt(total_bits / 3)))

    while True:
        rings = _distribute_cells(total_bits, ring_count)
        total_capacity = sum(c for c, _ in rings)
        if total_capacity >= total_bits:
            break
        ring_count += 1

    return rings


def _distribute_cells(total_bits: int, ring_count: int) -> list[tuple[int, float]]:
    """Distribute cells across a fixed number of rings.

    Ensures cells are well-separated (gap >= MIN_CELL_GAP_RATIO * diameter)
    for reliable decoding.

    Args:
        total_bits: Total cells needed.
        ring_count: Number of rings to use.

    Returns:
        List of (cells_in_ring, ring_radius) tuples.
    """
    rings: list[tuple[int, float]] = []
    ring_spacing = (OUTER_RADIUS - INNER_RADIUS) / max(ring_count, 1)
    cell_diameter = ring_spacing * CELL_RADIUS_FRACTION * 2

    remaining = total_bits
    for r in range(ring_count):
        radius = INNER_RADIUS + (r + 0.5) * ring_spacing
        # Max cells such that arc gap between cell centers >= MIN_CELL_GAP_RATIO * diameter
        circumference = 2 * math.pi * radius
        min_arc_gap = cell_diameter * MIN_CELL_GAP_RATIO
        max_cells = max(3, int(circumference / min_arc_gap))
        cells = min(max_cells, remaining)
        rings.append((cells, radius))
        remaining -= cells
        if remaining <= 0:
            break

    return rings


def encode(data_hex: str, shape_name: str) -> TopologyTree:
    """Encode hex data into a radial-ring topology tree.

    Args:
        data_hex: Hex-encoded data payload.
        shape_name: Shape template name for the outer boundary.

    Returns:
        TopologyTree ready for rendering.

    Raises:
        ValueError: If data exceeds maximum size or shape is invalid.
    """
    # Validate shape name early (before doing any work)
    select_shape(shape_name)

    clean = data_hex.removeprefix("0x").removeprefix("0X")
    if len(clean) // 2 > MAX_DATA_BYTES:
        raise ValueError(f"Data too large: {len(clean) // 2} bytes (max {MAX_DATA_BYTES})")

    # Append CRC-32 for error detection (4 bytes)
    data_with_crc = append_crc(clean)
    bits = hex_to_bits(data_with_crc)

    logger.debug(
        "encoding_topology",
        data_bytes=len(clean) // 2,
        total_bits=len(bits),
        shape=shape_name,
    )

    tree = _build_radial_tree(bits)
    tree.bit_length = len(hex_to_bits(clean))

    return tree


def _build_radial_tree(bits: list[int]) -> TopologyTree:
    """Build a radial-ring topology tree from bits.

    Structure:
    - Root (depth 0): outer shape boundary
    - Ring nodes (depth 1): one per ring, no bit value
    - Cell nodes (depth 2): one per bit, positioned in ring

    Args:
        bits: Bit sequence including CRC.

    Returns:
        TopologyTree with radial layout.
    """
    tree = TopologyTree()
    root = TreeNode(bit_value=None, depth=0, polygon_index=0)

    rings = compute_ring_layout(len(bits))
    bit_idx = 0

    for ring_idx, (cells_in_ring, _radius) in enumerate(rings):
        ring_node = TreeNode(bit_value=None, depth=1, polygon_index=ring_idx)
        root.children.append(ring_node)

        for cell_idx in range(cells_in_ring):
            if bit_idx < len(bits):
                bit_val = bits[bit_idx]
                bit_idx += 1
            else:
                bit_val = 0

            cell = TreeNode(bit_value=bit_val, depth=2, polygon_index=cell_idx)
            ring_node.children.append(cell)

    tree.root = root
    return tree


def compute_polygon_positions(
    tree: TopologyTree,
    shape: ShapeTemplate,
    size: int,
) -> list[dict]:
    """Compute absolute polygon positions for rendering.

    Returns outer boundary polygon plus one polygon per data cell.

    Args:
        tree: Topology tree with radial layout.
        shape: Shape template for the outer boundary.
        size: Output image size in pixels (square).

    Returns:
        List of polygon dicts with keys:
        - vertices: List of (x, y) tuples in pixel space
        - depth: 0 for boundary, 2 for data cells
        - bit_value: The bit this polygon encodes (None for root/rings)
        - polygon_index: Cell index within ring
        - ring_index: Ring index (for data cells)
    """
    polygons: list[dict] = []

    # Root polygon (outer boundary)
    scaled_boundary = [(x * size, y * size) for x, y in shape.vertices]
    polygons.append(
        {
            "vertices": scaled_boundary,
            "depth": 0,
            "bit_value": None,
            "polygon_index": 0,
        }
    )

    # Compute ring layout from tree structure
    for ring_node in tree.root.children:
        ring_idx = ring_node.polygon_index
        cells_in_ring = len(ring_node.children)

        # Recompute ring layout to get radius
        total_cells = sum(len(rn.children) for rn in tree.root.children)
        rings = compute_ring_layout(total_cells)
        if ring_idx >= len(rings):
            continue
        _cells, ring_radius = rings[ring_idx]

        # Ring spacing for cell sizing
        ring_count = len(rings)
        ring_spacing = (OUTER_RADIUS - INNER_RADIUS) / max(ring_count, 1)
        cell_r = ring_spacing * CELL_RADIUS_FRACTION

        for cell_node in ring_node.children:
            cell_idx = cell_node.polygon_index
            angle = (2 * math.pi * cell_idx / cells_in_ring) - (math.pi / 2)

            cx = CENTER + ring_radius * math.cos(angle)
            cy = CENTER + ring_radius * math.sin(angle)

            # Create small polygon (hexagonal cell) at this position
            cell_vertices = _make_cell_polygon(cx, cy, cell_r, shape.symmetry)
            scaled_cell = [(x * size, y * size) for x, y in cell_vertices]

            polygons.append(
                {
                    "vertices": scaled_cell,
                    "depth": 2,
                    "bit_value": cell_node.bit_value,
                    "polygon_index": cell_idx,
                    "ring_index": ring_idx,
                }
            )

    return polygons


def _make_cell_polygon(
    cx: float, cy: float, radius: float, sides: int = 6
) -> list[tuple[float, float]]:
    """Create a small regular polygon for a data cell.

    Args:
        cx, cy: Center in normalized space.
        radius: Radius in normalized space.
        sides: Number of sides (6 = hexagon for visual consistency).

    Returns:
        List of (x, y) vertex tuples.
    """
    n = min(max(sides, 4), 8)  # Clamp to 4-8 sides
    vertices = []
    for i in range(n):
        angle = (2 * math.pi * i / n) - (math.pi / 2)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices


# ---------------------------------------------------------------------------
# Typed payload encoders (application-level extensions)
# ---------------------------------------------------------------------------
# These encode structured data into typed payloads. The first byte identifies
# the payload type so the decoder can parse fields back out. Applications
# can define additional types by reserving new type bytes (0x05+).


def encode_member_payload(pubkey_hash_prefix: bytes, tier: int = 0) -> str:
    """Encode a MEMBER glyph payload.

    Layout (14 bytes data, 4 bytes CRC appended later = 18 total):
    - 1 byte: glyph type (0x01 = MEMBER)
    - 12 bytes: pubkey hash prefix
    - 1 byte: trust tier (0-3)

    Args:
        pubkey_hash_prefix: First 12 bytes of the public key hash.
        tier: Trust tier (0-3).

    Returns:
        Hex string (28 chars = 14 bytes) ready for encode().

    Raises:
        ValueError: If pubkey_hash_prefix is not 12 bytes or tier is out of range.
    """
    if len(pubkey_hash_prefix) != 12:
        raise ValueError(f"pubkey_hash_prefix must be 12 bytes, got {len(pubkey_hash_prefix)}")
    if not 0 <= tier <= 3:
        raise ValueError(f"tier must be 0-3, got {tier}")

    payload = bytes([GLYPH_TYPE_MEMBER]) + pubkey_hash_prefix + bytes([tier])
    return payload.hex()


def encode_community_payload(
    community_id_prefix: bytes,
    join_requirement: int,
    expiry_timestamp: int,
    presenter_prefix: bytes,
) -> str:
    """Encode a COMMUNITY glyph payload.

    Layout (14 bytes data, 4 bytes CRC appended later = 18 total):
    - 1 byte: glyph type (0x02 = COMMUNITY)
    - 4 bytes: community ID prefix
    - 1 byte: join requirement (0-3)
    - 4 bytes: expiry timestamp (unix, big-endian)
    - 4 bytes: presenter pubkey prefix

    Args:
        community_id_prefix: First 4 bytes of the community ID.
        join_requirement: Join requirement level (0-3).
        expiry_timestamp: Unix timestamp for glyph expiry (big-endian).
        presenter_prefix: First 4 bytes of the presenter's pubkey hash.

    Returns:
        Hex string (28 chars = 14 bytes) ready for encode().

    Raises:
        ValueError: If field sizes are wrong or values out of range.
    """
    if len(community_id_prefix) != 4:
        raise ValueError(f"community_id_prefix must be 4 bytes, got {len(community_id_prefix)}")
    if not 0 <= join_requirement <= 3:
        raise ValueError(f"join_requirement must be 0-3, got {join_requirement}")
    if not 0 <= expiry_timestamp <= 0xFFFFFFFF:
        raise ValueError(
            f"expiry_timestamp must fit in 4 bytes (0-4294967295), got {expiry_timestamp}"
        )
    if len(presenter_prefix) != 4:
        raise ValueError(f"presenter_prefix must be 4 bytes, got {len(presenter_prefix)}")

    payload = (
        bytes([GLYPH_TYPE_COMMUNITY])
        + community_id_prefix
        + bytes([join_requirement])
        + expiry_timestamp.to_bytes(4, "big")
        + presenter_prefix
    )
    return payload.hex()


def encode_recovery_payload(
    pubkey_hash_prefix: bytes,
    recovery_nonce: int,
) -> str:
    """Encode a RECOVERY glyph payload (type 0x04).

    Layout (17 bytes data, 4 bytes CRC appended later = 21 total):
    - 1 byte: glyph type (0x04 = RECOVERY)
    - 12 bytes: pubkey hash prefix
    - 4 bytes: recovery nonce (unique recovery request ID, big-endian)

    Args:
        pubkey_hash_prefix: First 12 bytes of the public key hash.
        recovery_nonce: Unique recovery request identifier (fits in 4 bytes).

    Returns:
        Hex string (34 chars = 17 bytes) ready for encode().

    Raises:
        ValueError: If pubkey_hash_prefix is not 12 bytes.
    """
    if len(pubkey_hash_prefix) != 12:
        raise ValueError(f"pubkey_hash_prefix must be 12 bytes, got {len(pubkey_hash_prefix)}")

    payload = bytes([GLYPH_TYPE_RECOVERY]) + pubkey_hash_prefix + recovery_nonce.to_bytes(4, "big")
    return payload.hex()


def encode_invite_payload(
    community_id_prefix: bytes,
    invite_nonce: bytes,
    inviter_prefix: bytes,
) -> str:
    """Encode an INVITE glyph payload (type 0x03).

    Layout (17 bytes data, 4 bytes CRC appended later = 21 total):
    - 1 byte: glyph type (0x03 = INVITE)
    - 4 bytes: community ID prefix
    - 8 bytes: one-time invitation nonce
    - 4 bytes: inviter's pubkey prefix

    Args:
        community_id_prefix: First 4 bytes of the community ID.
        invite_nonce: 8-byte one-time invitation nonce.
        inviter_prefix: First 4 bytes of the inviter's pubkey hash.

    Returns:
        Hex string (34 chars = 17 bytes) ready for encode().

    Raises:
        ValueError: If field sizes are wrong.
    """
    if len(community_id_prefix) != 4:
        raise ValueError(f"community_id_prefix must be 4 bytes, got {len(community_id_prefix)}")
    if len(invite_nonce) != 8:
        raise ValueError(f"invite_nonce must be 8 bytes, got {len(invite_nonce)}")
    if len(inviter_prefix) != 4:
        raise ValueError(f"inviter_prefix must be 4 bytes, got {len(inviter_prefix)}")

    payload = bytes([GLYPH_TYPE_INVITE]) + community_id_prefix + invite_nonce + inviter_prefix
    return payload.hex()
