"""Shape templates for Glyph boundaries.

Each shape defines the outermost polygon boundary of a glyph.
The shape can be deterministically selected from a data hash.

Shapes are defined as normalized polygon vertices in [0, 1] space
and scaled to the requested output size during rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeTemplate:
    """A shape template for glyph boundaries.

    Attributes:
        name: Human-readable shape name.
        vertices: List of (x, y) tuples in normalized [0, 1] space.
        child_regions: Number of regions available for child polygons.
        symmetry: Rotational symmetry order (e.g. 6 for hexagon).
    """

    name: str
    vertices: list[tuple[float, float]]
    child_regions: int
    symmetry: int


def _regular_polygon(
    n: int,
    radius: float = 0.45,
    center: tuple[float, float] = (0.5, 0.5),
) -> list[tuple[float, float]]:
    """Generate vertices for a regular n-gon."""
    cx, cy = center
    vertices = []
    for i in range(n):
        angle = (2 * math.pi * i / n) - (math.pi / 2)  # Start from top
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices


def _shield_shape(
    radius: float = 0.45,
    center: tuple[float, float] = (0.5, 0.5),
) -> list[tuple[float, float]]:
    """Generate a shield/badge shape (pointed bottom)."""
    cx, cy = center
    return [
        (cx - radius, cy - radius * 0.6),  # top-left
        (cx + radius, cy - radius * 0.6),  # top-right
        (cx + radius, cy + radius * 0.3),  # mid-right
        (cx, cy + radius),  # bottom point
        (cx - radius, cy + radius * 0.3),  # mid-left
    ]


def _diamond_shape(
    radius: float = 0.42,
    center: tuple[float, float] = (0.5, 0.5),
) -> list[tuple[float, float]]:
    """Generate a diamond/rhombus shape."""
    cx, cy = center
    return [
        (cx, cy - radius),  # top
        (cx + radius, cy),  # right
        (cx, cy + radius),  # bottom
        (cx - radius, cy),  # left
    ]


# Available shape templates, indexed by selection byte
SHAPES: list[ShapeTemplate] = [
    ShapeTemplate(
        name="circle",
        vertices=_regular_polygon(32),  # Approximate circle with 32-gon
        child_regions=8,
        symmetry=32,
    ),
    ShapeTemplate(
        name="hexagon",
        vertices=_regular_polygon(6),
        child_regions=6,
        symmetry=6,
    ),
    ShapeTemplate(
        name="octagon",
        vertices=_regular_polygon(8),
        child_regions=8,
        symmetry=8,
    ),
    ShapeTemplate(
        name="shield",
        vertices=_shield_shape(),
        child_regions=5,
        symmetry=1,
    ),
    ShapeTemplate(
        name="diamond",
        vertices=_diamond_shape(),
        child_regions=4,
        symmetry=2,
    ),
    ShapeTemplate(
        name="pentagon",
        vertices=_regular_polygon(5),
        child_regions=5,
        symmetry=5,
    ),
    ShapeTemplate(
        name="heptagon",
        vertices=_regular_polygon(7),
        child_regions=7,
        symmetry=7,
    ),
    ShapeTemplate(
        name="triangle",
        vertices=_regular_polygon(3),
        child_regions=3,
        symmetry=3,
    ),
]

SHAPE_INDEX: dict[str, ShapeTemplate] = {s.name: s for s in SHAPES}


def select_shape(shape_name: str) -> ShapeTemplate:
    """Select a shape template by name.

    Args:
        shape_name: Shape name (circle, hexagon, octagon, shield, diamond,
                     pentagon, heptagon, triangle).

    Returns:
        ShapeTemplate for the requested shape.

    Raises:
        ValueError: If shape_name is not recognized.
    """
    if shape_name not in SHAPE_INDEX:
        valid = ", ".join(SHAPE_INDEX.keys())
        raise ValueError(f"Unknown shape '{shape_name}'. Valid shapes: {valid}")
    return SHAPE_INDEX[shape_name]
