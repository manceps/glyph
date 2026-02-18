"""Tree data structures for Glyph topology encoding.

A Glyph encodes data as a hierarchy of polygons arranged in
concentric rings. Each leaf node represents a single bit, with its
fill (colored or white) encoding 1 or 0.

The tree structure organizes cells into rings for systematic encoding
and decoding: root -> ring nodes -> cell nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A node in the topology tree.

    Each node represents a polygon region in the rendered glyph.
    Children are nested polygons contained within the parent.

    Attributes:
        bit_value: The bit this node encodes (0 or 1). None for root/ring nodes.
        children: Child nodes (nested polygons).
        depth: Depth in the tree (0 = root).
        polygon_index: Index into the shape's polygon template.
    """

    bit_value: int | None = None
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0
    polygon_index: int = 0

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0

    def leaf_count(self) -> int:
        """Count total leaves in this subtree."""
        if self.is_leaf:
            return 1
        return sum(child.leaf_count() for child in self.children)

    def max_depth(self) -> int:
        """Maximum depth in this subtree."""
        if self.is_leaf:
            return self.depth
        return max(child.max_depth() for child in self.children)


@dataclass
class TopologyTree:
    """Complete topology tree for a Glyph.

    The tree encodes a sequence of bits as polygons arranged in
    concentric rings within a boundary shape. The root node is
    the outer boundary of the glyph.

    Attributes:
        root: Root node (outermost polygon).
        bit_length: Number of data bits encoded (excluding CRC).
        crc_bits: Number of CRC check bits appended.
    """

    root: TreeNode = field(default_factory=TreeNode)
    bit_length: int = 0
    crc_bits: int = 8  # CRC-8

    @property
    def total_bits(self) -> int:
        """Total bits including CRC."""
        return self.bit_length + self.crc_bits

    @property
    def total_nodes(self) -> int:
        """Total nodes in the tree."""
        return self._count_nodes(self.root)

    def _count_nodes(self, node: TreeNode) -> int:
        """Recursively count nodes."""
        return 1 + sum(self._count_nodes(c) for c in node.children)
