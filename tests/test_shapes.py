"""Tests for shape templates."""

import pytest

from src.shapes import SHAPE_INDEX, SHAPES, select_shape


class TestShapes:
    def test_all_shapes_have_vertices(self):
        for shape in SHAPES:
            assert len(shape.vertices) >= 3

    def test_all_shapes_in_index(self):
        assert len(SHAPE_INDEX) == len(SHAPES)

    def test_select_known_shape(self):
        shape = select_shape("hexagon")
        assert shape.name == "hexagon"
        assert shape.symmetry == 6

    def test_select_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            select_shape("star")

    def test_circle_is_32gon(self):
        circle = select_shape("circle")
        assert len(circle.vertices) == 32

    def test_all_vertices_in_unit_space(self):
        for shape in SHAPES:
            for x, y in shape.vertices:
                assert 0.0 <= x <= 1.0, f"{shape.name}: x={x} out of [0,1]"
                assert 0.0 <= y <= 1.0, f"{shape.name}: y={y} out of [0,1]"

    def test_shape_names(self):
        expected = {
            "circle",
            "hexagon",
            "octagon",
            "shield",
            "diamond",
            "pentagon",
            "heptagon",
            "triangle",
        }
        assert set(SHAPE_INDEX.keys()) == expected
