"""Tests for Boxes2D batch length and indexing."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.planar.box import Box2D, Box2DFormat, Boxes2D


@pytest.fixture
def xyxy_boxes() -> Boxes2D:
    """Two valid XYXY boxes."""
    value = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    return Boxes2D.register(value=value, box2d_format=Box2DFormat.XYXY)


class TestBoxes2DLen:
    """Length of a Boxes2D batch."""

    def test_len_returns_row_count(self, xyxy_boxes: Boxes2D) -> None:
        assert len(xyxy_boxes) == 2


class TestBoxes2DGetItem:
    """Indexing and slicing Boxes2D rows."""

    def test_int_index_returns_box2d(self, xyxy_boxes: Boxes2D) -> None:
        box = xyxy_boxes[0]
        assert isinstance(box, Box2D)
        assert box.box_format == Box2DFormat.XYXY
        np.testing.assert_array_equal(box.value, np.array([10.0, 20.0, 30.0, 40.0]))

    def test_slice_returns_boxes2d_same_type(self, xyxy_boxes: Boxes2D) -> None:
        subset = xyxy_boxes[1:]
        assert type(subset) is type(xyxy_boxes)
        assert len(subset) == 1
        np.testing.assert_array_equal(subset.value, np.array([[50.0, 60.0, 70.0, 80.0]]))

    def test_negative_index_returns_last_box(self, xyxy_boxes: Boxes2D) -> None:
        box = xyxy_boxes[-1]
        assert isinstance(box, Box2D)
        np.testing.assert_array_equal(box.value, np.array([50.0, 60.0, 70.0, 80.0]))

    def test_xywh_format_preserved_on_index(self) -> None:
        value = np.array([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 15.0, 25.0]])
        boxes = Boxes2D.register(value=value, box2d_format=Box2DFormat.XYWH)
        box = boxes[0]
        subset = boxes[:1]
        assert box.box_format == Box2DFormat.XYWH
        assert type(subset) is type(boxes)
        assert len(subset) == 1
