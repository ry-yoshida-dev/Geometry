"""Tests for crop_slice on single and batched 2D boxes."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar.box import Box2D, Box2DFormat, Boxes2D


class TestBox2DCropSlice:
    """Cropping with a single box."""

    def test_xyxy_crop_slice_matches_edges(self) -> None:
        box = Box2D.register(value=np.array([2.0, 3.0, 7.0, 9.0]), box2d_format=Box2DFormat.XYXY)
        assert box.crop_slice == (slice(3, 9), slice(2, 7))

    def test_xywh_crop_slice_crops_expected_shape(self) -> None:
        box = Box2D.register(value=np.array([2, 3, 5, 6]), box2d_format=Box2DFormat.XYWH)
        image = np.zeros((20, 20))
        assert image[box.crop_slice].shape == (6, 5)

    def test_edge_at_origin_is_accepted(self) -> None:
        box = Box2D.register(value=np.array([0.0, 0.0, 4.0, 4.0]), box2d_format=Box2DFormat.XYXY)
        assert box.crop_slice == (slice(0, 4), slice(0, 4))

    @pytest.mark.parametrize(
        "value",
        [
            np.array([-5.0, 0.0, 10.0, 10.0]),
            np.array([0.0, -5.0, 10.0, 10.0]),
            np.array([-5.0, -5.0, 10.0, 10.0]),
        ],
    )
    def test_negative_edge_raises(self, value: np.ndarray) -> None:
        box = Box2D.register(value=value, box2d_format=Box2DFormat.XYXY)
        with pytest.raises(ValueError, match="non-negative"):
            _ = box.crop_slice


class TestBoxes2DCropSlice:
    """Cropping with a batch of boxes."""

    def test_crop_slices_match_each_row(self) -> None:
        boxes = Boxes2D.register(
            value=np.array([[2.0, 3.0, 7.0, 9.0], [0.0, 0.0, 4.0, 4.0]]),
            box2d_format=Box2DFormat.XYXY,
        )
        assert boxes.crop_slice == [
            (slice(3, 9), slice(2, 7)),
            (slice(0, 4), slice(0, 4)),
        ]

    def test_crop_slices_are_python_ints(self) -> None:
        boxes = Boxes2D.register(value=np.array([[2, 3, 5, 6]]), box2d_format=Box2DFormat.XYWH)
        y_slice, x_slice = boxes.crop_slice[0]
        for bound in (y_slice.start, y_slice.stop, x_slice.start, x_slice.stop):
            assert type(bound) is int

    def test_negative_edge_reports_offending_rows(self) -> None:
        boxes = Boxes2D.register(
            value=np.array([[0.0, 0.0, 4.0, 4.0], [-3.0, 1.0, 5.0, 5.0], [1.0, -2.0, 5.0, 5.0]]),
            box2d_format=Box2DFormat.XYXY,
        )
        with pytest.raises(ValueError, match=r"rows \[1, 2\]"):
            _ = boxes.crop_slice
