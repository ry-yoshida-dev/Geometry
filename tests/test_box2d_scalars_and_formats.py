"""Tests for single-box scalar accessors and center-based format conversions."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import Box2D, Box2dConverter, Box2DFormat


class TestBox2DScalarTypes:
    """Single-box accessors must return Python scalars, not NumPy scalars."""

    @pytest.mark.parametrize(
        ("value", "box_format"),
        [
            (np.array([0, 0, 10, 20]), Box2DFormat.XYWH),
            (np.array([0, 0, 10, 20]), Box2DFormat.XYXY),
        ],
    )
    def test_accessors_return_python_scalars(
        self,
        value: np.ndarray,
        box_format: Box2DFormat,
    ) -> None:
        box = Box2D.register(value=value, box2d_format=box_format)
        for scalar in (box.x1, box.y1, box.x2, box.y2, box.y_max, box.width, box.height, box.area):
            assert type(scalar) in (int, float)


class TestCenterFormatConversion:
    """UVSR is (cx, cy, area, w/h) and UVAH is (cx, cy, w/h, h)."""

    def test_xywh_to_uvsr(self) -> None:
        box = Box2D.register(value=np.array([0.0, 0.0, 10.0, 20.0]), box2d_format=Box2DFormat.XYWH)
        np.testing.assert_allclose(box.to_format(Box2DFormat.UVSR), [5.0, 10.0, 200.0, 0.5])

    def test_xywh_to_uvah(self) -> None:
        box = Box2D.register(value=np.array([0.0, 0.0, 10.0, 20.0]), box2d_format=Box2DFormat.XYWH)
        np.testing.assert_allclose(box.to_format(Box2DFormat.UVAH), [5.0, 10.0, 0.5, 20.0])

    @pytest.mark.parametrize("center_format", [Box2DFormat.UVSR, Box2DFormat.UVAH])
    def test_round_trip_through_center_format(self, center_format: Box2DFormat) -> None:
        original = np.array([[3.0, 4.0, 13.0, 24.0]])
        center = Box2dConverter.convert_format(original, Box2DFormat.XYXY, center_format)
        restored = Box2dConverter.convert_format(center, center_format, Box2DFormat.XYXY)
        np.testing.assert_allclose(restored, original)
