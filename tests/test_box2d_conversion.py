"""Tests for format conversion coverage, dtype handling, centers, and aspect ratios."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from geometry.planar import Box2D, Box2dConverter, Box2DFormat
from geometry.planar.box.boxes import Boxes2D
from geometry.planar.box.utils.box2d_calculator import BboxCalculator

XYXY_BOXES: np.ndarray = np.array([[3.0, 4.0, 13.0, 24.0], [0.0, 0.0, 5.0, 2.0]])
CENTERS: np.ndarray = np.array([[8.0, 14.0], [2.5, 1.0]])


class TestConvertFormat:
    """Every pair of formats, including aliases, must round-trip."""

    @pytest.mark.parametrize(
        ("source_format", "target_format"),
        list(itertools.product(list(Box2DFormat), repeat=2)),
    )
    def test_round_trip_all_pairs(self, source_format: Box2DFormat, target_format: Box2DFormat) -> None:
        source = Box2dConverter.convert_format(XYXY_BOXES, Box2DFormat.XYXY, source_format)
        target = Box2dConverter.convert_format(source, source_format, target_format)
        restored = Box2dConverter.convert_format(target, target_format, Box2DFormat.XYXY)
        np.testing.assert_allclose(restored, XYXY_BOXES)

    def test_float64_precision_is_preserved(self) -> None:
        boxes = np.array([[1e8 + 0.25, 0.0, 1e8 + 10.75, 5.0]])
        converted = Box2dConverter.convert_format(boxes, Box2DFormat.XYXY, Box2DFormat.XYWH)
        assert converted.dtype == np.float64
        np.testing.assert_allclose(converted[0, 2], 10.5)

    def test_integer_input_is_promoted_to_float64(self) -> None:
        boxes = np.array([[0, 0, 3, 3]])
        converted = Box2dConverter.convert_format(boxes, Box2DFormat.XYXY, Box2DFormat.UVWH)
        assert converted.dtype == np.float64
        np.testing.assert_allclose(converted, [[1.5, 1.5, 3.0, 3.0]])

    def test_as_int_applies_to_same_format(self) -> None:
        boxes = np.array([[0.4, 0.6, 3.2, 3.9]])
        converted = Box2dConverter.convert_format(boxes, Box2DFormat.XYXY, Box2DFormat.TLBR, as_int=True)
        assert converted.dtype == np.int32


class TestBoxCenters:
    """get_box_centers must support every format."""

    @pytest.mark.parametrize("box_format", list(Box2DFormat))
    def test_centers(self, box_format: Box2DFormat) -> None:
        boxes = Box2dConverter.convert_format(XYXY_BOXES, Box2DFormat.XYXY, box_format)
        np.testing.assert_allclose(BboxCalculator.get_box_centers(boxes, box_format), CENTERS)


class TestAspectRatio:
    """Aspect ratio is width / height, consistent with UVAH and UVSR."""

    def test_single_box(self) -> None:
        box = Box2D.register(value=np.array([0.0, 0.0, 10.0, 20.0]), box2d_format=Box2DFormat.XYWH)
        assert box.aspect_ratio == pytest.approx(0.5)
        assert box.aspect_ratio == pytest.approx(box.to_format(Box2DFormat.UVAH)[2])

    def test_batch(self) -> None:
        boxes = Boxes2D.register(value=XYXY_BOXES, box2d_format=Box2DFormat.XYXY)
        np.testing.assert_allclose(boxes.aspect_ratio, [0.5, 2.5])
        np.testing.assert_allclose(BboxCalculator.measure_aspect_ratios(XYXY_BOXES), [0.5, 2.5])
