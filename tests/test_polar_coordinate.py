"""Tests for PolarCoordinate conversion to and from Cartesian (u, v)."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import PolarCoordinate
from units import Angle, AngleUnit


@pytest.fixture
def polar() -> PolarCoordinate:
    """Radius 2 at 0, 90, and 180 degrees."""
    return PolarCoordinate(
        radius=np.array([2.0, 2.0, 2.0]),
        angle=Angle(value=np.array([0.0, 90.0, 180.0]), unit=AngleUnit.DEGREE),
    )


class TestPolarCoordinateValidation:
    """Constructor validation."""

    def test_non_1d_radius_raises(self) -> None:
        with pytest.raises(ValueError):
            PolarCoordinate(
                radius=np.array([[1.0, 2.0]]),
                angle=Angle(value=np.array([0.0, 1.0]), unit=AngleUnit.RADIAN),
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            PolarCoordinate(
                radius=np.array([1.0, 2.0]),
                angle=Angle(value=np.array([0.0]), unit=AngleUnit.RADIAN),
            )


class TestPolarCoordinateConversion:
    """Conversion between polar and Cartesian components."""

    def test_len(self, polar: PolarCoordinate) -> None:
        assert len(polar) == 3

    def test_angle_units(self, polar: PolarCoordinate) -> None:
        np.testing.assert_allclose(polar.degree, [0.0, 90.0, 180.0])
        np.testing.assert_allclose(polar.radian, [0.0, np.pi / 2, np.pi])

    def test_uv(self, polar: PolarCoordinate) -> None:
        expected = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
        np.testing.assert_allclose(polar.u, expected[:, 0], atol=1e-12)
        np.testing.assert_allclose(polar.v, expected[:, 1], atol=1e-12)
        np.testing.assert_allclose(polar.uv, expected, atol=1e-12)

    def test_from_uv(self) -> None:
        polar = PolarCoordinate.from_uv(u=np.array([3.0, 0.0, -1.0]), v=np.array([4.0, -2.0, 0.0]))
        np.testing.assert_allclose(polar.radius, [5.0, 2.0, 1.0])
        np.testing.assert_allclose(polar.degree, [np.degrees(np.arctan2(4.0, 3.0)), -90.0, 180.0])

    def test_round_trip(self) -> None:
        generator = np.random.default_rng(0)
        u = generator.uniform(-10.0, 10.0, size=20)
        v = generator.uniform(-10.0, 10.0, size=20)
        polar = PolarCoordinate.from_uv(u=u, v=v)
        np.testing.assert_allclose(polar.u, u)
        np.testing.assert_allclose(polar.v, v)
