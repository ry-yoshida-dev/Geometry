"""Tests for NormalMap and NormalValue."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.spatial import CartesianCoordinateSystem, NormalMap, NormalValue, SoftwareCoordinateSystem


@pytest.fixture
def z_up_system() -> CartesianCoordinateSystem:
    """Coordinate system whose up axis is Z (index 2)."""
    return SoftwareCoordinateSystem.PLOTLY.coordinate_system


@pytest.fixture
def normal_map(z_up_system: CartesianCoordinateSystem) -> NormalMap:
    """A 1x3 map: floor-facing, wall-facing, and slanted normals."""
    slanted = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)])
    value = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], slanted]])
    return NormalMap(value=value, coordinate_system=z_up_system)


class TestNormalMapValidation:
    """Constructor validation."""

    @pytest.mark.parametrize("shape", [(2, 3), (2, 2, 2)])
    def test_invalid_shape_raises(self, z_up_system: CartesianCoordinateSystem, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            NormalMap(value=np.zeros(shape), coordinate_system=z_up_system)

    def test_out_of_range_raises(self, z_up_system: CartesianCoordinateSystem) -> None:
        with pytest.raises(ValueError):
            NormalMap(value=np.full((1, 1, 3), 1.5), coordinate_system=z_up_system)


class TestNormalMapMasks:
    """Plane classification by the up component."""

    def test_horizontal_plane_mask(self, normal_map: NormalMap) -> None:
        np.testing.assert_array_equal(normal_map.horizontal_plane_mask, [[True, False, False]])

    def test_vertical_plane_mask(self, normal_map: NormalMap) -> None:
        np.testing.assert_array_equal(normal_map.vertical_plane_mask, [[False, True, False]])

    def test_mask_follows_coordinate_system_up_axis(self) -> None:
        y_up_system = SoftwareCoordinateSystem.OPENCV.coordinate_system
        value = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        normal_map = NormalMap(value=value, coordinate_system=y_up_system)
        np.testing.assert_array_equal(normal_map.horizontal_plane_mask, [[True, False]])


class TestNormalMapConversion:
    """Conversion between [-1, 1] and [0, 1] encodings."""

    def test_zero_one_normal_map(self, normal_map: NormalMap) -> None:
        encoded = normal_map.zero_one_normal_map
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0
        np.testing.assert_allclose(encoded[0, 0], [0.5, 0.5, 1.0])

    def test_round_trip(self, normal_map: NormalMap, z_up_system: CartesianCoordinateSystem) -> None:
        decoded = NormalMap.from_zero_one_normal_map(normal_map.zero_one_normal_map, coordinate_system=z_up_system)
        np.testing.assert_allclose(decoded.value, normal_map.value, atol=1e-12)

    def test_from_zero_one_normalizes(self, z_up_system: CartesianCoordinateSystem) -> None:
        unnormalized = np.array([[[0.5, 0.5, 0.75]]])
        decoded = NormalMap.from_zero_one_normal_map(unnormalized, coordinate_system=z_up_system)
        np.testing.assert_allclose(np.linalg.norm(decoded.value, axis=2), 1.0)

    def test_from_zero_one_handles_zero_vector(self, z_up_system: CartesianCoordinateSystem) -> None:
        decoded = NormalMap.from_zero_one_normal_map(np.full((1, 1, 3), 0.5), coordinate_system=z_up_system)
        np.testing.assert_allclose(decoded.value, 0.0)

    def test_from_zero_one_out_of_range_raises(self, z_up_system: CartesianCoordinateSystem) -> None:
        with pytest.raises(ValueError):
            NormalMap.from_zero_one_normal_map(np.full((1, 1, 3), -0.1), coordinate_system=z_up_system)


class TestNormalValue:
    """Pixel access and validation."""

    def test_getitem_returns_normal_value(self, normal_map: NormalMap) -> None:
        normal_value = normal_map[0, 1]
        assert isinstance(normal_value, NormalValue)
        np.testing.assert_allclose(normal_value.value, [1.0, 0.0, 0.0])
        assert normal_value.coordinate_system is normal_map.coordinate_system

    def test_invalid_shape_raises(self, z_up_system: CartesianCoordinateSystem) -> None:
        with pytest.raises(ValueError):
            NormalValue(value=np.array([0.0, 1.0]), coordinate_system=z_up_system)

    def test_out_of_range_raises(self, z_up_system: CartesianCoordinateSystem) -> None:
        with pytest.raises(ValueError):
            NormalValue(value=np.array([0.0, 0.0, 2.0]), coordinate_system=z_up_system)
