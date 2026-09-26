"""Tests for GeometryRotationMatrix composition."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.utils.rotation import GeometryRotationMatrix
from handedness_rotation import IntrinsicRotationOrder, RotationMatrix
from scipy.spatial.transform import Rotation

X_RADIAN: float = 0.3
Y_RADIAN: float = -0.7
Z_RADIAN: float = 1.1


class TestGeometryRotationMatrix:
    """Composite rotations must follow the intrinsic convention."""

    def test_returns_rotation_matrix(self) -> None:
        matrix = GeometryRotationMatrix.from_xyz_angles(X_RADIAN, Y_RADIAN, Z_RADIAN)
        assert isinstance(matrix, RotationMatrix)

    @pytest.mark.parametrize("order", list(IntrinsicRotationOrder))
    def test_matches_scipy_intrinsic_euler(self, order: IntrinsicRotationOrder) -> None:
        angles: dict[str, float] = {"X": X_RADIAN, "Y": Y_RADIAN, "Z": Z_RADIAN}
        matrix = GeometryRotationMatrix.from_xyz_angles(X_RADIAN, Y_RADIAN, Z_RADIAN, order=order)
        expected = Rotation.from_euler(order.value, [angles[axis] for axis in order.value]).as_matrix()
        np.testing.assert_allclose(matrix.value, expected, atol=1e-12)
