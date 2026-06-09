from __future__ import annotations
from ...array_types import NumericArray
import numpy as np
from dataclasses import dataclass

from units import Angle, AngleUnit

@dataclass
class PolarCoordinate:
    """
    Polar coordinate (r, theta) in polar coordinate system.
    
    Parameters
    ----------
    radius: NumericArray
        The radius of the polar coordinate with shape (n,).
    angle: Angle
        The angle of the polar coordinate with shape (n,).
    """
    radius: NumericArray
    angle: Angle

    def __post_init__(self):
        """
        Post-init validation.

        Raises
        ------
        ValueError: If the radius is not a 1D array.
        ValueError: If the radius and angle have different lengths.
        """
        if self.radius.ndim != 1:
            raise ValueError("radius must be a 1D array")
        if len(self.radius) != len(self.angle):
            raise ValueError("radius and angle must have the same length")

    @property
    def u(self) -> NumericArray:
        """
        The u coordinate.
        
        Returns
        -------
        NumericArray:
            The u coordinate
        """
        return self.radius * np.cos(self.radian)
    
    @property
    def v(self) -> NumericArray:
        """
        The v coordinate.
        
        Returns
        -------
        NumericArray:
            The v coordinate
        """
        return self.radius * np.sin(self.radian)

    @property
    def uv(self) -> NumericArray:
        """
        The uv coordinate.
        
        Returns
        -------
        NumericArray:
            The uv coordinate
        """
        rad = self.radian
        return np.stack([self.radius * np.cos(rad), self.radius * np.sin(rad)], axis=-1)

    @property
    def radian(self) -> NumericArray:
        """
        The angle in radians.
        
        Returns
        -------
        NumericArray:
            The angle in radians.
        """
        return self.angle.radian
    
    @property
    def degree(self) -> NumericArray:
        """
        The angle in degrees.
        
        Returns
        -------
        NumericArray:
            The angle in degrees.
        """
        return self.angle.degree

    def __len__(self) -> int:
        """
        Return the length of the polar coordinate.
        
        Returns
        -------
        int:
            The length of the polar coordinate.
        """
        return len(self.radius)

    @classmethod
    def from_uv(
        cls, 
        u: NumericArray, 
        v: NumericArray
        ) -> PolarCoordinate:
        """
        Create a polar coordinate from u and v.
        
        Parameters
        ----------
        u: NumericArray
            The u coordinate.
        v: NumericArray
            The v coordinate.

        Returns
        -------
        PolarCoordinate:
            The polar coordinate.
        """
        radius = np.sqrt(u**2 + v**2)
        angle_array = np.arctan2(v, u)
        angle = Angle(value=angle_array, unit=AngleUnit.RADIAN)
        return cls(radius=radius, angle=angle)
    
