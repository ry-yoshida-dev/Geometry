from enum import Enum

class CoordinateType(Enum):
    """
    CoordinateType is an enum that represents the type of coordinate system.

    Attributes:
    ----------
    CARTESIAN: str
        Cartesian coordinate system.
    SPHERICAL: str
        Spherical coordinate system.
    CYLINDRICAL: str
        Cylindrical coordinate system.
    """
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"

