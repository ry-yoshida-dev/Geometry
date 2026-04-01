import numpy as np

from cartesian_axis import CoordinateHandedness

from handedness_rotation import (
    HandednessRotationMatrix, 
    IntrinsicRotationOrder,
)

class GeometryRotationMatrix:
    @staticmethod
    def register_from_x_radian(
        x_radian: float,
        coordinate_system: CoordinateHandedness = CoordinateHandedness.RIGHT
        ) -> HandednessRotationMatrix:
        """
        Register a rotation matrix for a rotation around the X-axis.

        Parameters
        ----------
        x_radian : float
            Rotation angle in radians around the X-axis.
        coordinate_system : CoordinateHandedness
            Handedness of the coordinate system for the rotation matrix.

        Returns
        -------
        HandednessRotationMatrix
            Rotation matrix.
        """
        cos = np.cos(x_radian)
        sin = np.sin(x_radian)
        match coordinate_system:
            case CoordinateHandedness.RIGHT:
                value = np.array([[1, 0, 0],
                                [0, cos, -sin],
                                [0, sin,  cos]])
            case CoordinateHandedness.LEFT:
                value = np.array([[1, 0, 0],
                                [0, cos, sin],
                                [0, -sin,  cos]])

        return HandednessRotationMatrix(
            value=value, 
            coordinate_handedness=coordinate_system
            )

    @staticmethod
    def register_from_y_radian(
        y_radian: float,
        coordinate_system: CoordinateHandedness = CoordinateHandedness.RIGHT
        ) -> HandednessRotationMatrix:
        """
        Register a rotation matrix for a rotation around the Y-axis.
        
        Parameters
        ----------
        y_radian : float
            Rotation angle in radians around the Y-axis.
        coordinate_system : CoordinateHandedness
            Handedness of the coordinate system for the rotation matrix.

        Returns
        -------
        HandednessRotationMatrix
            Rotation matrix.
        """
        cos = np.cos(y_radian)
        sin = np.sin(y_radian)
        match coordinate_system:
            case CoordinateHandedness.RIGHT:
                value = np.array([[cos, 0, sin],
                                [0, 1, 0],
                                [-sin, 0, cos]])
            case CoordinateHandedness.LEFT:
                value = np.array([[cos, 0, -sin],
                                [0, 1, 0],
                                [sin, 0, cos]])

        return HandednessRotationMatrix(
            value=value, 
            coordinate_handedness=coordinate_system
            )

    @staticmethod
    def register_from_z_radian(
        z_radian: float,
        coordinate_system: CoordinateHandedness = CoordinateHandedness.RIGHT
        ) -> HandednessRotationMatrix:
        """
        Register a rotation matrix for a rotation around the Z-axis.
        
        Parameters
        ----------
        z_radian : float
            Rotation angle in radians around the Z-axis.
        coordinate_system : CoordinateHandedness
            Handedness of the coordinate system for the rotation matrix.

        Returns
        -------
        HandednessRotationMatrix
            Rotation matrix.
        """
        cos = np.cos(z_radian)
        sin = np.sin(z_radian)
        match coordinate_system:
            case CoordinateHandedness.RIGHT:
                value = np.array([[cos, -sin, 0],
                                [sin, cos, 0],
                                [0, 0, 1]])
            case CoordinateHandedness.LEFT:
                value = np.array([[cos, sin, 0],
                                [-sin, cos, 0],
                                [0, 0, 1]])
        return HandednessRotationMatrix(
            value=value, 
            coordinate_handedness=coordinate_system
            )

    @staticmethod
    def from_Rx_Ry_Rz(
        Rx: HandednessRotationMatrix, 
        Ry: HandednessRotationMatrix, 
        Rz: HandednessRotationMatrix, 
        order: IntrinsicRotationOrder = IntrinsicRotationOrder.XYZ,
        coordinate_system: CoordinateHandedness = CoordinateHandedness.RIGHT
        ):
        """
        Create a composite rotation matrix from individual rotation matrices.
        
        This method implements INTRINSIC rotation, where each rotation is applied
        relative to the coordinate system that results from the previous rotations.
        This is the standard approach for camera poses and robot joint rotations.

        Parameters
        ----------
        Rx : HandednessRotationMatrix
            Rotation around the X-axis.
        Ry : HandednessRotationMatrix
            Rotation around the Y-axis.
        Rz : HandednessRotationMatrix
            Rotation around the Z-axis.
        order : IntrinsicRotationOrder
            Intrinsic order of rotation application (default: ``XYZ``).
        coordinate_system : CoordinateHandedness
            Handedness recorded on the resulting matrix.

        Returns
        -------
        HandednessRotationMatrix
            Composite intrinsic rotation matrix.
        """
        rot_map = {"X": Rx.value, "Y": Ry.value, "Z": Rz.value}
        value = np.eye(3)
        for axis in order.value:
            value = rot_map[axis] @ value
        return HandednessRotationMatrix(value=value, coordinate_handedness=coordinate_system)

    @classmethod
    def from_xyz_angles(
        cls, 
        x_radian: float, 
        y_radian: float, 
        z_radian: float, 
        order: IntrinsicRotationOrder = IntrinsicRotationOrder.ZXY,
        coordinate_system: CoordinateHandedness = CoordinateHandedness.RIGHT
        ) -> HandednessRotationMatrix:
        """
        Create a composite rotation matrix from x, y, and z angles.

        Parameters
        ----------
        x_radian : float
            X-axis angle in radians.
        y_radian : float
            Y-axis angle in radians.
        z_radian : float
            Z-axis angle in radians.
        order : IntrinsicRotationOrder
            Intrinsic order of rotation application (default: ``ZXY``).
        coordinate_system : CoordinateHandedness
            Handedness of the coordinate system for each elementary rotation.

        Returns
        -------
        HandednessRotationMatrix
            Composite rotation matrix.
        """
        Rx = GeometryRotationMatrix.register_from_x_radian(
            x_radian=x_radian, 
            coordinate_system=coordinate_system
            )
        Ry = GeometryRotationMatrix.register_from_y_radian(
            y_radian=y_radian, 
            coordinate_system=coordinate_system
            )
        Rz = GeometryRotationMatrix.register_from_z_radian(
            z_radian=z_radian, 
            coordinate_system=coordinate_system
            )
        return GeometryRotationMatrix.from_Rx_Ry_Rz(
            Rx=Rx, 
            Ry=Ry, 
            Rz=Rz, 
            order=order, 
            coordinate_system=coordinate_system
            )
