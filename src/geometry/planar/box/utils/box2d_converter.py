import numpy as np

from ....array_types import NumericArray
from ..format import Box2DFormat
from .converters import Box2dConverters


class Box2dConverter:
    """
    A utility class for converting bounding box formats using NumPy for batch processing.

    Format aliases (e.g. TLBR, CXCYWH) are standardized before conversion, and
    every pair of formats is supported by routing through the XYWH layout.
    """

    @staticmethod
    def convert_format(
        boxes: NumericArray,
        input_format: Box2DFormat,
        output_format: Box2DFormat,
        as_int: bool = False,
    ) -> NumericArray:
        """
        Convert bounding boxes from one format to another.

        Integer inputs are promoted to float64 before conversion; floating
        inputs keep their dtype so no precision is lost.

        Parameters
        ----------
        boxes : NumericArray
            Input bounding boxes with shape (..., 4).
        input_format : Box2DFormat
            Format of input boxes. Aliases are accepted.
        output_format : Box2DFormat
            Desired output format. Aliases are accepted.
        as_int : bool
            Whether to return results cast to int32. Defaults to False.

        Returns
        -------
        NumericArray
            Converted bounding boxes with the same leading shape as boxes.

        Examples
        --------
        >>> boxes = np.array([[10, 20, 30, 40]])
        >>> Box2dConverter.convert_format(boxes, Box2DFormat.XYWH, Box2DFormat.XYXY)
        array([[10., 20., 40., 60.]])
        """
        source_format = input_format.standardize_format()
        target_format = output_format.standardize_format()

        if source_format == target_format:
            result = np.asarray(boxes)
        else:
            float_boxes = Box2dConverter._as_float(boxes)
            xywh_boxes = Box2dConverter._to_xywh(float_boxes, source_format)
            result = Box2dConverter._from_xywh(xywh_boxes, target_format)

        return result.astype(np.int32) if as_int else result

    @staticmethod
    def _as_float(boxes: NumericArray) -> NumericArray:
        """
        Promote integer boxes to float64 while keeping floating dtypes intact.

        Parameters
        ----------
        boxes : NumericArray
            Input bounding boxes.

        Returns
        -------
        NumericArray
            Floating-point view or copy of boxes.
        """
        array = np.asarray(boxes)
        if np.issubdtype(array.dtype, np.floating):
            return array
        return array.astype(np.float64)

    @staticmethod
    def _to_xywh(boxes: NumericArray, source_format: Box2DFormat) -> NumericArray:
        """
        Convert boxes in a canonical format to XYWH.

        Parameters
        ----------
        boxes : NumericArray
            Input bounding boxes in source_format.
        source_format : Box2DFormat
            Canonical format of boxes.

        Returns
        -------
        NumericArray
            Boxes in XYWH format.

        Raises
        ------
        ValueError
            If source_format is not a canonical format.
        """
        match source_format:
            case Box2DFormat.XYWH:
                return boxes
            case Box2DFormat.XYXY:
                return Box2dConverters.xyxy2xywh(boxes)
            case Box2DFormat.UVWH:
                return Box2dConverters.uvwh2xywh(boxes)
            case Box2DFormat.UVAH:
                return Box2dConverters.uvah2xywh(boxes)
            case Box2DFormat.UVSR:
                return Box2dConverters.uvsr2xywh(boxes)
            case _:
                raise ValueError(f"Unsupported source format: {source_format}")

    @staticmethod
    def _from_xywh(boxes: NumericArray, target_format: Box2DFormat) -> NumericArray:
        """
        Convert XYWH boxes to a canonical target format.

        Parameters
        ----------
        boxes : NumericArray
            Input bounding boxes in XYWH format.
        target_format : Box2DFormat
            Canonical format to convert to.

        Returns
        -------
        NumericArray
            Boxes in target_format.

        Raises
        ------
        ValueError
            If target_format is not a canonical format.
        """
        match target_format:
            case Box2DFormat.XYWH:
                return boxes
            case Box2DFormat.XYXY:
                return Box2dConverters.xywh2xyxy(boxes)
            case Box2DFormat.UVWH:
                return Box2dConverters.xywh2uvwh(boxes)
            case Box2DFormat.UVAH:
                return Box2dConverters.xywh2uvah(boxes)
            case Box2DFormat.UVSR:
                return Box2dConverters.xywh2uvsr(boxes)
            case _:
                raise ValueError(f"Unsupported target format: {target_format}")

    @staticmethod
    def add_dimension(box: NumericArray) -> NumericArray:
        """
        Add an extra dimension to the array, converting it to a column vector.
        
        This utility method reshapes a 1D array into a 2D column vector,
        useful for certain matrix operations.
        
        Parameters:
        ----------
        box: NumericArray
            Input array or list
            
        Returns:
        ---------
        NumericArray: Reshaped array with shape (N, 1)
            
        Returns:
        ---------
        NumericArray: Reshaped array with shape (N, 1)
            
        Example:
            >>> box = [1, 2, 3, 4]
            >>> result = Box2dConverter.add_dimension(box)
            >>> print(result.shape)  # (4, 1)
        """
        box = np.asarray(box)
        return box.reshape((-1, 1))
