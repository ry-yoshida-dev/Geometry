import numpy as np
from ....array_types import NumericArray

from ..format import Box2DFormat
from .converters import Box2dConverters

class Box2dConverter:
    """
    A utility class for converting bounding box formats using NumPy for batch processing.
    
    This class provides a high-level interface for converting bounding boxes between
    different coordinate representations. It supports batch processing and handles
    format standardization automatically.
    
    Attributes:
        conversion_map (dict): Mapping of (input_format, output_format) tuples to
            conversion functions from DetectionConverters class.
    """
    
    # Mapping of (input_format, output_format) to conversion functions
    conversion_map = {
        (Box2DFormat.XYWH, Box2DFormat.XYXY): Box2dConverters.xywh2xyxy,
        (Box2DFormat.XYWH, Box2DFormat.UVAH): Box2dConverters.xywh2uvah,
        (Box2DFormat.XYWH, Box2DFormat.UVWH): Box2dConverters.xywh2uvwh,
        (Box2DFormat.XYWH, Box2DFormat.UVSR): Box2dConverters.xywh2uvsr,
        (Box2DFormat.XYXY, Box2DFormat.XYWH): Box2dConverters.xyxy2xywh,
        (Box2DFormat.XYXY, Box2DFormat.UVAH): Box2dConverters.xyxy2uvah,
        (Box2DFormat.XYXY, Box2DFormat.UVSR): Box2dConverters.xyxy2uvsr,
        (Box2DFormat.UVAH, Box2DFormat.XYWH): Box2dConverters.uvah2xywh,
        (Box2DFormat.UVAH, Box2DFormat.XYXY): Box2dConverters.uvah2xyxy,
        (Box2DFormat.UVSR, Box2DFormat.XYXY): Box2dConverters.uvsr2xyxy,
        (Box2DFormat.UVSR, Box2DFormat.XYWH): Box2dConverters.uvsr2xywh,
        (Box2DFormat.UVWH, Box2DFormat.XYWH): Box2dConverters.uvwh2xywh,
        }

    @staticmethod
    def convert_format(
        boxes: NumericArray, 
        input_format: Box2DFormat, 
        output_format: Box2DFormat, 
        as_int: bool = False) -> NumericArray:
        """
        Convert bounding boxes from one format to another.
        
        This method handles the conversion between different bounding box formats.
        It automatically standardizes format names and applies the appropriate
        conversion function.
        
        Parameters:
        ----------
        boxes: NumericArray
            Input bounding boxes with shape (N, 4)
        input_format: Box2DFormat
            Format of input boxes
        output_format: Box2DFormat
            Desired output format
        as_int: bool
            Whether to return results as integers. Defaults to False.
            
        Returns:
        ---------
        NumericArray: Converted bounding boxes in the specified output format
            
        Raises:
        ---------
        ValueError: If the conversion is not supported or format is invalid
            
        Example:
            >>> boxes = np.array([[10, 20, 30, 40]])  # xywh format
            >>> result = Box2dConverter.convert_format(boxes, Box2DFormat.XYWH, Box2DFormat.XYXY)
            >>> print(result)  # [[10, 20, 40, 60]]
        """

        # Return original boxes if formats are the same
        if input_format == output_format:
            return boxes

        # Ensure boxes are float32 for calculations
        boxes = np.asarray(boxes, dtype=np.float32)
        
        # Get conversion function
        convert_func = Box2dConverter.conversion_map.get((input_format, output_format), None)
        if convert_func is None:
            raise ValueError(f"Unsupported conversion: {input_format} to {output_format}")

        result = convert_func(boxes)
        return result.astype(np.int32) if as_int else result

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
