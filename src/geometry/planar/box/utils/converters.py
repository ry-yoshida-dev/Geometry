import numpy as np
from ....array_types import FloatArray


class Box2dConverters:
    """
    Static methods for converting between different bounding box formats.
    
    This class provides utility functions to convert bounding boxes between various
    coordinate representations commonly used in computer vision and object detection.
    
    Supported formats:
    - xywh: (x, y, width, height) - top-left corner with width and height
    - xyxy: (x1, y1, x2, y2) - top-left and bottom-right corners
    - uvwh: (cx, cy, width, height) - center point with width and height
    - uvsr: (cx, cy, area, aspect_ratio) - center point with area and width/height ratio
    - uvah: (cx, cy, aspect_ratio, height) - center point with aspect ratio and height
    """
    
    @staticmethod
    def xywh2xyxy(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x, y, width, height) to (x1, y1, x2, y2) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xywh format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xyxy format with shape (N, 4)
        """
        x, y, w, h = np.split(boxes, 4, axis=-1)
        return np.concatenate([x, y, x + w, y + h], axis=-1)

    @staticmethod
    def xywh2uvwh(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x, y, width, height) to (cx, cy, width, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xywh format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in uvwh format with shape (N, 4)
        """
        x, y, w, h = np.split(boxes, 4, axis=-1)
        return np.concatenate([x + w / 2, y + h / 2, w, h], axis=-1)

    @staticmethod
    def xywh2uvsr(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x, y, width, height) to (cx, cy, area, aspect_ratio) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xywh format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in uvsr format with shape (N, 4)
        """
        x, y, w, h = np.split(boxes, 4, axis=-1)
        return np.concatenate([x + w / 2, y + h / 2, w * h, w / h], axis=-1)

    @staticmethod
    def xywh2uvah(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x, y, width, height) to (cx, cy, aspect_ratio, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xywh format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in uvah format with shape (N, 4)
        """
        x, y, w, h = np.split(boxes, 4, axis=-1)
        return np.concatenate([x + w / 2, y + h / 2, w / h, h], axis=-1)

    @staticmethod
    def xyxy2xywh(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x1, y1, x2, y2) to (x, y, width, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xyxy format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xywh format with shape (N, 4)
        """
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
        return np.concatenate([x1, y1, x2 - x1, y2 - y1], axis=-1)

    @staticmethod
    def xyxy2uvsr(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x1, y1, x2, y2) to (cx, cy, area, aspect_ratio) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xyxy format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in uvsr format with shape (N, 4)
        """
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / np.maximum(h, np.finfo(float).eps)  # Avoid division by zero
        return np.concatenate([cx, cy, s, r], axis=-1)

    @staticmethod
    def xyxy2uvah(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (x1, y1, x2, y2) to (cx, cy, aspect_ratio, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in xyxy format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in uvah format with shape (N, 4)
        """
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
        w, h = x2 - x1, y2 - y1
        return np.concatenate([(x1 + x2) / 2, (y1 + y2) / 2, w / h, h], axis=-1)

    @staticmethod
    def uvah2xywh(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (cx, cy, aspect_ratio, height) to (x, y, width, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in uvah format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xywh format with shape (N, 4)
        """
        cx, cy, r, h = np.split(boxes, 4, axis=-1)
        w = r * h  # r = w/h, so w = r * h
        return np.concatenate([cx - w / 2, cy - h / 2, w, h], axis=-1)

    @staticmethod
    def uvah2xyxy(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (cx, cy, aspect_ratio, height) to (x1, y1, x2, y2) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in uvah format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xyxy format with shape (N, 4)
        """
        cx, cy, r, h = np.split(boxes, 4, axis=-1)  # r = w/h
        w = r * h
        return np.concatenate([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    @staticmethod
    def uvwh2xywh(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (cx, cy, width, height) to (x, y, width, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in uvwh format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xywh format with shape (N, 4)
        """
        cx, cy, w, h = np.split(boxes, 4, axis=-1)
        return np.concatenate([cx - w / 2, cy - h / 2, w, h], axis=-1)

    @staticmethod
    def uvsr2xywh(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (cx, cy, area, aspect_ratio) to (x, y, width, height) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in uvsr format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xywh format with shape (N, 4)
        """
        cx, cy, s, r = np.split(boxes, 4, axis=-1)  # s = w*h, r = w/h
        w = np.sqrt(s * r)  # s * r = wh * w/h = w^2
        h = np.sqrt(s / r)  # s / r = wh * h/w = h^2
        return np.concatenate([cx - w / 2, cy - h / 2, w, h], axis=-1)

    @staticmethod
    def uvsr2xyxy(boxes: FloatArray) -> FloatArray:
        """
        Convert boxes from (cx, cy, area, aspect_ratio) to (x1, y1, x2, y2) format.
        
        Parameters:
        ----------
        boxes: FloatArray
            Input boxes in uvsr format with shape (N, 4)
            
        Returns:
        ---------
        FloatArray: Converted boxes in xyxy format with shape (N, 4)
        """
        cx, cy, s, r = np.split(boxes, 4, axis=-1)  # s = w*h, r = w/h
        
        w = np.sqrt(s * r)  # s * r = wh * w/h = w^2
        h = np.sqrt(s / r)  # s / r = wh * h/w = h^2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.concatenate([x1, y1, x2, y2], axis=-1)