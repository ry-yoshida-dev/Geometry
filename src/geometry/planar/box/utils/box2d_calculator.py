import numpy as np

from ..format import Box2DFormat

class BboxCalculator:
    @staticmethod
    def measure_aspect_ratios(bboxes: np.ndarray) -> np.ndarray:
        """
        Calculates the aspect ratio (height / width) for each bounding box.

        Parameters:
        ----------
        bboxes : np.ndarray 
            Array of shape (N, 4) containing bounding box coordinates in the format [x1, y1, x2, y2].

        Returns:
        ----------
        np.ndarray 
            Array of aspect ratios for each bounding box.
        """
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] 
        return (y2-y1)/(x2-x1)

    @staticmethod
    def _compute_intersection_area(
        boxes1: np.ndarray, 
        boxes2: np.ndarray, 
        is_all_combinations: bool = True
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the intersection area between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        is_all_combinations: bool
            If True, computes pairwise combinations of bounding boxes from boxes1 and boxes2.
            Otherwise, computes intersection for each bounding box in boxes1 with the corresponding one in boxes2.

        Returns:
        ---------
        tuple[np.ndarray, np.ndarray, np.ndarray]: Intersection area, area of boxes1, and area of boxes2 of shape (N, M).
        """
        # Extract coordinates of the bounding boxes
        x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

        # Compute the intersection coordinates
        inter_x1 = np.maximum(x1_1[:, None] if is_all_combinations else x1_1, x1_2)
        inter_y1 = np.maximum(y1_1[:, None] if is_all_combinations else y1_1, y1_2)
        inter_x2 = np.minimum(x2_1[:, None] if is_all_combinations else x2_1, x2_2)
        inter_y2 = np.minimum(y2_1[:, None] if is_all_combinations else y2_1, y2_2)

        # Calculate intersection dimensions and area
        inter_width = np.maximum(0, inter_x2 - inter_x1)
        inter_height = np.maximum(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        # Calculate the areas of the individual bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        return intersection, area1, area2

    @staticmethod
    def compute_min_intersection_ratio(
        boxes1: np.ndarray, 
        boxes2: np.ndarray
        ) -> np.ndarray:
        """
        Compute the minimum intersection ratio between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns:
        ---------
        np.ndarray: Array of minimum intersection ratios of shape (N, M).
        """
        intersection, area1, area2 = BboxCalculator._compute_intersection_area(boxes1, boxes2)

        area1 = area1[:, None]
        area2 = area2[None, :]

        ratio1 = intersection / (area1 + np.finfo(float).eps)
        ratio2 = intersection / (area2 + np.finfo(float).eps)
        return np.minimum(ratio1, ratio2)

    @staticmethod
    def compute_iou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray, 
        is_all_combinations: bool = True
        ) -> np.ndarray:
        """
        Compute the Intersection over Union (IoU) between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        is_all_combinations: bool
            If True, computes pairwise IoU calculation, otherwise computes IoU for each bounding box in boxes1 with the corresponding one in boxes2.

        Returns:
        ---------
        np.ndarray: Array of IoU values of shape (N, M).
        """
        intersection, area1, area2 = BboxCalculator._compute_intersection_area(
            boxes1=boxes1, 
            boxes2=boxes2, 
            is_all_combinations=is_all_combinations
            )
        denominator = (area1[:, None] if is_all_combinations else area1) + area2 - intersection
        return intersection / np.maximum(denominator, np.finfo(float).eps)

    @staticmethod
    def compute_saiou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray
        ) -> np.ndarray:
        """
        Compute the soft alignment IoU between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns:
        ---------
        np.ndarray: Array of soft IoU values of shape (N, M).
        """
        iou_matrix = BboxCalculator.compute_iou(boxes1, boxes2)
        return BboxCalculator.compute_saiou_from_iou(iou_matrix)

    @staticmethod
    def compute_saiou_from_iou(
        iou_matrix: np.ndarray
        ) -> np.ndarray:
        """
        Compute the soft IoU from the IoU matrix.

        Parameters:
        ----------
        iou_matrix: np.ndarray
            Array of IoU values with shape (N, M).

        Returns:
        ---------
        np.ndarray: Array of soft IoU values of shape (N, M).
        """
        sum_pred = np.sum(iou_matrix, axis=0, keepdims=True) # (1, N)
        sum_gt = np.sum(iou_matrix, axis=1, keepdims=True)   # (M, 1)
        union = sum_pred + sum_gt - iou_matrix
        eps = 1e-9
        return iou_matrix / (union + eps)

    @staticmethod
    def compute_diou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray
        ) -> np.ndarray:
        """
        Compute the Distance Intersection over Union (DIoU) between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns:
        ---------
        np.ndarray: Array of DIoU values of shape (N, M).
        """
        iou = BboxCalculator.compute_iou(
            boxes1=boxes1, 
            boxes2=boxes2,
            is_all_combinations=True
            )

        centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2  # (N, 2)
        centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # (M, 2)

        delta = centers1[:, None, :] - centers2[None, :, :]  # (N, M, 2)
        center_distances = np.sum(delta ** 2, axis=-1)  # (N, M)

        enclosing_tl = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
        enclosing_br = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
        enclosing_wh = enclosing_br - enclosing_tl
        enclosing_diagonal = np.sum(enclosing_wh ** 2, axis=-1)  # (N, M)

        diou = iou - (center_distances / np.clip(enclosing_diagonal, a_min=np.finfo(float).eps, a_max=None))
        return diou
    
    @staticmethod
    def compute_ciou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray
        ) -> np.ndarray:
        """
        Compute the CIoU between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns:
        ---------
        np.ndarray: Array of CIoU values of shape (N, M).
        """
        diou = BboxCalculator.compute_diou(
            boxes1=boxes1, 
            boxes2=boxes2
            )

        w1 = boxes1[:, 2] - boxes1[:, 0]  # (N,)
        h1 = boxes1[:, 3] - boxes1[:, 1]  # (N,)
        w2 = boxes2[:, 2] - boxes2[:, 0]  # (M,)
        h2 = boxes2[:, 3] - boxes2[:, 1]  # (M,)


        v = (4 / (np.pi ** 2)) * (
            np.arctan(w2[None, :] / h2[None, :]) - np.arctan(w1[:, None] / h1[:, None])
        ) ** 2  # (N, M)


        with np.errstate(divide='ignore', invalid='ignore'):
            iou = BboxCalculator.compute_iou(boxes1, boxes2)
            S = 1 - iou
            alpha = v / (S + v + np.finfo(float).eps)  # avoid division by zero

        ciou = diou - alpha * v 
        return ciou

    @staticmethod
    def compute_giou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray
        ) -> np.ndarray:
        """
        Compute the Generalized Intersection over Union (GIoU) between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns:
        ---------
        np.ndarray: Array of GIoU values of shape (N, M).
        """
        intersection, area1, area2 = BboxCalculator._compute_intersection_area(boxes1, boxes2)
        union_area = area1[:, None] + area2 - intersection
        iou = intersection / np.maximum(union_area, 1e-7)

        # Compute the convex bounding boxes that cover both sets of boxes
        convex_x1 = np.minimum(boxes1[:, None, 0], boxes2[:, 0])
        convex_y1 = np.minimum(boxes1[:, None, 1], boxes2[:, 1])
        convex_x2 = np.maximum(boxes1[:, None, 2], boxes2[:, 2])
        convex_y2 = np.maximum(boxes1[:, None, 3], boxes2[:, 3])

        # Calculate the area of the convex bounding boxes
        convex_area = (convex_x2 - convex_x1) * (convex_y2 - convex_y1)
        
        # Calculate GIoU
        giou = iou - (convex_area - union_area) / np.maximum(convex_area, 1e-7)
        return giou

    @staticmethod
    def _apply_buffer(
        boxes: np.ndarray, 
        buffer: float
        ) -> np.ndarray:
        """
        Apply a buffer around the bounding boxes.

        Parameters:
        ----------
        boxes: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        buffer: float
            Buffer factor to expand the bounding boxes.

        Returns:
        ---------
        np.ndarray: Array of buffered bounding boxes of shape (N, 4).
        """
        boxes = boxes.astype(np.float64)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        buffered_boxes = boxes.copy()
        
        # Apply buffer to all bounding box edges
        buffered_boxes[:, 0] -= buffer * w
        buffered_boxes[:, 1] -= buffer * h
        buffered_boxes[:, 2] += buffer * w
        buffered_boxes[:, 3] += buffer * h
        
        return buffered_boxes

    @staticmethod
    def compute_biou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray, 
        buffer: float = 0.1, 
        is_BGIoU_enabled: bool = False
        ) -> np.ndarray:
        """
        Compute the buffered Intersection over Union (BIoU) between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        buffer: float
        is_BGIoU_enabled: bool
            Whether to compute Buffered GIoU (BGIoU). Defaults to False.

        Returns:
        ---------
        np.ndarray: Array of buffered IoU values of shape (N, M).
        """
        buffered_boxes1 = BboxCalculator._apply_buffer(boxes1, buffer)
        buffered_boxes2 = BboxCalculator._apply_buffer(boxes2, buffer)

        iou_func = BboxCalculator.compute_giou if is_BGIoU_enabled else BboxCalculator.compute_iou
        return iou_func(buffered_boxes1, buffered_boxes2)

    @staticmethod
    def compute_soft_biou(
        boxes1: np.ndarray, 
        boxes2: np.ndarray, 
        confidences1: np.ndarray, 
        k1: float = 0.25, 
        k2: float = 0.5
        ) -> np.ndarray:
        """
        Compute Soft BIoU between two sets of bounding boxes.

        Parameters:
        ----------
        boxes1: np.ndarray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2: np.ndarray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        confidences1: np.ndarray
            Array of confidence values with shape (N,).
        k1: float
            Expansion scale for bboxes1.
        k2: float
            Expansion scale for bboxes2.

        Returns:
        ---------
        np.ndarray: Array of soft BIoU values of shape (N, M).
        """

        def expand_bboxes(bboxes: np.ndarray, confidences: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            w = bboxes[..., 2] - bboxes[..., 0]
            h = bboxes[..., 3] - bboxes[..., 1]
            x1 = bboxes[..., 0] - w * (1 - confidences) * k
            y1 = bboxes[..., 1] - h * (1 - confidences) * k
            x2 = bboxes[..., 2] + w * (1 - confidences) * k
            y2 = bboxes[..., 3] + h * (1 - confidences) * k
            return x1, y1, x2, y2

        def compute_area(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
            return np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

        boxes1 = np.expand_dims(boxes1, 1)  # (N, 1, 4)
        boxes2 = np.expand_dims(boxes2, 0)  # (1, M, 4)
        confidences1 = np.expand_dims(confidences1, 1)  # (N, 1)

        b1_x1, b1_y1, b1_x2, b1_y2 = expand_bboxes(boxes1, confidences1, k1)
        b2_x1, b2_y1, b2_x2, b2_y2 = expand_bboxes(boxes2, confidences1, k2)
        
        # Intersection
        xx1 = np.maximum(b1_x1, b2_x1)
        yy1 = np.maximum(b1_y1, b2_y1)
        xx2 = np.minimum(b1_x2, b2_x2)
        yy2 = np.minimum(b1_y2, b2_y2)
        inter_area = compute_area(xx1, yy1, xx2, yy2)

        # Union
        area1 = compute_area(b1_x1, b1_y1, b1_x2, b1_y2)
        area2 = compute_area(b2_x1, b2_y1, b2_x2, b2_y2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area

    @staticmethod
    def get_box_centers(
        boxes: np.ndarray, 
        box_format: Box2DFormat=Box2DFormat.XYXY
        ):
        """
        Compute the center coordinates of bounding boxes based on the given format.

        Parameters:
        ----------
        boxes: np.ndarray
            Array of shape (N, 4) containing bounding boxes.
        box_format: Box2DFormat
            Format of the bounding boxes.

        Returns:
        ---------
        np.ndarray: Array of shape (N, 2) containing the center coordinates.

        Raises:
            ValueError: If an invalid box format is provided.
        """
        match box_format:
            case Box2DFormat.XYXY | Box2DFormat.TLBR:
                return (boxes[:, :2] + boxes[:, 2:]) / 2
            case Box2DFormat.XYWH | Box2DFormat.TLWH:
                return boxes[:, :2] + boxes[:, 2:] / 2
            case Box2DFormat.UVAH | Box2DFormat.UVSR | Box2DFormat.CXCYAH | Box2DFormat.CXCYSR:
                return boxes[:, :2]
            case _:
                raise ValueError(f"Invalid box format: {box_format}")

    @staticmethod
    def compute_cdf_from_matrix(
        matrix: np.ndarray, 
        ignore_diagonal: bool = True
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulative relative frequency distribution (CDF) from a 2D matrix.

        Parameters:
        ----------
        matrix: np.ndarray
            Array of shape (N, M) containing the matrix.
        ignore_diagonal: bool
            If True, diagonal elements are excluded (useful for distance matrices).

        Returns:
        ---------
        tuple[np.ndarray, np.ndarray]: tuple containing the sorted values and the cumulative relative frequencies.

        - np.ndarray: Array of shape (N, 2) containing the sorted values.
        - np.ndarray: Array of shape (N, 2) containing the cumulative relative frequencies.
        """
        values = matrix.flatten()

        if ignore_diagonal and matrix.shape[0] == matrix.shape[1]:
            values = matrix[~np.eye(matrix.shape[0], dtype=bool)]

        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

        return sorted_values, cumulative


########################################################

if __name__ == "__main__":
    boxes1 = np.array([
        [100, 100, 200, 200],  # Box A
        [10, 10, 50, 50],      # Box B
        [300, 300, 500, 350]   # Box C
        ], dtype=np.float64)
    boxes2 = np.array([
        [190, 190, 290, 290],  # A' (IoU low)
        [60, 60, 100, 100],    # B' (IoU 0, BIoU > 0)
        [300, 300, 500, 350]   # C' (IoU 1.0)
        ], dtype=np.float64)
    iou = BboxCalculator.compute_iou(boxes1, boxes2)
    print(f"IoU: \n {iou}")
    biou = BboxCalculator.compute_biou(boxes1, boxes2, buffer=0.2, is_BGIoU_enabled=False)
    print(f"BIoU: \n {biou}")
