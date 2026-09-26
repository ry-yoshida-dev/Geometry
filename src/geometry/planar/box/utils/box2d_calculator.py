import numpy as np
from numpy.typing import NDArray

from ....array_types import FloatArray, NumericArray
from ..format import Box2DFormat


class BboxCalculator:
    """
    Vectorized metrics for batches of 2D bounding boxes.

    Box inputs may hold integer or floating dtypes; every metric is computed
    in float64 so integer inputs neither overflow nor truncate.
    """

    @staticmethod
    def _as_float64(values: NumericArray) -> NDArray[np.float64]:
        """
        Convert an integer or floating array to float64.

        Parameters
        ----------
        values : NumericArray
            Input array of any numeric dtype.

        Returns
        -------
        NDArray[np.float64]
            The same values as float64, without copying when already float64.
        """
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def measure_aspect_ratios(bboxes: NumericArray) -> FloatArray:
        """
        Calculate the aspect ratio (width / height) for each bounding box.

        Parameters
        ----------
        bboxes : NumericArray
            Array of shape (N, 4) in xyxy format.

        Returns
        -------
        FloatArray
            Array of shape (N,) with the aspect ratio of each box.
        """
        boxes = BboxCalculator._as_float64(bboxes)
        return (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])

    @staticmethod
    def compute_intersection_area(
        boxes1: NumericArray,
        boxes2: NumericArray,
        is_all_combinations: bool = True,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Compute the intersection area between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        is_all_combinations : bool
            If True, computes every pair between boxes1 and boxes2. Otherwise,
            pairs each row of boxes1 with the same row of boxes2.

        Returns
        -------
        tuple[FloatArray, FloatArray, FloatArray]
            Intersection areas of shape (N, M) (or (N,) when not all
            combinations), areas of boxes1 of shape (N,), and areas of boxes2
            of shape (M,).
        """
        first = BboxCalculator._as_float64(boxes1)
        second = BboxCalculator._as_float64(boxes2)
        first_x1, first_y1, first_x2, first_y2 = first[:, 0], first[:, 1], first[:, 2], first[:, 3]
        second_x1, second_y1, second_x2, second_y2 = second[:, 0], second[:, 1], second[:, 2], second[:, 3]

        if is_all_combinations:
            first_x1 = first_x1[:, None]
            first_y1 = first_y1[:, None]
            first_x2 = first_x2[:, None]
            first_y2 = first_y2[:, None]

        intersection_width = np.clip(np.minimum(first_x2, second_x2) - np.maximum(first_x1, second_x1), 0.0, None)
        intersection_height = np.clip(np.minimum(first_y2, second_y2) - np.maximum(first_y1, second_y1), 0.0, None)
        intersection = intersection_width * intersection_height

        first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
        second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])

        return intersection, first_area, second_area

    @staticmethod
    def compute_min_intersection_ratio(
        boxes1: NumericArray,
        boxes2: NumericArray,
    ) -> FloatArray:
        """
        Compute the intersection divided by the larger of each pair's areas.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns
        -------
        FloatArray
            Array of minimum intersection ratios of shape (N, M).
        """
        intersection, first_area, second_area = BboxCalculator.compute_intersection_area(boxes1, boxes2)
        epsilon = np.finfo(np.float64).eps

        first_ratio = intersection / (first_area[:, None] + epsilon)
        second_ratio = intersection / (second_area[None, :] + epsilon)
        return np.minimum(first_ratio, second_ratio)

    @staticmethod
    def compute_iou(
        boxes1: NumericArray,
        boxes2: NumericArray,
        is_all_combinations: bool = True,
    ) -> FloatArray:
        """
        Compute the Intersection over Union (IoU) between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        is_all_combinations : bool
            If True, computes pairwise IoU. Otherwise, computes IoU for each
            row of boxes1 with the same row of boxes2.

        Returns
        -------
        FloatArray
            IoU values of shape (N, M), or (N,) when not all combinations.
        """
        intersection, first_area, second_area = BboxCalculator.compute_intersection_area(
            boxes1=boxes1,
            boxes2=boxes2,
            is_all_combinations=is_all_combinations,
        )
        broadcast_first_area = first_area[:, None] if is_all_combinations else first_area
        union = broadcast_first_area + second_area - intersection
        return intersection / np.maximum(union, np.finfo(np.float64).eps)

    @staticmethod
    def compute_saiou(
        boxes1: NumericArray,
        boxes2: NumericArray,
    ) -> FloatArray:
        """
        Compute the soft alignment IoU between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns
        -------
        FloatArray
            Soft IoU values of shape (N, M).
        """
        iou_matrix = BboxCalculator.compute_iou(boxes1, boxes2)
        return BboxCalculator.compute_saiou_from_iou(iou_matrix)

    @staticmethod
    def compute_saiou_from_iou(iou_matrix: NumericArray) -> FloatArray:
        """
        Compute the soft alignment IoU from an IoU matrix.

        Parameters
        ----------
        iou_matrix : NumericArray
            IoU values of shape (N, M).

        Returns
        -------
        FloatArray
            Soft IoU values of shape (N, M).
        """
        iou = BboxCalculator._as_float64(iou_matrix)
        column_sum = np.sum(iou, axis=0, keepdims=True)
        row_sum = np.sum(iou, axis=1, keepdims=True)
        union = column_sum + row_sum - iou
        return iou / (union + 1e-9)

    @staticmethod
    def compute_diou(
        boxes1: NumericArray,
        boxes2: NumericArray,
    ) -> FloatArray:
        """
        Compute the Distance IoU (DIoU) between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns
        -------
        FloatArray
            DIoU values of shape (N, M).
        """
        iou = BboxCalculator.compute_iou(boxes1=boxes1, boxes2=boxes2, is_all_combinations=True)
        first = BboxCalculator._as_float64(boxes1)
        second = BboxCalculator._as_float64(boxes2)

        first_centers = (first[:, :2] + first[:, 2:]) / 2
        second_centers = (second[:, :2] + second[:, 2:]) / 2
        center_offsets = first_centers[:, None, :] - second_centers[None, :, :]
        squared_center_distances = np.sum(center_offsets**2, axis=-1)

        enclosing_top_left = np.minimum(first[:, None, :2], second[None, :, :2])
        enclosing_bottom_right = np.maximum(first[:, None, 2:], second[None, :, 2:])
        enclosing_size = enclosing_bottom_right - enclosing_top_left
        squared_enclosing_diagonal = np.sum(enclosing_size**2, axis=-1)

        return iou - squared_center_distances / np.maximum(squared_enclosing_diagonal, np.finfo(np.float64).eps)

    @staticmethod
    def compute_ciou(
        boxes1: NumericArray,
        boxes2: NumericArray,
    ) -> FloatArray:
        """
        Compute the Complete IoU (CIoU) between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns
        -------
        FloatArray
            CIoU values of shape (N, M).
        """
        diou = BboxCalculator.compute_diou(boxes1=boxes1, boxes2=boxes2)
        first = BboxCalculator._as_float64(boxes1)
        second = BboxCalculator._as_float64(boxes2)

        first_width = first[:, 2] - first[:, 0]
        first_height = first[:, 3] - first[:, 1]
        second_width = second[:, 2] - second[:, 0]
        second_height = second[:, 3] - second[:, 1]

        aspect_angle_difference: NDArray[np.float64] = np.arctan(second_width[None, :] / second_height[None, :]) - np.arctan(
            first_width[:, None] / first_height[:, None]
        )
        aspect_consistency = (4 / np.pi**2) * aspect_angle_difference**2

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = BboxCalculator.compute_iou(boxes1, boxes2)
            trade_off = aspect_consistency / (1 - iou + aspect_consistency + np.finfo(np.float64).eps)

        return diou - trade_off * aspect_consistency

    @staticmethod
    def compute_giou(
        boxes1: NumericArray,
        boxes2: NumericArray,
    ) -> FloatArray:
        """
        Compute the Generalized IoU (GIoU) between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.

        Returns
        -------
        FloatArray
            GIoU values of shape (N, M).
        """
        intersection, first_area, second_area = BboxCalculator.compute_intersection_area(boxes1, boxes2)
        union = first_area[:, None] + second_area - intersection
        iou: NDArray[np.float64] = intersection / np.maximum(union, 1e-7)

        first = BboxCalculator._as_float64(boxes1)
        second = BboxCalculator._as_float64(boxes2)
        enclosing_x1: NDArray[np.float64] = np.minimum(first[:, None, 0], second[:, 0])
        enclosing_y1: NDArray[np.float64] = np.minimum(first[:, None, 1], second[:, 1])
        enclosing_x2: NDArray[np.float64] = np.maximum(first[:, None, 2], second[:, 2])
        enclosing_y2: NDArray[np.float64] = np.maximum(first[:, None, 3], second[:, 3])
        enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)

        return iou - (enclosing_area - union) / np.maximum(enclosing_area, 1e-7)

    @staticmethod
    def _apply_buffer(
        boxes: NumericArray,
        buffer: float,
    ) -> NDArray[np.float64]:
        """
        Expand every box edge outward by a fraction of its width or height.

        Parameters
        ----------
        boxes : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        buffer : float
            Fraction of the width (height) added to each horizontal (vertical) edge.

        Returns
        -------
        NDArray[np.float64]
            Buffered bounding boxes of shape (N, 4).
        """
        float_boxes = BboxCalculator._as_float64(boxes)
        width = float_boxes[:, 2] - float_boxes[:, 0]
        height = float_boxes[:, 3] - float_boxes[:, 1]
        horizontal_margin = buffer * width
        vertical_margin = buffer * height
        return np.stack(
            [
                float_boxes[:, 0] - horizontal_margin,
                float_boxes[:, 1] - vertical_margin,
                float_boxes[:, 2] + horizontal_margin,
                float_boxes[:, 3] + vertical_margin,
            ],
            axis=-1,
        )

    @staticmethod
    def compute_biou(
        boxes1: NumericArray,
        boxes2: NumericArray,
        buffer: float = 0.1,
        is_BGIoU_enabled: bool = False,
    ) -> FloatArray:
        """
        Compute the buffered IoU (BIoU) between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        buffer : float
            Fraction of each box's width and height added to its edges.
        is_BGIoU_enabled : bool
            Whether to compute Buffered GIoU (BGIoU). Defaults to False.

        Returns
        -------
        FloatArray
            Buffered IoU values of shape (N, M).
        """
        buffered_boxes1 = BboxCalculator._apply_buffer(boxes1, buffer)
        buffered_boxes2 = BboxCalculator._apply_buffer(boxes2, buffer)

        if is_BGIoU_enabled:
            return BboxCalculator.compute_giou(buffered_boxes1, buffered_boxes2)
        return BboxCalculator.compute_iou(buffered_boxes1, buffered_boxes2)

    @staticmethod
    def compute_soft_biou(
        boxes1: NumericArray,
        boxes2: NumericArray,
        confidences1: NumericArray,
        k1: float = 0.25,
        k2: float = 0.5,
    ) -> FloatArray:
        """
        Compute Soft BIoU between two sets of bounding boxes.

        Parameters
        ----------
        boxes1 : NumericArray
            Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes2 : NumericArray
            Array of bounding boxes with shape (M, 4) in xyxy format.
        confidences1 : NumericArray
            Array of confidence values with shape (N,). Following the BoostTrack
            definition, the confidence of each boxes1 row drives the expansion of
            both boxes in every (i, j) pair.
        k1 : float
            Expansion scale for boxes1.
        k2 : float
            Expansion scale for boxes2.

        Returns
        -------
        FloatArray
            Soft BIoU values of shape (N, M).
        """
        first = BboxCalculator._as_float64(boxes1)[:, None, :]
        second = BboxCalculator._as_float64(boxes2)[None, :, :]
        uncertainty = 1 - BboxCalculator._as_float64(confidences1)[:, None]

        first_expanded = BboxCalculator._expand_by_uncertainty(first, uncertainty * k1)
        second_expanded = BboxCalculator._expand_by_uncertainty(second, uncertainty * k2)

        intersection_width: NDArray[np.float64] = np.clip(
            np.minimum(first_expanded[..., 2], second_expanded[..., 2])
            - np.maximum(first_expanded[..., 0], second_expanded[..., 0]),
            0.0,
            None,
        )
        intersection_height: NDArray[np.float64] = np.clip(
            np.minimum(first_expanded[..., 3], second_expanded[..., 3])
            - np.maximum(first_expanded[..., 1], second_expanded[..., 1]),
            0.0,
            None,
        )
        intersection_area = intersection_width * intersection_height

        first_area = BboxCalculator._clipped_area(first_expanded)
        second_area = BboxCalculator._clipped_area(second_expanded)
        union_area = first_area + second_area - intersection_area

        return intersection_area / union_area

    @staticmethod
    def _expand_by_uncertainty(
        boxes: NDArray[np.float64],
        scale: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Expand xyxy boxes by scale times their width and height on every side.

        Parameters
        ----------
        boxes : NDArray[np.float64]
            Boxes of shape (..., 4) in xyxy format.
        scale : NDArray[np.float64]
            Expansion factor broadcastable to the leading shape of boxes.

        Returns
        -------
        NDArray[np.float64]
            Expanded boxes with the broadcast leading shape and 4 columns.
        """
        horizontal_margin = (boxes[..., 2] - boxes[..., 0]) * scale
        vertical_margin = (boxes[..., 3] - boxes[..., 1]) * scale
        return np.stack(
            [
                boxes[..., 0] - horizontal_margin,
                boxes[..., 1] - vertical_margin,
                boxes[..., 2] + horizontal_margin,
                boxes[..., 3] + vertical_margin,
            ],
            axis=-1,
        )

    @staticmethod
    def _clipped_area(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Area of xyxy boxes, treating negative widths or heights as zero.

        Parameters
        ----------
        boxes : NDArray[np.float64]
            Boxes of shape (..., 4) in xyxy format.

        Returns
        -------
        NDArray[np.float64]
            Areas with the leading shape of boxes.
        """
        width = np.clip(boxes[..., 2] - boxes[..., 0], 0.0, None)
        height = np.clip(boxes[..., 3] - boxes[..., 1], 0.0, None)
        return width * height

    @staticmethod
    def get_box_centers(
        boxes: NumericArray,
        box_format: Box2DFormat = Box2DFormat.XYXY,
    ) -> NumericArray:
        """
        Compute the center coordinates of bounding boxes in the given format.

        Parameters
        ----------
        boxes : NumericArray
            Array of shape (N, 4) containing bounding boxes.
        box_format : Box2DFormat
            Format of the bounding boxes.

        Returns
        -------
        NumericArray
            Array of shape (N, 2) containing the center coordinates.

        Raises
        ------
        ValueError
            If an invalid box format is provided.
        """
        match box_format:
            case Box2DFormat.XYXY | Box2DFormat.TLBR:
                return (boxes[:, :2] + boxes[:, 2:]) / 2
            case Box2DFormat.XYWH | Box2DFormat.TLWH:
                return boxes[:, :2] + boxes[:, 2:] / 2
            case (
                Box2DFormat.UVWH
                | Box2DFormat.CXCYWH
                | Box2DFormat.UVAH
                | Box2DFormat.CXCYAH
                | Box2DFormat.UVSR
                | Box2DFormat.CXCYSR
            ):
                return boxes[:, :2]
            case _:
                raise ValueError(f"Invalid box format: {box_format}")

    @staticmethod
    def compute_cdf_from_matrix(
        matrix: NumericArray,
        ignore_diagonal: bool = True,
    ) -> tuple[NumericArray, FloatArray]:
        """
        Compute the cumulative relative frequency distribution (CDF) of a 2D matrix.

        Parameters
        ----------
        matrix : NumericArray
            Array of shape (N, M).
        ignore_diagonal : bool
            If True and the matrix is square, diagonal elements are excluded
            (useful for distance matrices).

        Returns
        -------
        tuple[NumericArray, FloatArray]
            Sorted values and their cumulative relative frequencies, both 1-D
            with the same length.
        """
        values = matrix.flatten()

        if ignore_diagonal and matrix.shape[0] == matrix.shape[1]:
            values = matrix[~np.eye(matrix.shape[0], dtype=bool)]

        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

        return sorted_values, cumulative
