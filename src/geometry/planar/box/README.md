# box

Axis-aligned 2D bounding boxes in image pixel coordinates (x right, y down). Single-box and batched APIs, XYXY/XYWH implementations under `boxes_/`, and shared helpers in `utils/`.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Abstract `Box2D` over `float` or `np.ndarray`. |
| [box.py](./box.py) | Single box — `value` shape `(4,)`. |
| [boxes.py](./boxes.py) | Batch — `value` shape `(N, 4)`. |
| [format.py](./format.py) | `Box2DFormat` enum. |
| [boxes_/](./boxes_/README.md) | Concrete XYXY / XYWH classes. |
| [utils/](./utils/README.md) | `BboxCalculator`, `Box2dConverter`, low-level converters. |
