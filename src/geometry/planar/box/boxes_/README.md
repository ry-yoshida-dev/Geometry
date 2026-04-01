# boxes_

Concrete XYXY and XYWH implementations for one box (`(4,)`) or a batch (`(N, 4)`).

## Components

| Component | Description |
|-----------|-------------|
| [box_xyxy.py](./box_xyxy.py) | `Box2D_XYXY` — corners `x1, y1, x2, y2`. |
| [box_xywh.py](./box_xywh.py) | `Box2D_XYWH` — top-left plus `width`, `height`. |
| [boxes_xyxy.py](./boxes_xyxy.py) | `Boxes2D_XYXY`. |
| [boxes_xywh.py](./boxes_xywh.py) | `Boxes2D_XYWH`. |

Use `Box2D.register` / `Boxes2D.register` with `Box2DFormat` (XYXY/TLBR or XYWH/TLWH), or construct these classes directly.
