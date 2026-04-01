from __future__ import annotations
from enum import Enum

class Box2DFormat(Enum):
    """
    Enum class for the format of the bounding box.

    Attributes:
    ----------
    XYXY: top-left and bottom-right corners
    TLBR: alias of XYXY
    XYWH: top-left and width/height
    TLWH: alias of XYWH
    UVWH: center (u,v) and width/height
    CXCYWH: alias of UVWH
    UVAH: center (u,v), aspect ratio, height
    CXCYAH: alias of UVAH
    UVSR: center (u,v), scale, rotation
    CXCYSR: alias of UVSR
    """    
    XYXY = "xyxy"      # top-left and bottom-right corners
    TLBR = "tlbr"      # alias of xyxy
    XYWH = "xywh"      # top-left and width/height
    TLWH = "tlwh"      # alias of xywh
    UVWH = "uvwh"      # center (u,v) and width/height
    CXCYWH = "cxcywh"  # alias of uvwh
    UVAH = "uvah"      # center (u,v), aspect ratio, height
    CXCYAH = "cxcyah"  # alias of uvah
    UVSR = "uvsr"      # center (u,v), scale, rotation
    CXCYSR = "cxcysr"  # alias of uvsr

    @property
    def fields(self) -> tuple[str, str, str, str]:
        """
        Get the fields of the box.

        Returns
        -------
        tuple[str, str, str, str]: The tuple of the fields of the box.
        """
        match self:
            case Box2DFormat.XYXY | Box2DFormat.TLBR:
                return ("x1", "y1", "x2", "y2")
            case Box2DFormat.XYWH | Box2DFormat.TLWH:
                return ("x", "y", "w", "h")
            case Box2DFormat.UVWH:
                return ("u", "v", "w", "h")
            case Box2DFormat.CXCYWH:
                return ("cx", "cy", "w", "h")
            case Box2DFormat.UVAH:
                return ("u", "v", "a", "h")
            case Box2DFormat.CXCYAH:
                return ("cx", "cy", "a", "h")
            case Box2DFormat.UVSR:
                return ("u", "v", "s", "r")
            case Box2DFormat.CXCYSR:
                return ("cx", "cy", "s", "r")

    def standardize_format(self) -> Box2DFormat:
        """
        Standardize box format to canonical Box2DFormat enum.

        Returns
        -------
        Box2DFormat: Canonical Box2DFormat enum.
        """
        match self:
            case Box2DFormat.XYWH | Box2DFormat.TLWH:
                return Box2DFormat.XYWH
            case Box2DFormat.XYXY | Box2DFormat.TLBR:
                return Box2DFormat.XYXY
            case Box2DFormat.UVWH | Box2DFormat.CXCYWH:
                return Box2DFormat.UVWH
            case Box2DFormat.UVAH | Box2DFormat.CXCYAH:
                return Box2DFormat.UVAH
            case Box2DFormat.UVSR | Box2DFormat.CXCYSR:
                return Box2DFormat.UVSR

