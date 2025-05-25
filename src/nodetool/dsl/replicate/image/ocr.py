from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class LatexOCR(GraphNode):
    """Optical character recognition to turn images of latex equations into latex format."""

    image_path: str | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Input image"
    )

    @classmethod
    def get_node_type(cls):
        return "replicate.image.ocr.LatexOCR"


class TextExtractOCR(GraphNode):
    """A simple OCR Model that can easily extract text from an image."""

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="Image to process",
    )

    @classmethod
    def get_node_type(cls):
        return "replicate.image.ocr.TextExtractOCR"
