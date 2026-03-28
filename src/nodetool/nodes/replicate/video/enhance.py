from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class Topaz_Video_Upscale(ReplicateNode):
    """Video Upscaling from Topaz Labs"""

    class Target_resolution(str, Enum):
        _720P = "720p"
        _1080P = "1080p"
        _4K = "4k"

    @classmethod
    def get_basic_fields(cls):
        return ["video", "target_fps", "target_resolution"]

    @classmethod
    def replicate_model_id(cls):
        return "topazlabs/video-upscale:f4dad23bbe2d0bf4736d2ea8c9156f1911d8eeb511c8d0bb390931e25caaef61"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8252807c-a5ac-4008-96a5-814201b4738a/topaz_img.png",
            "created_at": "2025-04-24T21:56:18.449571Z",
            "description": "Video Upscaling from Topaz Labs",
            "github_url": None,
            "license_url": None,
            "name": "video-upscale",
            "owner": "topazlabs",
            "is_official": True,
            "paper_url": None,
            "run_count": 851373,
            "url": "https://replicate.com/topazlabs/video-upscale",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    video: types.VideoRef = Field(
        default=types.VideoRef(), description="Video file to upscale"
    )
    target_fps: int = Field(
        title="Target Fps",
        description="Target FPS (choose from 15fps to 120fps)",
        ge=15.0,
        le=120.0,
        default=60,
    )
    target_resolution: Target_resolution = Field(
        description="Target resolution", default="1080p"
    )
