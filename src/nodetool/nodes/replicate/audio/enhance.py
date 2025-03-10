from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class AudioSuperResolution(ReplicateNode):
    """AudioSR: Versatile Audio Super-resolution at Scale"""

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "ddim_steps", "input_file"]

    @classmethod
    def replicate_model_id(cls):
        return "nateraw/audio-super-resolution:0e453d5e4c2e0ef4f8d38a6167053dda09cf3c8dbca2355cde61dca55a915bc5"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/6bc4e480-6695-451b-940d-48a3b83a1356/replicate-prediction-jvi5xvlbg4v4.png",
            "created_at": "2023-09-20T04:53:52.943393Z",
            "description": "AudioSR: Versatile Audio Super-resolution at Scale",
            "github_url": "https://github.com/haoheliu/versatile_audio_super_resolution",
            "license_url": "https://huggingface.co/haoheliu/wellsolve_audio_super_resolution_48k/blob/main/README.md",
            "name": "audio-super-resolution",
            "owner": "nateraw",
            "paper_url": "https://arxiv.org/abs/2309.07314",
            "run_count": 53659,
            "url": "https://replicate.com/nateraw/audio-super-resolution",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.AudioRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    ddim_steps: int = Field(
        title="Ddim Steps",
        description="Number of inference steps",
        ge=10.0,
        le=500.0,
        default=50,
    )
    input_file: types.AudioRef = Field(
        default=types.AudioRef(), description="Audio to upsample"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier free guidance",
        ge=1.0,
        le=20.0,
        default=3.5,
    )
