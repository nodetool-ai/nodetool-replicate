from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class RemoveBackground(ReplicateNode):
    """Remove images background"""

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def replicate_model_id(cls):
        return "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/pbxt/2hczaMwD9xrsIR8h3Cl8iYGbHaCdFhIOMZ0LfoYfHlKuuIBQA/out.png",
            "created_at": "2022-11-18T00:55:22.939155Z",
            "description": "Remove images background",
            "github_url": "https://github.com/chenxwh/rembg/tree/replicate",
            "license_url": "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
            "name": "rembg",
            "owner": "cjwbw",
            "paper_url": None,
            "run_count": 8092828,
            "url": "https://replicate.com/cjwbw/rembg",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    image: types.ImageRef = Field(default=types.ImageRef(), description="Input image")


class ModNet(ReplicateNode):
    """A deep learning approach to remove background & adding new background image"""

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def replicate_model_id(cls):
        return "pollinations/modnet:da7d45f3b836795f945f221fc0b01a6d3ab7f5e163f13208948ad436001e2255"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/bb0ab3e4-5efa-446f-939a-23e78f2b82de/output.png",
            "created_at": "2022-11-19T04:56:59.860128Z",
            "description": "A deep learning approach to remove background & adding new background image",
            "github_url": "https://github.com/pollinations/MODNet-BGRemover",
            "license_url": None,
            "name": "modnet",
            "owner": "pollinations",
            "paper_url": "https://arxiv.org/pdf/2011.11961.pdf",
            "run_count": 667715,
            "url": "https://replicate.com/pollinations/modnet",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    image: types.ImageRef = Field(default=types.ImageRef(), description="input image")


class DD_Color(ReplicateNode):
    """Towards Photo-Realistic Image Colorization via Dual Decoders"""

    class Model_size(str, Enum):
        LARGE = "large"
        TINY = "tiny"

    @classmethod
    def get_basic_fields(cls):
        return ["image", "model_size"]

    @classmethod
    def replicate_model_id(cls):
        return "piddnad/ddcolor:ca494ba129e44e45f661d6ece83c4c98a9a7c774309beca01429b58fce8aa695"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/d8d3648a-044e-4474-8392-87d52c0c2c68/ddcolor.jpg",
            "created_at": "2024-01-12T15:02:06.387410Z",
            "description": "Towards Photo-Realistic Image Colorization via Dual Decoders",
            "github_url": "https://github.com/piddnad/DDColor",
            "license_url": "https://github.com/piddnad/DDColor/blob/master/LICENSE",
            "name": "ddcolor",
            "owner": "piddnad",
            "paper_url": "https://arxiv.org/abs/2212.11613",
            "run_count": 215563,
            "url": "https://replicate.com/piddnad/ddcolor",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    image: str | None = Field(
        title="Image", description="Grayscale input image.", default=None
    )
    model_size: Model_size = Field(
        description="Choose the model size.", default=Model_size("large")
    )


class Magic_Style_Transfer(ReplicateNode):
    """Restyle an image with the style of another one. I strongly suggest to upscale the results with Clarity AI"""

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        HEUNDISCRETE = "HeunDiscrete"
        KARRASDPM = "KarrasDPM"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        K_EULER = "K_EULER"
        PNDM = "PNDM"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "batouresearch/magic-style-transfer:3b5fa5d360c361090f11164292e45cc5d14cea8d089591d47c580cac9ec1c7ca"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/pbxt/CgdTGuA9wdoWGhVUMgpPIv9mh4rpLnYYViUmeLKV8wF2QGRJA/out-0.png",
            "created_at": "2024-03-20T16:20:23.445929Z",
            "description": "Restyle an image with the style of another one. I strongly suggest to upscale the results with Clarity AI",
            "github_url": "https://github.com/BatouResearch/Cog-Face-to-Anything/tree/magic-style-transfer",
            "license_url": None,
            "name": "magic-style-transfer",
            "owner": "fermatresearch",
            "paper_url": None,
            "run_count": 32308,
            "url": "https://replicate.com/fermatresearch/magic-style-transfer",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(default=types.ImageRef(), description="Input image")
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="An astronaut riding a rainbow unicorn",
    )
    ip_image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    ip_scale: float = Field(
        title="Ip Scale",
        description="IP Adapter strength.",
        ge=0.0,
        le=1.0,
        default=0.3,
    )
    strength: float = Field(
        title="Strength",
        description="When img2img is active, the denoising strength. 1 means total destruction of the input image.",
        ge=0.0,
        le=1.0,
        default=0.9,
    )
    scheduler: Scheduler = Field(description="scheduler", default=Scheduler("K_EULER"))
    lora_scale: float = Field(
        title="Lora Scale",
        description="LoRA additive scale. Only applicable on trained models.",
        ge=0.0,
        le=1.0,
        default=0.9,
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output",
        ge=1.0,
        le=4.0,
        default=1,
    )
    lora_weights: str | None = Field(
        title="Lora Weights",
        description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        default=None,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=4,
    )
    resizing_scale: float = Field(
        title="Resizing Scale",
        description="If you want the image to have a solid margin. Scale of the solid margin. 1.0 means no resizing.",
        ge=1.0,
        le=10.0,
        default=1,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    background_color: str = Field(
        title="Background Color",
        description="When passing an image with alpha channel, it will be replaced with this color",
        default="#A2A2A2",
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=30,
    )
    condition_canny_scale: float = Field(
        title="Condition Canny Scale",
        description="The bigger this number is, the more ControlNet interferes",
        ge=0.0,
        le=2.0,
        default=0.15,
    )
    condition_depth_scale: float = Field(
        title="Condition Depth Scale",
        description="The bigger this number is, the more ControlNet interferes",
        ge=0.0,
        le=2.0,
        default=0.35,
    )


class ObjectRemover(ReplicateNode):
    """None"""

    @classmethod
    def get_basic_fields(cls):
        return ["org_image", "mask_image"]

    @classmethod
    def replicate_model_id(cls):
        return "codeplugtech/object_remover:499559d430d997c34aa80142bfede2ad182b78e9dda9e8e03be5689d99969282"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/pbxt/PeUSD8TLKs0lXSTavj96kkOSfpoAKhRIG8LY5U0erX53QgskA/in-painted.png",
            "created_at": "2024-02-13T07:17:58.590961Z",
            "description": None,
            "github_url": None,
            "license_url": None,
            "name": "object_remover",
            "owner": "codeplugtech",
            "paper_url": None,
            "run_count": 6338,
            "url": "https://replicate.com/codeplugtech/object_remover",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    org_image: types.ImageRef = Field(
        default=types.ImageRef(), description="Original input image"
    )
    mask_image: types.ImageRef = Field(
        default=types.ImageRef(), description="Mask image"
    )
