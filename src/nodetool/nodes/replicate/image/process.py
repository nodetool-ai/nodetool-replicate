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
            "is_official": False,
            "paper_url": None,
            "run_count": 10409096,
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
            "is_official": False,
            "paper_url": "https://arxiv.org/pdf/2011.11961.pdf",
            "run_count": 1535301,
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
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2212.11613",
            "run_count": 1668853,
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
        description="Choose the model size.", default="large"
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
            "is_official": False,
            "paper_url": None,
            "run_count": 51713,
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
    scheduler: Scheduler = Field(description="scheduler", default="K_EULER")
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
            "is_official": False,
            "paper_url": None,
            "run_count": 17935,
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


class Nano_Banana(ReplicateNode):
    """Google's latest image editing model in Gemini 2.5"""

    class Aspect_ratio(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        _1_1 = "1:1"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _16_9 = "16:9"
        _21_9 = "21:9"

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_input", "aspect_ratio"]

    @classmethod
    def replicate_model_id(cls):
        return "google/nano-banana:d05a591283da31be3eea28d5634ef9e26989b351718b6489bd308426ebd0a3e8"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/ed879e99-71b5-4689-bed3-e7305e35a28a/this.png",
            "created_at": "2025-08-26T21:08:24.983047Z",
            "description": "Google's latest image editing model in Gemini 2.5",
            "github_url": None,
            "license_url": None,
            "name": "nano-banana",
            "owner": "google",
            "is_official": True,
            "paper_url": None,
            "run_count": 73705765,
            "url": "https://replicate.com/google/nano-banana",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    prompt: str | None = Field(
        title="Prompt",
        description="A text description of the image you want to generate",
        default=None,
    )
    image_input: list = Field(
        title="Image Input",
        description="Input images to transform or use as reference (supports multiple images)",
        default=[],
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="match_input_image"
    )
    output_format: Output_format = Field(
        description="Format of the output image", default="jpg"
    )


class Expand_Image(ReplicateNode):
    """Bria Expand expands images beyond their borders in high quality. Resizing the image by generating new pixels to expand to the desired aspect ratio. Trained exclusively on licensed data for safe and risk-free commercial use"""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _16_9 = "16:9"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "sync", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "bria/expand-image:18d2dffd371ca05b45b7a4e9d82bae0f1f356563633f48d48dca4ccf82ec489d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/Kke10tmiC2yaekp8wUGz4BJ4SgDafE2zGUXDXNGanjEhqL6pA/tmpiutbqh22.png",
            "created_at": "2025-07-03T06:08:32.245632Z",
            "description": "Bria Expand expands images beyond their borders in high quality. Resizing the image by generating new pixels to expand to the desired aspect ratio. Trained exclusively on licensed data for safe and risk-free commercial use",
            "github_url": None,
            "license_url": "https://learn.bria.ai/hubfs/Terms%20and%20Conditions/Bria%20AI%20Online%20Terms%20and%20Conditions%20(March%202024)%20v1.1c.pdf?hsLang=en&_gl=1*iwvu7w*_gcl_au*MzQyMzUxMzAxLjE3NDcwNDU4NTg.*_ga*MjAxNDky",
            "name": "expand-image",
            "owner": "bria",
            "is_official": True,
            "paper_url": None,
            "run_count": 333527,
            "url": "https://replicate.com/bria/expand-image",
            "visibility": "public",
            "weights_url": "https://huggingface.co/briaai/BRIA-2.3-ControlNet-Inpainting",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    sync: bool = Field(
        title="Sync", description="Synchronous response mode", default=True
    )
    image: types.ImageRef = Field(default=types.ImageRef(), description="Image file")
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    image_url: str | None = Field(
        title="Image Url", description="Image URL", default=None
    )
    canvas_size: list | None = Field(
        title="Canvas Size",
        description="Desired output canvas dimensions [width, height]. Default [1000, 1000]",
        default=None,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for expansion.", default="1:1"
    )
    preserve_alpha: bool = Field(
        title="Preserve Alpha",
        description="Preserve alpha channel in output",
        default=True,
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Negative prompt for image generation",
        default=None,
    )
    content_moderation: bool = Field(
        title="Content Moderation",
        description="Enable content moderation",
        default=False,
    )
    original_image_size: list | None = Field(
        title="Original Image Size",
        description="Size of original image in canvas [width, height]",
        default=None,
    )
    original_image_location: list | None = Field(
        title="Original Image Location",
        description="Position of original image in canvas [x, y]",
        default=None,
    )
