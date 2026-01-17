from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class AdInpaint(ReplicateNode):
    """Product advertising image generator"""

    class Pixel(str, Enum):
        _512___512 = "512 * 512"
        _768___768 = "768 * 768"
        _1024___1024 = "1024 * 1024"

    class Product_size(str, Enum):
        ORIGINAL = "Original"
        _0_6___WIDTH = "0.6 * width"
        _0_5___WIDTH = "0.5 * width"
        _0_4___WIDTH = "0.4 * width"
        _0_3___WIDTH = "0.3 * width"
        _0_2___WIDTH = "0.2 * width"

    @classmethod
    def get_basic_fields(cls):
        return ["pixel", "scale", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "logerzhu/ad-inpaint:b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/e21a425b-4cdb-445b-8b2f-5be1b26fb78d/ad_inpaint_2.jpg",
            "created_at": "2023-04-03T11:25:28.290524Z",
            "description": "Product advertising image generator",
            "github_url": None,
            "license_url": None,
            "name": "ad-inpaint",
            "owner": "logerzhu",
            "is_official": False,
            "paper_url": None,
            "run_count": 626132,
            "url": "https://replicate.com/logerzhu/ad-inpaint",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    pixel: Pixel = Field(description="image total pixel", default="512 * 512")
    scale: int = Field(
        title="Scale",
        description="Factor to scale image by (maximum: 4)",
        ge=0.0,
        le=4.0,
        default=3,
    )
    prompt: str | None = Field(
        title="Prompt", description="Product name or prompt", default=None
    )
    image_num: int = Field(
        title="Image Num",
        description="Number of image to generate",
        ge=0.0,
        le=4.0,
        default=1,
    )
    image_path: types.ImageRef = Field(
        default=types.ImageRef(), description="input image"
    )
    manual_seed: int = Field(title="Manual Seed", description="Manual Seed", default=-1)
    product_size: Product_size = Field(
        description="Max product size", default="Original"
    )
    guidance_scale: float = Field(
        title="Guidance Scale", description="Guidance Scale", default=7.5
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Anything you don't want in the photo",
        default="low quality, out of frame, illustration, 3d, sepia, painting, cartoons, sketch, watermark, text, Logo, advertisement",
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps", description="Inference Steps", default=20
    )


class ConsistentCharacter(ReplicateNode):
    """Create images of a given character in different poses"""

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "subject"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/consistent-character:9c77a3c2f884193fcee4d89645f02a0b9def9434f9e03cb98460456b831c8772"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/TOVBIdLIP6rPLdn4HUtnZHyvHwBAaXzCNUWenllOPD8hBw6JA/ComfyUI_00005_.webp",
            "created_at": "2024-05-30T16:48:52.345721Z",
            "description": "Create images of a given character in different poses",
            "github_url": "https://github.com/fofr/cog-consistent-character",
            "license_url": "https://github.com/fofr/cog-consistent-character/blob/main/LICENSE",
            "name": "consistent-character",
            "owner": "sdxl-based",
            "is_official": False,
            "paper_url": None,
            "run_count": 1409183,
            "url": "https://replicate.com/sdxl-based/consistent-character",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    prompt: str = Field(
        title="Prompt",
        description="Describe the subject. Include clothes and hairstyle for more consistency.",
        default="A headshot photo",
    )
    subject: types.ImageRef = Field(
        default=types.ImageRef(),
        description="An image of a person. Best images are square close ups of a face, but they do not have to be.",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
        ge=0.0,
        le=100.0,
        default=80,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want to see in your image",
        default="",
    )
    randomise_poses: bool = Field(
        title="Randomise Poses", description="Randomise the poses used.", default=True
    )
    number_of_outputs: int = Field(
        title="Number Of Outputs",
        description="The number of images to generate.",
        ge=1.0,
        le=20.0,
        default=3,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )
    number_of_images_per_pose: int = Field(
        title="Number Of Images Per Pose",
        description="The number of images to generate for each pose.",
        ge=1.0,
        le=4.0,
        default=1,
    )


class PulidBase(ReplicateNode):
    """Use a face to make images. Uses SDXL fine-tuned checkpoints."""

    class Face_style(str, Enum):
        HIGH_FIDELITY = "high-fidelity"
        STYLIZED = "stylized"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    class Checkpoint_model(str, Enum):
        GENERAL___ALBEDOBASEXL_V21 = "general - albedobaseXL_v21"
        GENERAL___DREAMSHAPERXL_ALPHA2XL10 = "general - dreamshaperXL_alpha2Xl10"
        ANIMATED___STARLIGHTXLANIMATED_V3 = "animated - starlightXLAnimated_v3"
        ANIMATED___PIXLANIMECARTOONCOMIC_V10 = "animated - pixlAnimeCartoonComic_v10"
        REALISTIC___RUNDIFFUSIONXL_BETA = "realistic - rundiffusionXL_beta"
        REALISTIC___REALVISXL_V4_0 = "realistic - RealVisXL_V4.0"
        REALISTIC___SDXLUNSTABLEDIFFUSERS_NIHILMANIA = (
            "realistic - sdxlUnstableDiffusers_nihilmania"
        )
        CINEMATIC___CINEMATICREDMOND = "cinematic - CinematicRedmond"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/pulid-base:65ea75658bf120abbbdacab07e89e78a74a6a1b1f504349f4c4e3b01a655ee7a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/fb4110a1-ae64-491f-8a3e-0fdd4ef991bb/pulid-base-cover.webp",
            "created_at": "2024-05-09T13:48:08.359715Z",
            "description": "Use a face to make images. Uses SDXL fine-tuned checkpoints.",
            "github_url": "https://github.com/fofr/cog-comfyui-pulid/tree/pulid-base",
            "license_url": None,
            "name": "pulid-base",
            "owner": "fofr",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2404.16022",
            "run_count": 323672,
            "url": "https://replicate.com/fofr/pulid-base",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    width: int = Field(
        title="Width",
        description="Width of the output image (ignored if structure image given)",
        default=1024,
    )
    height: int = Field(
        title="Height",
        description="Height of the output image (ignored if structure image given)",
        default=1024,
    )
    prompt: str = Field(
        title="Prompt",
        description="You might need to include a gender in the prompt to get the desired result",
        default="A photo of a person",
    )
    face_image: types.ImageRef = Field(
        default=types.ImageRef(), description="The face image to use for the generation"
    )
    face_style: Face_style = Field(
        description="Style of the face", default="high-fidelity"
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
        ge=0.0,
        le=100.0,
        default=80,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want to see in your image",
        default="",
    )
    checkpoint_model: Checkpoint_model = Field(
        description="Model to use for the generation",
        default="general - dreamshaperXL_alpha2Xl10",
    )
    number_of_images: int = Field(
        title="Number Of Images",
        description="Number of images to generate",
        ge=1.0,
        le=10.0,
        default=1,
    )


class StableDiffusion(ReplicateNode):
    """A latent text-to-image diffusion model capable of generating photo-realistic images given any text input"""

    class Width(int, Enum):
        _64 = 64
        _128 = 128
        _192 = 192
        _256 = 256
        _320 = 320
        _384 = 384
        _448 = 448
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Height(int, Enum):
        _64 = 64
        _128 = 128
        _192 = 192
        _256 = 256
        _320 = 320
        _384 = 384
        _448 = 448
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        K_EULER = "K_EULER"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        PNDM = "PNDM"
        KLMS = "KLMS"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/710f5e9f-9561-4e4f-9d1e-614205f62597/stable-diffusion.webp",
            "created_at": "2022-08-22T21:37:08.396208Z",
            "description": "A latent text-to-image diffusion model capable of generating photo-realistic images given any text input",
            "github_url": "https://github.com/replicate/cog-stable-diffusion",
            "license_url": "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
            "name": "stable-diffusion",
            "owner": "stability-ai",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2112.10752",
            "run_count": 110886731,
            "url": "https://replicate.com/stability-ai/stable-diffusion",
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
    width: Width = Field(
        description="Width of generated image in pixels. Needs to be a multiple of 64",
        default=768,
    )
    height: Height = Field(
        description="Height of generated image in pixels. Needs to be a multiple of 64",
        default=768,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="a vision of paradise. unreal engine",
    )
    scheduler: Scheduler = Field(
        description="Choose a scheduler.", default="DPMSolverMultistep"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to generate.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=20.0,
        default=7.5,
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Specify things to not see in the output",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )


class StableDiffusion3_5_Medium(ReplicateNode):
    """2.5 billion parameter image model with improved MMDiT-X architecture"""

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _1_1 = "1:1"
        _21_9 = "21:9"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["cfg", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "stability-ai/stable-diffusion-3.5-medium:1323a3a68cbf2b58c708f38fba9557e39d68a77cb287a8d7372ba0443f6f0767"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/d65fc397-135b-4976-a84d-12980ab2c0bc/replicate-prediction-_4kWPYZu.webp",
            "created_at": "2024-10-29T12:55:45.899504Z",
            "description": "2.5 billion parameter image model with improved MMDiT-X architecture",
            "github_url": None,
            "license_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/LICENSE.md",
            "name": "stable-diffusion-3.5-medium",
            "owner": "stability-ai",
            "is_official": True,
            "paper_url": "https://arxiv.org/abs/2403.03206",
            "run_count": 98649,
            "url": "https://replicate.com/stability-ai/stable-diffusion-3.5-medium",
            "visibility": "public",
            "weights_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    cfg: float = Field(
        title="Cfg",
        description="The guidance scale tells the model how similar the output should be to the prompt.",
        ge=1.0,
        le=10.0,
        default=5,
    )
    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image mode. The aspect ratio of your output will match this image.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="The aspect ratio of your output image. This value is ignored if you are using an input image.",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="What you do not want to see in the image",
        default=None,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength (or denoising strength) when using image to image. 1.0 corresponds to full destruction of information in image.",
        ge=0.0,
        le=1.0,
        default=0.85,
    )


class StableDiffusion3_5_Large(ReplicateNode):
    """A text-to-image model that generates high-resolution images with fine details. It supports various artistic styles and produces diverse outputs from the same prompt, thanks to Query-Key Normalization."""

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _1_1 = "1:1"
        _21_9 = "21:9"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["cfg", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "stability-ai/stable-diffusion-3.5-large:2fdf9488b53c1e0fd3aef7b477def1c00d1856a38466733711f9c769942598f5"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/4b03d178-eaf3-4458-a752-dbc76098396b/replicate-prediction-_ycGb1jN.webp",
            "created_at": "2024-10-21T20:53:39.435334Z",
            "description": "A text-to-image model that generates high-resolution images with fine details. It supports various artistic styles and produces diverse outputs from the same prompt, thanks to Query-Key Normalization.",
            "github_url": None,
            "license_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md",
            "name": "stable-diffusion-3.5-large",
            "owner": "stability-ai",
            "is_official": True,
            "paper_url": "https://arxiv.org/abs/2403.03206",
            "run_count": 1854285,
            "url": "https://replicate.com/stability-ai/stable-diffusion-3.5-large",
            "visibility": "public",
            "weights_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    cfg: float = Field(
        title="Cfg",
        description="The guidance scale tells the model how similar the output should be to the prompt.",
        ge=1.0,
        le=10.0,
        default=5,
    )
    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image mode. The aspect ratio of your output will match this image.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="The aspect ratio of your output image. This value is ignored if you are using an input image.",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="What you do not want to see in the image",
        default=None,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength (or denoising strength) when using image to image. 1.0 corresponds to full destruction of information in image.",
        ge=0.0,
        le=1.0,
        default=0.85,
    )


class StableDiffusion3_5_Large_Turbo(ReplicateNode):
    """A text-to-image model that generates high-resolution images with fine details. It supports various artistic styles and produces diverse outputs from the same prompt, with a focus on fewer inference steps"""

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _1_1 = "1:1"
        _21_9 = "21:9"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["cfg", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "stability-ai/stable-diffusion-3.5-large-turbo:6ce89263555dde3393564e799f1310ee247c5339c3c665250b5dd5d26b7bcc3d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/9e1b4258-22bd-4a59-ba4a-ecac220a8a9b/replicate-prediction-_WU4XtaV.webp",
            "created_at": "2024-10-22T12:09:38.705615Z",
            "description": "A text-to-image model that generates high-resolution images with fine details. It supports various artistic styles and produces diverse outputs from the same prompt, with a focus on fewer inference steps",
            "github_url": None,
            "license_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/main/LICENSE.md",
            "name": "stable-diffusion-3.5-large-turbo",
            "owner": "stability-ai",
            "is_official": True,
            "paper_url": "https://arxiv.org/abs/2403.03206",
            "run_count": 881288,
            "url": "https://replicate.com/stability-ai/stable-diffusion-3.5-large-turbo",
            "visibility": "public",
            "weights_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    cfg: float = Field(
        title="Cfg",
        description="The guidance scale tells the model how similar the output should be to the prompt.",
        ge=1.0,
        le=10.0,
        default=1,
    )
    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image mode. The aspect ratio of your output will match this image.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="The aspect ratio of your output image. This value is ignored if you are using an input image.",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="What you do not want to see in the image",
        default=None,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength (or denoising strength) when using image to image. 1.0 corresponds to full destruction of information in image.",
        ge=0.0,
        le=1.0,
        default=0.85,
    )


class Photon_Flash(ReplicateNode):
    """Accelerated variant of Photon prioritizing speed while maintaining quality"""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _16_9 = "16:9"
        _9_21 = "9:21"
        _21_9 = "21:9"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "aspect_ratio"]

    @classmethod
    def replicate_model_id(cls):
        return "luma/photon-flash:8cee7d47f81d8f4f77c1aec44ffb3d1ce09d36388db637ceaa8a6cbcf30b63e1"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8459f7e9-7445-4046-82aa-917a0f561b80/tmpyf9dx02r.webp",
            "created_at": "2024-12-05T15:18:04.364421Z",
            "description": "Accelerated variant of Photon prioritizing speed while maintaining quality",
            "github_url": None,
            "license_url": "https://lumalabs.ai/dream-machine/api/terms",
            "name": "photon-flash",
            "owner": "luma",
            "is_official": True,
            "paper_url": None,
            "run_count": 241085,
            "url": "https://replicate.com/luma/photon-flash",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="16:9"
    )
    image_reference: str | None = Field(
        title="Image Reference",
        description="Reference image to guide generation",
        default=None,
    )
    style_reference: str | None = Field(
        title="Style Reference",
        description="Style reference image to guide generation",
        default=None,
    )
    character_reference: str | None = Field(
        title="Character Reference",
        description="Character reference image to guide generation",
        default=None,
    )
    image_reference_url: types.ImageRef = Field(
        default=types.ImageRef(), description="Deprecated: Use image_reference instead"
    )
    style_reference_url: types.ImageRef = Field(
        default=types.ImageRef(), description="Deprecated: Use style_reference instead"
    )
    image_reference_weight: float = Field(
        title="Image Reference Weight",
        description="Weight of the reference image. Larger values will make the reference image have a stronger influence on the generated image.",
        ge=0.0,
        le=1.0,
        default=0.85,
    )
    style_reference_weight: float = Field(
        title="Style Reference Weight",
        description="Weight of the style reference image",
        ge=0.0,
        le=1.0,
        default=0.85,
    )
    character_reference_url: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Deprecated: Use character_reference instead",
    )


class StableDiffusionXL(ReplicateNode):
    """A text-to-image generative AI model that creates beautiful images"""

    class Refine(str, Enum):
        NO_REFINER = "no_refiner"
        EXPERT_ENSEMBLE_REFINER = "expert_ensemble_refiner"
        BASE_IMAGE_REFINER = "base_image_refiner"

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        HEUNDISCRETE = "HeunDiscrete"
        KARRASDPM = "KarrasDPM"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        K_EULER = "K_EULER"
        PNDM = "PNDM"

    class Lr_scheduler(str, Enum):
        CONSTANT = "constant"
        LINEAR = "linear"

    class Input_images_filetype(str, Enum):
        ZIP = "zip"
        TAR = "tar"
        INFER = "infer"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/9065f9e3-40da-4742-8cb8-adfa8e794c0d/sdxl_cover.jpg",
            "created_at": "2023-07-26T17:53:09.882651Z",
            "description": "A text-to-image generative AI model that creates beautiful images",
            "github_url": "https://github.com/replicate/cog-sdxl",
            "license_url": "https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SDXL1.0",
            "name": "sdxl",
            "owner": "stability-ai",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2307.01952",
            "run_count": 83539235,
            "url": "https://replicate.com/stability-ai/sdxl",
            "visibility": "public",
            "weights_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    width: int = Field(title="Width", description="Width of output image", default=1024)
    height: int = Field(
        title="Height", description="Height of output image", default=1024
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="An astronaut riding a rainbow unicorn",
    )
    refine: Refine = Field(
        description="Which refine style to use", default="no_refiner"
    )
    scheduler: Scheduler = Field(description="scheduler", default="K_EULER")
    lora_scale: float = Field(
        title="Lora Scale",
        description="LoRA additive scale. Only applicable on trained models.",
        ge=0.0,
        le=1.0,
        default=0.6,
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    refine_steps: int | None = Field(
        title="Refine Steps",
        description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
        default=None,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=7.5,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    high_noise_frac: float = Field(
        title="High Noise Frac",
        description="For expert_ensemble_refiner, the fraction of noise to use",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
        default=False,
    )


class SDXL_Pixar(ReplicateNode):
    """Create Pixar poster easily with SDXL Pixar."""

    class Refine(str, Enum):
        NO_REFINER = "no_refiner"
        EXPERT_ENSEMBLE_REFINER = "expert_ensemble_refiner"
        BASE_IMAGE_REFINER = "base_image_refiner"

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
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "swartype/sdxl-pixar:81f8bbd3463056c8521eb528feb10509cc1385e2fabef590747f159848589048"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/68125b17-60d7-4949-8984-0d50d736a623/out-0_5.png",
            "created_at": "2023-10-21T10:32:49.911227Z",
            "description": "Create Pixar poster easily with SDXL Pixar.",
            "github_url": None,
            "license_url": None,
            "name": "sdxl-pixar",
            "owner": "swartype",
            "is_official": False,
            "paper_url": None,
            "run_count": 657572,
            "url": "https://replicate.com/swartype/sdxl-pixar",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    width: int = Field(title="Width", description="Width of output image", default=1024)
    height: int = Field(
        title="Height", description="Height of output image", default=1024
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="An astronaut riding a rainbow unicorn",
    )
    refine: Refine = Field(
        description="Which refine style to use", default="no_refiner"
    )
    scheduler: Scheduler = Field(description="scheduler", default="K_EULER")
    lora_scale: float = Field(
        title="Lora Scale",
        description="LoRA additive scale. Only applicable on trained models.",
        ge=0.0,
        le=1.0,
        default=0.6,
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    refine_steps: int | None = Field(
        title="Refine Steps",
        description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
        default=None,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=7.5,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    high_noise_frac: float = Field(
        title="High Noise Frac",
        description="For expert_ensemble_refiner, the fraction of noise to use",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See https://replicate.com/docs/how-does-replicate-work#safety",
        default=False,
    )


class SDXL_Emoji(ReplicateNode):
    """An SDXL fine-tune based on Apple Emojis"""

    class Refine(str, Enum):
        NO_REFINER = "no_refiner"
        EXPERT_ENSEMBLE_REFINER = "expert_ensemble_refiner"
        BASE_IMAGE_REFINER = "base_image_refiner"

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
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8629c6ba-b94c-4cbd-93aa-bda2b8ebecd9/F5Mg2KeXgAAkfre.jpg",
            "created_at": "2023-09-04T09:18:11.028708Z",
            "description": "An SDXL fine-tune based on Apple Emojis",
            "github_url": None,
            "license_url": None,
            "name": "sdxl-emoji",
            "owner": "fofr",
            "is_official": False,
            "paper_url": None,
            "run_count": 11598128,
            "url": "https://replicate.com/fofr/sdxl-emoji",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    width: int = Field(title="Width", description="Width of output image", default=1024)
    height: int = Field(
        title="Height", description="Height of output image", default=1024
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="An astronaut riding a rainbow unicorn",
    )
    refine: Refine = Field(
        description="Which refine style to use", default="no_refiner"
    )
    scheduler: Scheduler = Field(description="scheduler", default="K_EULER")
    lora_scale: float = Field(
        title="Lora Scale",
        description="LoRA additive scale. Only applicable on trained models.",
        ge=0.0,
        le=1.0,
        default=0.6,
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    refine_steps: int | None = Field(
        title="Refine Steps",
        description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
        default=None,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=7.5,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    high_noise_frac: float = Field(
        title="High Noise Frac",
        description="For expert_ensemble_refiner, the fraction of noise to use",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See https://replicate.com/docs/how-does-replicate-work#safety",
        default=False,
    )


class StableDiffusionInpainting(ReplicateNode):
    """Fill in masked parts of images with Stable Diffusion"""

    class Width(int, Enum):
        _64 = 64
        _128 = 128
        _192 = 192
        _256 = 256
        _320 = 320
        _384 = 384
        _448 = 448
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Height(int, Enum):
        _64 = 64
        _128 = 128
        _192 = 192
        _256 = 256
        _320 = 320
        _384 = 384
        _448 = 448
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        K_EULER = "K_EULER"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        PNDM = "PNDM"
        KLMS = "KLMS"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/pbxt/xs0pPOUM6HKmPlJJBXqKfE1YsiMzgNsCuGedlX0VqvPYifLgA/out-0.png",
            "created_at": "2022-12-02T17:40:01.152489Z",
            "description": "Fill in masked parts of images with Stable Diffusion",
            "github_url": "https://github.com/replicate/cog-stable-diffusion-inpainting",
            "license_url": "https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL",
            "name": "stable-diffusion-inpainting",
            "owner": "stability-ai",
            "is_official": False,
            "paper_url": None,
            "run_count": 20917509,
            "url": "https://replicate.com/stability-ai/stable-diffusion-inpainting",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Initial image to generate variations of. Will be resized to height x width",
    )
    width: Width = Field(
        description="Width of generated image in pixels. Needs to be a multiple of 64",
        default=512,
    )
    height: Height = Field(
        description="Height of generated image in pixels. Needs to be a multiple of 64",
        default=512,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="a vision of paradise. unreal engine",
    )
    scheduler: Scheduler = Field(
        description="Choose a scheduler.", default="DPMSolverMultistep"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to generate.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=20.0,
        default=7.5,
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Specify things to not see in the output",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
        default=False,
    )


class Kandinsky_2_2(ReplicateNode):
    """multilingual text2image latent diffusion model"""

    class Width(int, Enum):
        _384 = 384
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _960 = 960
        _1024 = 1024
        _1152 = 1152
        _1280 = 1280
        _1536 = 1536
        _1792 = 1792
        _2048 = 2048

    class Height(int, Enum):
        _384 = 384
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _960 = 960
        _1024 = 1024
        _1152 = 1152
        _1280 = 1280
        _1536 = 1536
        _1792 = 1792
        _2048 = 2048

    class Output_format(str, Enum):
        WEBP = "webp"
        JPEG = "jpeg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "ai-forever/kandinsky-2.2:ad9d7879fbffa2874e1d909d1d37d9bc682889cc65b31f7bb00d2362619f194a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/618e68d3-fba3-4fd0-a060-cdd46b2ab7cf/out-0_2.jpg",
            "created_at": "2023-07-12T21:53:29.439515Z",
            "description": "multilingual text2image latent diffusion model",
            "github_url": "https://github.com/chenxwh/Kandinsky-2/tree/v2.2",
            "license_url": "https://github.com/ai-forever/Kandinsky-2/blob/main/license",
            "name": "kandinsky-2.2",
            "owner": "ai-forever",
            "is_official": False,
            "paper_url": None,
            "run_count": 10028882,
            "url": "https://replicate.com/ai-forever/kandinsky-2.2",
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
    width: Width = Field(
        description="Width of output image. Lower the setting if hits memory limits.",
        default=512,
    )
    height: Height = Field(
        description="Height of output image. Lower the setting if hits memory limits.",
        default=512,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="A moss covered astronaut with a black background",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    output_format: Output_format = Field(
        description="Output image format", default="webp"
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Specify things to not see in the output",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=75,
    )
    num_inference_steps_prior: int = Field(
        title="Num Inference Steps Prior",
        description="Number of denoising steps for priors",
        ge=1.0,
        le=500.0,
        default=25,
    )


class Flux_Schnell(ReplicateNode):
    """The fastest image generation model tailored for local development and personal use"""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "go_fast"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-schnell:c846a69991daf4c0e5d016514849d14ee5b2e6846ce6b9d6f21369e564cfe51e"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/67c990ba-bb67-4355-822f-2bd8c42b2f0d/flux-schnell.webp",
            "created_at": "2024-07-30T00:32:11.473557Z",
            "description": "The fastest image generation model tailored for local development and personal use",
            "github_url": "https://github.com/replicate/cog-flux",
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-schnell",
            "name": "flux-schnell",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 594913202,
            "url": "https://replicate.com/black-forest-labs/flux-schnell",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16. Note that outputs will not be deterministic when this is enabled, even if you set a seed.",
        default=True,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. 4 is recommended, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=4.0,
        default=4,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Dev(ReplicateNode):
    """A 12 billion parameter rectified flow transformer capable of generating images from text descriptions"""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-dev:6e4a938f85952bdabcc15aa329178c4d681c52bf25a0342403287dc26944661d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/cb4203e5-9ece-42e7-b326-98ff3fa35c3a/Replicate_Prediction_15.webp",
            "created_at": "2024-07-29T23:25:06.100855Z",
            "description": "A 12 billion parameter rectified flow transformer capable of generating images from text descriptions",
            "github_url": "https://github.com/replicate/cog-flux",
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev",
            "name": "flux-dev",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 37659127,
            "url": "https://replicate.com/black-forest-labs/flux-dev",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: str | None = Field(
        title="Image",
        description="Input image for image to image mode. The aspect ratio of your output will match this image",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16. Note that outputs will not be deterministic when this is enabled, even if you set a seed.",
        default=True,
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image",
        ge=0.0,
        le=10.0,
        default=3,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Pro(ReplicateNode):
    """State-of-the-art image generation with top of the line prompt following, visual quality, image detail and output diversity."""

    class Aspect_ratio(str, Enum):
        CUSTOM = "custom"
        _1_1 = "1:1"
        _16_9 = "16:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _3_4 = "3:4"
        _4_3 = "4:3"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "steps", "width"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-pro:ce4035b99fc7bac18bc2f0384632858f126f6b4d96c88603a898a76b8e0c4ac2"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/a36275e2-34d4-4b3d-83cd-f9aaf73c9386/https___replicate.delive_o40qpZl.webp",
            "created_at": "2024-08-01T09:32:10.863297Z",
            "description": "State-of-the-art image generation with top of the line prompt following, visual quality, image detail and output diversity.",
            "github_url": None,
            "license_url": "https://replicate.com/black-forest-labs/flux-pro#license",
            "name": "flux-pro",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 13822157,
            "url": "https://replicate.com/black-forest-labs/flux-pro",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    steps: int = Field(
        title="Steps", description="Deprecated", ge=1.0, le=50.0, default=25
    )
    width: int | None = Field(
        title="Width",
        description="Width of the generated image in text-to-image mode. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32). Note: Ignored in img2img and inpainting modes.",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of the generated image in text-to-image mode. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32). Note: Ignored in img2img and inpainting modes.",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Controls the balance between adherence to the text prompt and image quality/diversity. Higher values make the output more closely match the prompt but may reduce overall image quality. Lower values allow for more creative freedom but might produce results less relevant to the prompt.",
        ge=2.0,
        le=5.0,
        default=3,
    )
    interval: float = Field(
        title="Interval", description="Deprecated", ge=1.0, le=4.0, default=2
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    image_prompt: str | None = Field(
        title="Image Prompt",
        description="Image to use with Flux Redux. This is used together with the text prompt to guide the generation towards the composition of the image_prompt. Must be jpeg, png, gif, or webp.",
        default=None,
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 6 is most permissive",
        ge=1.0,
        le=6.0,
        default=2,
    )
    prompt_upsampling: bool = Field(
        title="Prompt Upsampling",
        description="Automatically modify the prompt for more creative generation",
        default=False,
    )


class Flux_1_1_Pro_Ultra(ReplicateNode):
    """FLUX1.1 [pro] in ultra and raw modes. Images are up to 4 megapixels. Use raw mode for realism."""

    class Aspect_ratio(str, Enum):
        _21_9 = "21:9"
        _16_9 = "16:9"
        _3_2 = "3:2"
        _4_3 = "4:3"
        _5_4 = "5:4"
        _1_1 = "1:1"
        _4_5 = "4:5"
        _3_4 = "3:4"
        _2_3 = "2:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["raw", "seed", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-1.1-pro-ultra:5ea10f739af9f6d4002fae9aee4c15be14c3c8d7f8b309e634bf68df09159863"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8121c76b-fbff-41d9-834d-c70dea9d2191/flux-ultra-cover.jpg",
            "created_at": "2024-11-06T19:13:05.091037Z",
            "description": "FLUX1.1 [pro] in ultra and raw modes. Images are up to 4 megapixels. Use raw mode for realism.",
            "github_url": None,
            "license_url": "https://replicate.com/black-forest-labs/flux-pro#license",
            "name": "flux-1.1-pro-ultra",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": "https://blackforestlabs.ai/flux-1-1-ultra/",
            "run_count": 19826685,
            "url": "https://replicate.com/black-forest-labs/flux-1.1-pro-ultra",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    raw: bool = Field(
        title="Raw",
        description="Generate less processed, more natural-looking images",
        default=False,
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    image_prompt: str | None = Field(
        title="Image Prompt",
        description="Image to use with Flux Redux. This is used together with the text prompt to guide the generation towards the composition of the image_prompt. Must be jpeg, png, gif, or webp.",
        default=None,
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="jpg"
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 6 is most permissive",
        ge=1.0,
        le=6.0,
        default=2,
    )
    image_prompt_strength: float = Field(
        title="Image Prompt Strength",
        description="Blend between the prompt and the image prompt.",
        ge=0.0,
        le=1.0,
        default=0.1,
    )


class Flux_Dev_Lora(ReplicateNode):
    """A version of flux-dev, a text to image model, that supports fast fine-tuned lora inference"""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-dev-lora:ae0d7d645446924cf1871e3ca8796e8318f72465d2b5af9323a835df93bf0917"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/a79cc4a8-318c-4316-a800-097ef0bdce7a/https___replicate.del_25H5GQ7.webp",
            "created_at": "2024-11-11T23:03:07.000926Z",
            "description": "A version of flux-dev, a text to image model, that supports fast fine-tuned lora inference",
            "github_url": "https://github.com/replicate/cog-flux",
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev",
            "name": "flux-dev-lora",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 5512728,
            "url": "https://replicate.com/black-forest-labs/flux-dev-lora",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image mode. The aspect ratio of your output will match this image",
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16. Note that outputs will not be deterministic when this is enabled, even if you set a seed.",
        default=True,
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image",
        ge=0.0,
        le=10.0,
        default=3,
    )
    extra_lora: str | None = Field(
        title="Extra Lora",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    hf_api_token: str | None = Field(
        title="Hf Api Token",
        description="HuggingFace API token. If you're using a hf lora that needs authentication, you'll need to provide an API token.",
        default=None,
    )
    lora_weights: str | None = Field(
        title="Lora Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>[/<lora-weights-file.safetensors>], CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet, including signed URLs. For example, 'fofr/flux-pixar-cars'. Civit AI and HuggingFace LoRAs may require an API token to access, which you can provide in the `civitai_api_token` and `hf_api_token` inputs respectively.",
        default=None,
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    extra_lora_scale: float = Field(
        title="Extra Lora Scale",
        description="Determines how strongly the extra LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    civitai_api_token: str | None = Field(
        title="Civitai Api Token",
        description="Civitai API token. If you're using a civitai lora that needs authentication, you'll need to provide an API token.",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Schnell_Lora(ReplicateNode):
    """The fastest image generation model tailored for fine-tuned use"""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "go_fast"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-schnell-lora:83180e3ae073b7f87cd85b8bb649337412fd006d10db49e04ea5e821e87fbeb3"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/98c9bf91-5bc0-4a2d-960f-8c3fcd69f1f3/https___replicate.deliver_a20JvIo.png",
            "created_at": "2024-11-11T23:07:50.986160Z",
            "description": "The fastest image generation model tailored for fine-tuned use",
            "github_url": "https://github.com/replicate/cog-flux",
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-schnell",
            "name": "flux-schnell-lora",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 3634602,
            "url": "https://replicate.com/black-forest-labs/flux-schnell-lora",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16. Note that outputs will not be deterministic when this is enabled, even if you set a seed.",
        default=True,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    lora_weights: str | None = Field(
        title="Lora Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. 4 is recommended, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=4.0,
        default=4,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Depth_Pro(ReplicateNode):
    """Professional depth-aware image generation. Edit images while preserving spatial relationships."""

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "steps", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-depth-pro:0e370dce5fdf15aa8b5fe2491474be45628756e8fba97574bfb3bcab46d09fff"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/e365ecff-4023-49f8-96ba-abd710c4bdd9/https___replicate.deliver_xWYu8lC.jpg",
            "created_at": "2024-11-21T09:53:00.631446Z",
            "description": "Professional depth-aware image generation. Edit images while preserving spatial relationships.",
            "github_url": None,
            "license_url": "https://replicate.com/black-forest-labs/flux-depth-pro#license",
            "name": "flux-depth-pro",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 286140,
            "url": "https://replicate.com/black-forest-labs/flux-depth-pro",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    steps: int = Field(
        title="Steps",
        description="Number of diffusion steps. Higher values yield finer details but increase processing time.",
        ge=15.0,
        le=50.0,
        default=50,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Controls the balance between adherence to the text as well as image prompt and image quality/diversity. Higher values make the output more closely match the prompt but may reduce overall image quality. Lower values allow for more creative freedom but might produce results less relevant to the prompt.",
        ge=1.0,
        le=100.0,
        default=30,
    )
    control_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image to use as control input. Must be jpeg, png, gif, or webp.",
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="jpg"
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 6 is most permissive",
        ge=1.0,
        le=6.0,
        default=2,
    )
    prompt_upsampling: bool = Field(
        title="Prompt Upsampling",
        description="Automatically modify the prompt for more creative generation",
        default=False,
    )


class Flux_Canny_Pro(ReplicateNode):
    """Professional edge-guided image generation. Control structure and composition using Canny edge detection"""

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "steps", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-canny-pro:835f0372c2cf4b2e494c2b8626288212ea5c2694ccc2e29f00dfb8cbf2a5e0ce"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/4c07cacc-d206-4587-9357-8e4e81cd761a/https___replicate.deli_lsMxQWe.jpg",
            "created_at": "2024-11-21T09:53:08.913764Z",
            "description": "Professional edge-guided image generation. Control structure and composition using Canny edge detection",
            "github_url": None,
            "license_url": "https://replicate.com/black-forest-labs/flux-canny-pro#license",
            "name": "flux-canny-pro",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 406531,
            "url": "https://replicate.com/black-forest-labs/flux-canny-pro",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    steps: int = Field(
        title="Steps",
        description="Number of diffusion steps. Higher values yield finer details but increase processing time.",
        ge=15.0,
        le=50.0,
        default=50,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Controls the balance between adherence to the text as well as image prompt and image quality/diversity. Higher values make the output more closely match the prompt but may reduce overall image quality. Lower values allow for more creative freedom but might produce results less relevant to the prompt.",
        ge=1.0,
        le=100.0,
        default=30,
    )
    control_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image to use as control input. Must be jpeg, png, gif, or webp.",
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="jpg"
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 6 is most permissive",
        ge=1.0,
        le=6.0,
        default=2,
    )
    prompt_upsampling: bool = Field(
        title="Prompt Upsampling",
        description="Automatically modify the prompt for more creative generation",
        default=False,
    )


class Flux_Fill_Pro(ReplicateNode):
    """Professional inpainting and outpainting model with state-of-the-art performance. Edit or extend images with natural, seamless results."""

    class Outpaint(str, Enum):
        NONE = "None"
        ZOOM_OUT_1_5X = "Zoom out 1.5x"
        ZOOM_OUT_2X = "Zoom out 2x"
        MAKE_SQUARE = "Make square"
        LEFT_OUTPAINT = "Left outpaint"
        RIGHT_OUTPAINT = "Right outpaint"
        TOP_OUTPAINT = "Top outpaint"
        BOTTOM_OUTPAINT = "Bottom outpaint"

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-fill-pro:2d4197724d8ed13cc78191e794ebbe6aeedcfe4c5b36f464794732d5ccb9735f"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/13571f4b-d677-404f-bff0-ad44da9d5fa0/https___replicate.deli_llwvezd.jpg",
            "created_at": "2024-11-20T20:56:37.431006Z",
            "description": "Professional inpainting and outpainting model with state-of-the-art performance. Edit or extend images with natural, seamless results.",
            "github_url": None,
            "license_url": "https://replicate.com/black-forest-labs/flux-fill-pro#license",
            "name": "flux-fill-pro",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 3662617,
            "url": "https://replicate.com/black-forest-labs/flux-fill-pro",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: str | None = Field(
        title="Mask",
        description="A black-and-white image that describes the part of the image to inpaint. Black areas will be preserved while white areas will be inpainted. Must have the same size as image. Optional if you provide an alpha mask in the original image. Must be jpeg, png, gif, or webp.",
        default=None,
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: str | None = Field(
        title="Image",
        description="The image to inpaint. Can contain an alpha mask. Must be jpeg, png, gif, or webp.",
        default=None,
    )
    steps: int = Field(
        title="Steps",
        description="Number of diffusion steps. Higher values yield finer details but increase processing time.",
        ge=15.0,
        le=50.0,
        default=50,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Controls the balance between adherence to the text prompt and image quality/diversity. Higher values make the output more closely match the prompt but may reduce overall image quality. Lower values allow for more creative freedom but might produce results less relevant to the prompt.",
        ge=1.5,
        le=100.0,
        default=60,
    )
    outpaint: Outpaint = Field(
        description="A quick option for outpainting an input image. Mask will be ignored.",
        default="None",
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="jpg"
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 6 is most permissive",
        ge=1.0,
        le=6.0,
        default=2,
    )
    prompt_upsampling: bool = Field(
        title="Prompt Upsampling",
        description="Automatically modify the prompt for more creative generation",
        default=False,
    )


class Flux_Depth_Dev(ReplicateNode):
    """Open-weight depth-aware image generation. Edit images while preserving spatial relationships."""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"
        MATCH_INPUT = "match_input"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "guidance"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-depth-dev:fc4f1401056237174d207056c49cd2afd44ede232ba286a3d40eb6376b726600"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/4cfef8f5-5fcb-413c-bdaa-d6d4f41e5930/flux-depth-dev.jpg",
            "created_at": "2024-11-20T20:49:48.670385Z",
            "description": "Open-weight depth-aware image generation. Edit images while preserving spatial relationships.",
            "github_url": None,
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev",
            "name": "flux-depth-dev",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 1006962,
            "url": "https://replicate.com/black-forest-labs/flux-depth-dev",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image",
        ge=0.0,
        le=100.0,
        default=10,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image. Use match_input to match the size of the input (with an upper limit of 1440x1440 pixels)",
        default="1",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    control_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image used to control the generation. The depth map will be automatically generated.",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Hyper_Flux_8Step(ReplicateNode):
    """Hyper FLUX 8-step by ByteDance"""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"
        CUSTOM = "custom"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "bytedance/hyper-flux-8step:16084e9731223a4367228928a6cb393b21736da2a0ca6a5a492ce311f0a97143"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/yhqm/bKCAFhWFtbafL6Q31fEkfzVUKDUxY3GcdU1KGtR1AfRhcHOOB/out-0.webp",
            "created_at": "2024-08-27T19:33:25.004679Z",
            "description": "Hyper FLUX 8-step by ByteDance",
            "github_url": "https://github.com/lucataco/cog-hyper-flux-8step",
            "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
            "name": "hyper-flux-8step",
            "owner": "bytedance",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2404.13686",
            "run_count": 20336828,
            "url": "https://replicate.com/bytedance/hyper-flux-8step",
            "visibility": "public",
            "weights_url": "https://huggingface.co/ByteDance/Hyper-SD",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=0,
    )
    width: int = Field(
        title="Width",
        description="Width of the generated image. Optional, only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16)",
        ge=256.0,
        le=1440.0,
        default=848,
    )
    height: int = Field(
        title="Height",
        description="Height of the generated image. Optional, only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16)",
        ge=256.0,
        le=1440.0,
        default=848,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. The size will always be 1 megapixel, i.e. 1024x1024 if aspect ratio is 1:1. To use arbitrary width and height, set aspect ratio to 'custom'.",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale for the diffusion process",
        ge=0.0,
        le=10.0,
        default=3.5,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of inference steps",
        ge=1.0,
        le=30.0,
        default=8,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
        default=False,
    )


class Flux_Mona_Lisa(ReplicateNode):
    """Flux lora, use the term "MNALSA" to trigger generation"""

    class Model(str, Enum):
        DEV = "dev"
        SCHNELL = "schnell"

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"
        CUSTOM = "custom"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/flux-mona-lisa:6e7e34b8d739ab9d4d9a468ef773b5cd85a5c36b11f885379061ba2c70219d41"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/yhqm/apYK6kZFfZUYRyoJ11NzhHY2YXbrjCHajYIiN9EznGR4qVrJA/out-0.webp",
            "created_at": "2024-08-26T21:32:24.116430Z",
            "description": 'Flux lora, use the term "MNALSA" to trigger generation',
            "github_url": None,
            "license_url": None,
            "name": "flux-mona-lisa",
            "owner": "fofr",
            "is_official": False,
            "paper_url": None,
            "run_count": 3598,
            "url": "https://replicate.com/fofr/flux-mona-lisa",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image mask for image inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    model: Model = Field(
        description="Which model to run inference with. The dev model performs best with around 28 inference steps but the schnell model only needs 4 steps.",
        default="dev",
    )
    width: int | None = Field(
        title="Width",
        description="Width of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image.",
        default=None,
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16",
        default=False,
    )
    extra_lora: str | None = Field(
        title="Extra Lora",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. If custom is selected, uses height and width below & will run in bf16 mode",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
        ge=0.0,
        le=10.0,
        default=3,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    extra_lora_scale: float = Field(
        title="Extra Lora Scale",
        description="Determines how strongly the extra LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. More steps can give more detailed images, but take longer.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Cinestill(ReplicateNode):
    """Flux lora, use "CNSTLL" to trigger"""

    class Model(str, Enum):
        DEV = "dev"
        SCHNELL = "schnell"

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"
        CUSTOM = "custom"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "adirik/flux-cinestill:216a43b9975de9768114644bbf8cd0cba54a923c6d0f65adceaccfc9383a938f"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/2838391b-a106-4d81-b0ff-68aa5a0b9367/cinestill.png",
            "created_at": "2024-08-24T10:28:41.876414Z",
            "description": 'Flux lora, use "CNSTLL" to trigger',
            "github_url": None,
            "license_url": None,
            "name": "flux-cinestill",
            "owner": "adirik",
            "is_official": False,
            "paper_url": None,
            "run_count": 129874,
            "url": "https://replicate.com/adirik/flux-cinestill",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image mask for image inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    model: Model = Field(
        description="Which model to run inference with. The dev model performs best with around 28 inference steps but the schnell model only needs 4 steps.",
        default="dev",
    )
    width: int | None = Field(
        title="Width",
        description="Width of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image.",
        default=None,
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16",
        default=False,
    )
    extra_lora: str | None = Field(
        title="Extra Lora",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. If custom is selected, uses height and width below & will run in bf16 mode",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
        ge=0.0,
        le=10.0,
        default=3,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    extra_lora_scale: float = Field(
        title="Extra Lora Scale",
        description="Determines how strongly the extra LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. More steps can give more detailed images, but take longer.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Black_Light(ReplicateNode):
    """A flux lora fine-tuned on black light images"""

    class Model(str, Enum):
        DEV = "dev"
        SCHNELL = "schnell"

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"
        CUSTOM = "custom"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/flux-black-light:d0d48e298dcb51118c3f903817c833bba063936637a33ac52a8ffd6a94859af7"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/yhqm/ZehHgBlLml30R64OxKegcYEHlzGgd0rcIrf5ejXaFK5VuPMNB/out-0.webp",
            "created_at": "2024-08-15T22:48:12.421764Z",
            "description": "A flux lora fine-tuned on black light images",
            "github_url": None,
            "license_url": None,
            "name": "flux-black-light",
            "owner": "fofr",
            "is_official": False,
            "paper_url": None,
            "run_count": 2813487,
            "url": "https://replicate.com/fofr/flux-black-light",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image mask for image inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    model: Model = Field(
        description="Which model to run inference with. The dev model performs best with around 28 inference steps but the schnell model only needs 4 steps.",
        default="dev",
    )
    width: int | None = Field(
        title="Width",
        description="Width of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image.",
        default=None,
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16",
        default=False,
    )
    extra_lora: str | None = Field(
        title="Extra Lora",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. If custom is selected, uses height and width below & will run in bf16 mode",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
        ge=0.0,
        le=10.0,
        default=3,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    extra_lora_scale: float = Field(
        title="Extra Lora Scale",
        description="Determines how strongly the extra LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. More steps can give more detailed images, but take longer.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_360(ReplicateNode):
    """Generate 360 panorama images."""

    class Model(str, Enum):
        DEV = "dev"
        SCHNELL = "schnell"

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"
        CUSTOM = "custom"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "igorriti/flux-360:d26037255a2b298408505e2fbd0bf7703521daca8f07e8c8f335ba874b4aa11a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/3731b4ae-b13e-44e8-a12e-6d02627cbd23/forest.png",
            "created_at": "2024-08-26T15:51:12.459480Z",
            "description": "Generate 360 panorama images.",
            "github_url": "https://github.com/igorriti/pi",
            "license_url": None,
            "name": "flux-360",
            "owner": "igorriti",
            "is_official": False,
            "paper_url": None,
            "run_count": 19603,
            "url": "https://replicate.com/igorriti/flux-360",
            "visibility": "public",
            "weights_url": "https://huggingface.co/igorriti/flux-360",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image mask for image inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image for image to image or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
    )
    model: Model = Field(
        description="Which model to run inference with. The dev model performs best with around 28 inference steps but the schnell model only needs 4 steps.",
        default="dev",
    )
    width: int | None = Field(
        title="Width",
        description="Width of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
        ge=256.0,
        le=1440.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image.",
        default=None,
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16",
        default=False,
    )
    extra_lora: str | None = Field(
        title="Extra Lora",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. If custom is selected, uses height and width below & will run in bf16 mode",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
        ge=0.0,
        le=10.0,
        default=3,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    extra_lora_scale: float = Field(
        title="Extra Lora Scale",
        description="Determines how strongly the extra LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. More steps can give more detailed images, but take longer.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Recraft_V3(ReplicateNode):
    """Recraft V3 (code-named red_panda) is a text-to-image model with the ability to generate long texts, and images in a wide list of styles. As of today, it is SOTA in image generation, proven by the Text-to-Image Benchmark by Artificial Analysis"""

    class Size(str, Enum):
        _1024X1024 = "1024x1024"
        _1365X1024 = "1365x1024"
        _1024X1365 = "1024x1365"
        _1536X1024 = "1536x1024"
        _1024X1536 = "1024x1536"
        _1820X1024 = "1820x1024"
        _1024X1820 = "1024x1820"
        _1024X2048 = "1024x2048"
        _2048X1024 = "2048x1024"
        _1434X1024 = "1434x1024"
        _1024X1434 = "1024x1434"
        _1024X1280 = "1024x1280"
        _1280X1024 = "1280x1024"
        _1024X1707 = "1024x1707"
        _1707X1024 = "1707x1024"

    class Style(str, Enum):
        ANY = "any"
        REALISTIC_IMAGE = "realistic_image"
        DIGITAL_ILLUSTRATION = "digital_illustration"
        DIGITAL_ILLUSTRATION_PIXEL_ART = "digital_illustration/pixel_art"
        DIGITAL_ILLUSTRATION_HAND_DRAWN = "digital_illustration/hand_drawn"
        DIGITAL_ILLUSTRATION_GRAIN = "digital_illustration/grain"
        DIGITAL_ILLUSTRATION_INFANTILE_SKETCH = "digital_illustration/infantile_sketch"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER = "digital_illustration/2d_art_poster"
        DIGITAL_ILLUSTRATION_HANDMADE_3D = "digital_illustration/handmade_3d"
        DIGITAL_ILLUSTRATION_HAND_DRAWN_OUTLINE = (
            "digital_illustration/hand_drawn_outline"
        )
        DIGITAL_ILLUSTRATION_ENGRAVING_COLOR = "digital_illustration/engraving_color"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER_2 = "digital_illustration/2d_art_poster_2"
        REALISTIC_IMAGE_B_AND_W = "realistic_image/b_and_w"
        REALISTIC_IMAGE_HARD_FLASH = "realistic_image/hard_flash"
        REALISTIC_IMAGE_HDR = "realistic_image/hdr"
        REALISTIC_IMAGE_NATURAL_LIGHT = "realistic_image/natural_light"
        REALISTIC_IMAGE_STUDIO_PORTRAIT = "realistic_image/studio_portrait"
        REALISTIC_IMAGE_ENTERPRISE = "realistic_image/enterprise"
        REALISTIC_IMAGE_MOTION_BLUR = "realistic_image/motion_blur"

    class Aspect_ratio(str, Enum):
        NOT_SET = "Not set"
        _1_1 = "1:1"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _1_2 = "1:2"
        _2_1 = "2:1"
        _7_5 = "7:5"
        _5_7 = "5:7"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_5 = "3:5"
        _5_3 = "5:3"

    @classmethod
    def get_basic_fields(cls):
        return ["size", "style", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "recraft-ai/recraft-v3:9507e61ddace8b3a238371b17a61be203747c5081ea6070fecd3c40d27318922"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/a2b66c42-4633-443d-997f-cc987bca07c7/V3.webp",
            "created_at": "2024-10-30T12:41:06.099624Z",
            "description": "Recraft V3 (code-named red_panda) is a text-to-image model with the ability to generate long texts, and images in a wide list of styles. As of today, it is SOTA in image generation, proven by the Text-to-Image Benchmark by Artificial Analysis",
            "github_url": None,
            "license_url": "https://www.recraft.ai/terms",
            "name": "recraft-v3",
            "owner": "recraft-ai",
            "is_official": True,
            "paper_url": "https://recraft.ai",
            "run_count": 7568220,
            "url": "https://replicate.com/recraft-ai/recraft-v3",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    size: Size = Field(
        description="Width and height of the generated image. Size is ignored if an aspect ratio is set.",
        default="1024x1024",
    )
    style: Style = Field(description="Style of the generated image.", default="any")
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="Not set"
    )


class Recraft_20B(ReplicateNode):
    """Affordable and fast images"""

    class Size(str, Enum):
        _1024X1024 = "1024x1024"
        _1365X1024 = "1365x1024"
        _1024X1365 = "1024x1365"
        _1536X1024 = "1536x1024"
        _1024X1536 = "1024x1536"
        _1820X1024 = "1820x1024"
        _1024X1820 = "1024x1820"
        _1024X2048 = "1024x2048"
        _2048X1024 = "2048x1024"
        _1434X1024 = "1434x1024"
        _1024X1434 = "1024x1434"
        _1024X1280 = "1024x1280"
        _1280X1024 = "1280x1024"
        _1024X1707 = "1024x1707"
        _1707X1024 = "1707x1024"

    class Style(str, Enum):
        REALISTIC_IMAGE = "realistic_image"
        REALISTIC_IMAGE_B_AND_W = "realistic_image/b_and_w"
        REALISTIC_IMAGE_ENTERPRISE = "realistic_image/enterprise"
        REALISTIC_IMAGE_HARD_FLASH = "realistic_image/hard_flash"
        REALISTIC_IMAGE_HDR = "realistic_image/hdr"
        REALISTIC_IMAGE_MOTION_BLUR = "realistic_image/motion_blur"
        REALISTIC_IMAGE_NATURAL_LIGHT = "realistic_image/natural_light"
        REALISTIC_IMAGE_STUDIO_PORTRAIT = "realistic_image/studio_portrait"
        DIGITAL_ILLUSTRATION = "digital_illustration"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER = "digital_illustration/2d_art_poster"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER_2 = "digital_illustration/2d_art_poster_2"
        DIGITAL_ILLUSTRATION_3D = "digital_illustration/3d"
        DIGITAL_ILLUSTRATION_80S = "digital_illustration/80s"
        DIGITAL_ILLUSTRATION_ENGRAVING_COLOR = "digital_illustration/engraving_color"
        DIGITAL_ILLUSTRATION_GLOW = "digital_illustration/glow"
        DIGITAL_ILLUSTRATION_GRAIN = "digital_illustration/grain"
        DIGITAL_ILLUSTRATION_HAND_DRAWN = "digital_illustration/hand_drawn"
        DIGITAL_ILLUSTRATION_HAND_DRAWN_OUTLINE = (
            "digital_illustration/hand_drawn_outline"
        )
        DIGITAL_ILLUSTRATION_HANDMADE_3D = "digital_illustration/handmade_3d"
        DIGITAL_ILLUSTRATION_INFANTILE_SKETCH = "digital_illustration/infantile_sketch"
        DIGITAL_ILLUSTRATION_KAWAII = "digital_illustration/kawaii"
        DIGITAL_ILLUSTRATION_PIXEL_ART = "digital_illustration/pixel_art"
        DIGITAL_ILLUSTRATION_PSYCHEDELIC = "digital_illustration/psychedelic"
        DIGITAL_ILLUSTRATION_SEAMLESS = "digital_illustration/seamless"
        DIGITAL_ILLUSTRATION_VOXEL = "digital_illustration/voxel"
        DIGITAL_ILLUSTRATION_WATERCOLOR = "digital_illustration/watercolor"

    class Aspect_ratio(str, Enum):
        NOT_SET = "Not set"
        _1_1 = "1:1"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _1_2 = "1:2"
        _2_1 = "2:1"
        _7_5 = "7:5"
        _5_7 = "5:7"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_5 = "3:5"
        _5_3 = "5:3"

    @classmethod
    def get_basic_fields(cls):
        return ["size", "style", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "recraft-ai/recraft-20b:c303fbbc72c026aa4315e5efc5dd9d8a1dfb60927c84c8c32214cd1d39028701"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/2a808913-4306-43d4-b9da-db154e2faeab/recraft-cover-1.webp",
            "created_at": "2024-12-12T13:39:48.189902Z",
            "description": "Affordable and fast images",
            "github_url": None,
            "license_url": "https://www.recraft.ai/terms",
            "name": "recraft-20b",
            "owner": "recraft-ai",
            "is_official": True,
            "paper_url": "https://recraft.ai",
            "run_count": 304533,
            "url": "https://replicate.com/recraft-ai/recraft-20b",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    size: Size = Field(
        description="Width and height of the generated image. Size is ignored if an aspect ratio is set.",
        default="1024x1024",
    )
    style: Style = Field(
        description="Style of the generated image.", default="realistic_image"
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="Not set"
    )


class Recraft_20B_SVG(ReplicateNode):
    """Affordable and fast vector images"""

    class Size(str, Enum):
        _1024X1024 = "1024x1024"
        _1365X1024 = "1365x1024"
        _1024X1365 = "1024x1365"
        _1536X1024 = "1536x1024"
        _1024X1536 = "1024x1536"
        _1820X1024 = "1820x1024"
        _1024X1820 = "1024x1820"
        _1024X2048 = "1024x2048"
        _2048X1024 = "2048x1024"
        _1434X1024 = "1434x1024"
        _1024X1434 = "1024x1434"
        _1024X1280 = "1024x1280"
        _1280X1024 = "1280x1024"
        _1024X1707 = "1024x1707"
        _1707X1024 = "1707x1024"

    class Style(str, Enum):
        VECTOR_ILLUSTRATION = "vector_illustration"
        VECTOR_ILLUSTRATION_CARTOON = "vector_illustration/cartoon"
        VECTOR_ILLUSTRATION_DOODLE_LINE_ART = "vector_illustration/doodle_line_art"
        VECTOR_ILLUSTRATION_ENGRAVING = "vector_illustration/engraving"
        VECTOR_ILLUSTRATION_FLAT_2 = "vector_illustration/flat_2"
        VECTOR_ILLUSTRATION_KAWAII = "vector_illustration/kawaii"
        VECTOR_ILLUSTRATION_LINE_ART = "vector_illustration/line_art"
        VECTOR_ILLUSTRATION_LINE_CIRCUIT = "vector_illustration/line_circuit"
        VECTOR_ILLUSTRATION_LINOCUT = "vector_illustration/linocut"
        VECTOR_ILLUSTRATION_SEAMLESS = "vector_illustration/seamless"
        ICON = "icon"
        ICON_BROKEN_LINE = "icon/broken_line"
        ICON_COLORED_OUTLINE = "icon/colored_outline"
        ICON_COLORED_SHAPES = "icon/colored_shapes"
        ICON_COLORED_SHAPES_GRADIENT = "icon/colored_shapes_gradient"
        ICON_DOODLE_FILL = "icon/doodle_fill"
        ICON_DOODLE_OFFSET_FILL = "icon/doodle_offset_fill"
        ICON_OFFSET_FILL = "icon/offset_fill"
        ICON_OUTLINE = "icon/outline"
        ICON_OUTLINE_GRADIENT = "icon/outline_gradient"
        ICON_UNEVEN_FILL = "icon/uneven_fill"

    class Aspect_ratio(str, Enum):
        NOT_SET = "Not set"
        _1_1 = "1:1"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _1_2 = "1:2"
        _2_1 = "2:1"
        _7_5 = "7:5"
        _5_7 = "5:7"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_5 = "3:5"
        _5_3 = "5:3"

    @classmethod
    def get_basic_fields(cls):
        return ["size", "style", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "recraft-ai/recraft-20b-svg:666dcf90f18786723e083609cee6c84a0f162cc73d7066fd2d3ad3cb6ba88b1c"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/4993bd89-af70-43da-be68-18cc43437746/recraft-svg-20b-cover.webp",
            "created_at": "2024-12-12T13:39:54.180287Z",
            "description": "Affordable and fast vector images",
            "github_url": None,
            "license_url": "https://www.recraft.ai/terms",
            "name": "recraft-20b-svg",
            "owner": "recraft-ai",
            "is_official": True,
            "paper_url": "https://recraft.ai/",
            "run_count": 86030,
            "url": "https://replicate.com/recraft-ai/recraft-20b-svg",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.SVGRef

    size: Size = Field(
        description="Width and height of the generated image. Size is ignored if an aspect ratio is set.",
        default="1024x1024",
    )
    style: Style = Field(
        description="Style of the generated image.", default="vector_illustration"
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="Not set"
    )


class Recraft_V3_SVG(ReplicateNode):
    """Recraft V3 SVG (code-named red_panda) is a text-to-image model with the ability to generate high quality SVG images including logotypes, and icons. The model supports a wide list of styles."""

    class Size(str, Enum):
        _1024X1024 = "1024x1024"
        _1365X1024 = "1365x1024"
        _1024X1365 = "1024x1365"
        _1536X1024 = "1536x1024"
        _1024X1536 = "1024x1536"
        _1820X1024 = "1820x1024"
        _1024X1820 = "1024x1820"
        _1024X2048 = "1024x2048"
        _2048X1024 = "2048x1024"
        _1434X1024 = "1434x1024"
        _1024X1434 = "1024x1434"
        _1024X1280 = "1024x1280"
        _1280X1024 = "1280x1024"
        _1024X1707 = "1024x1707"
        _1707X1024 = "1707x1024"

    class Style(str, Enum):
        ANY = "any"
        ENGRAVING = "engraving"
        LINE_ART = "line_art"
        LINE_CIRCUIT = "line_circuit"
        LINOCUT = "linocut"

    class Aspect_ratio(str, Enum):
        NOT_SET = "Not set"
        _1_1 = "1:1"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _1_2 = "1:2"
        _2_1 = "2:1"
        _7_5 = "7:5"
        _5_7 = "5:7"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_5 = "3:5"
        _5_3 = "5:3"

    @classmethod
    def get_basic_fields(cls):
        return ["size", "style", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "recraft-ai/recraft-v3-svg:df041379628fa1d16bd406409930775b0904dc2bc0f3e3f38ecd2a4389e9329d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/223c73a9-0347-4daa-9710-3878f95479e3/svg-cover.webp",
            "created_at": "2024-10-30T13:59:33.006694Z",
            "description": "Recraft V3 SVG (code-named red_panda) is a text-to-image model with the ability to generate high quality SVG images including logotypes, and icons. The model supports a wide list of styles.",
            "github_url": None,
            "license_url": "https://recraft.ai/terms",
            "name": "recraft-v3-svg",
            "owner": "recraft-ai",
            "is_official": True,
            "paper_url": "https://recraft.ai",
            "run_count": 347294,
            "url": "https://replicate.com/recraft-ai/recraft-v3-svg",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.SVGRef

    size: Size = Field(
        description="Width and height of the generated image. Size is ignored if an aspect ratio is set.",
        default="1024x1024",
    )
    style: Style = Field(description="Style of the generated image.", default="any")
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="Not set"
    )


class Flux_Canny_Dev(ReplicateNode):
    """Open-weight edge-guided image generation. Control structure and composition using Canny edge detection."""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"
        MATCH_INPUT = "match_input"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "guidance"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-canny-dev:aeb2a8dbfe2580e25d41d8881cc1df1a0b1e52c87de99c1a65fc587ac3918179"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/788cf228-ae38-473a-a992-1e650aab0519/flux-canny-dev.jpg",
            "created_at": "2024-11-20T20:49:32.818286Z",
            "description": "Open-weight edge-guided image generation. Control structure and composition using Canny edge detection.",
            "github_url": None,
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev",
            "name": "flux-canny-dev",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 194591,
            "url": "https://replicate.com/black-forest-labs/flux-canny-dev",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image",
        ge=0.0,
        le=100.0,
        default=30,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image. Use match_input to match the size of the input (with an upper limit of 1440x1440 pixels)",
        default="1",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    control_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image used to control the generation. The canny edge detection will be automatically generated.",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Fill_Dev(ReplicateNode):
    """Open-weight inpainting model for editing and extending images. Guidance-distilled from FLUX.1 Fill [pro]."""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"
        MATCH_INPUT = "match_input"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-fill-dev:a053f84125613d83e65328a289e14eb6639e10725c243e8fb0c24128e5573f4c"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/b109cc9e-f3c2-4899-8428-df46a988c3f0/https___replicate.deliver_tmlMO9j.jpg",
            "created_at": "2024-11-20T20:49:17.667435Z",
            "description": "Open-weight inpainting model for editing and extending images. Guidance-distilled from FLUX.1 Fill [pro].",
            "github_url": None,
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev",
            "name": "flux-fill-dev",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 1372622,
            "url": "https://replicate.com/black-forest-labs/flux-fill-dev",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: str | None = Field(
        title="Mask",
        description="A black-and-white image that describes the part of the image to inpaint. Black areas will be preserved while white areas will be inpainted.",
        default=None,
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: str | None = Field(
        title="Image",
        description="The image to inpaint. Can contain alpha mask. If the image width or height are not multiples of 32, they will be scaled to the closest multiple of 32. If the image dimensions don't fit within 1440x1440, it will be scaled down to fit.",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image",
        ge=0.0,
        le=100.0,
        default=30,
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        ge=-1.0,
        le=3.0,
        default=1,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image. Use match_input to match the size of the input (with an upper limit of 1440x1440 pixels)",
        default="1",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    lora_weights: str | None = Field(
        title="Lora Weights",
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Redux_Schnell(ReplicateNode):
    """Fast, efficient image variation model for rapid iteration and experimentation."""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "megapixels", "num_outputs"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-redux-schnell:8a9ff6ce228b950c7079005fd0804f54c74c0113cda3f3c07eff10ab943f32a1"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/828d2d38-465e-4a65-a979-1f661000e978/https___replicate.deliver_018es9x.jpg",
            "created_at": "2024-11-20T22:30:41.021241Z",
            "description": "Fast, efficient image variation model for rapid iteration and experimentation.",
            "github_url": None,
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-schnell",
            "name": "flux-redux-schnell",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 66945,
            "url": "https://replicate.com/black-forest-labs/flux-redux-schnell",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    redux_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image to condition your output on. This replaces prompt for FLUX.1 Redux models",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. 4 is recommended, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=4.0,
        default=4,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_Redux_Dev(ReplicateNode):
    """Open-weight image variation model. Create new versions while preserving key elements of your original."""

    class Megapixels(str, Enum):
        _1 = "1"
        _0_25 = "0.25"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _21_9 = "21:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _9_16 = "9:16"
        _9_21 = "9:21"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "guidance", "megapixels"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-redux-dev:96b56814e57dfa601f3f524f82a2b336ef49012cda68828cb37cde66f481b7cb"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/daff59ba-540d-4111-a969-9119ee814f26/redux-cover.jpg",
            "created_at": "2024-11-20T22:29:45.623102Z",
            "description": "Open-weight image variation model. Create new versions while preserving key elements of your original.",
            "github_url": None,
            "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev",
            "name": "flux-redux-dev",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 286344,
            "url": "https://replicate.com/black-forest-labs/flux-redux-dev",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image",
        ge=0.0,
        le=10.0,
        default=3,
    )
    megapixels: Megapixels = Field(
        description="Approximate number of megapixels for generated image", default="1"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of outputs to generate",
        ge=1.0,
        le=4.0,
        default=1,
    )
    redux_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input image to condition your output on. This replaces prompt for FLUX.1 Redux models",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="1:1"
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50",
        ge=1.0,
        le=50.0,
        default=28,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class SDXL_Controlnet(ReplicateNode):
    """SDXL ControlNet - Canny"""

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "lucataco/sdxl-controlnet:06d6fae3b75ab68a28cd2900afa6033166910dd09fd9751047043a5bbb4c184b"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/7edf6f87-bd0d-4a4f-9e11-d944bb07a3ea/output.png",
            "created_at": "2023-08-14T07:15:37.417194Z",
            "description": "SDXL ControlNet - Canny",
            "github_url": "https://github.com/lucataco/cog-sdxl-controlnet",
            "license_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
            "name": "sdxl-controlnet",
            "owner": "lucataco",
            "is_official": False,
            "paper_url": None,
            "run_count": 3442595,
            "url": "https://replicate.com/lucataco/sdxl-controlnet",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int = Field(
        title="Seed",
        description="Random seed. Set to 0 to randomize the seed",
        default=0,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
    )
    condition_scale: float = Field(
        title="Condition Scale",
        description="controlnet conditioning scale for generalization",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Input Negative Prompt",
        default="low quality, bad quality, sketches",
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )


class SDXL_Ad_Inpaint(ReplicateNode):
    """Product advertising image generator using SDXL"""

    class Img_size(str, Enum):
        _512__2048 = "512, 2048"
        _512__1984 = "512, 1984"
        _512__1920 = "512, 1920"
        _512__1856 = "512, 1856"
        _576__1792 = "576, 1792"
        _576__1728 = "576, 1728"
        _576__1664 = "576, 1664"
        _640__1600 = "640, 1600"
        _640__1536 = "640, 1536"
        _704__1472 = "704, 1472"
        _704__1408 = "704, 1408"
        _704__1344 = "704, 1344"
        _768__1344 = "768, 1344"
        _768__1280 = "768, 1280"
        _832__1216 = "832, 1216"
        _832__1152 = "832, 1152"
        _896__1152 = "896, 1152"
        _896__1088 = "896, 1088"
        _960__1088 = "960, 1088"
        _960__1024 = "960, 1024"
        _1024__1024 = "1024, 1024"
        _1024__960 = "1024, 960"
        _1088__960 = "1088, 960"
        _1088__896 = "1088, 896"
        _1152__896 = "1152, 896"
        _1152__832 = "1152, 832"
        _1216__832 = "1216, 832"
        _1280__768 = "1280, 768"
        _1344__768 = "1344, 768"
        _1408__704 = "1408, 704"
        _1472__704 = "1472, 704"
        _1536__640 = "1536, 640"
        _1600__640 = "1600, 640"
        _1664__576 = "1664, 576"
        _1728__576 = "1728, 576"
        _1792__576 = "1792, 576"
        _1856__512 = "1856, 512"
        _1920__512 = "1920, 512"
        _1984__512 = "1984, 512"
        _2048__512 = "2048, 512"

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        HEUNDISCRETE = "HeunDiscrete"
        KARRASDPM = "KarrasDPM"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        K_EULER = "K_EULER"
        PNDM = "PNDM"

    class Product_fill(str, Enum):
        ORIGINAL = "Original"
        _80 = "80"
        _70 = "70"
        _60 = "60"
        _50 = "50"
        _40 = "40"
        _30 = "30"
        _20 = "20"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "catacolabs/sdxl-ad-inpaint:9c0cb4c579c54432431d96c70924afcca18983de872e8a221777fb1416253359"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://pbxt.replicate.delivery/ORbuWtoy0y6NI9f4DrJ2fxs92LgviBaOlzOVdYTr3pT8eKJjA/7-out.png",
            "created_at": "2023-09-15T15:37:19.970710Z",
            "description": "Product advertising image generator using SDXL",
            "github_url": "https://github.com/CatacoLabs/cog-sdxl-ad-inpaint",
            "license_url": "https://github.com/huggingface/hfapi/blob/master/LICENSE",
            "name": "sdxl-ad-inpaint",
            "owner": "catacolabs",
            "is_official": False,
            "paper_url": None,
            "run_count": 401402,
            "url": "https://replicate.com/catacolabs/sdxl-ad-inpaint",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed", description="Empty or 0 for a random image", default=None
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Remove background from this image"
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Describe the new setting for your product",
        default=None,
    )
    img_size: Img_size = Field(
        description="Possible SDXL image sizes", default="1024, 1024"
    )
    apply_img: bool = Field(
        title="Apply Img",
        description="Applies the original product image to the final result",
        default=True,
    )
    scheduler: Scheduler = Field(description="scheduler", default="K_EULER")
    product_fill: Product_fill = Field(
        description="What percentage of the image width to fill with product",
        default="Original",
    )
    guidance_scale: float = Field(
        title="Guidance Scale", description="Guidance Scale", default=7.5
    )
    condition_scale: float = Field(
        title="Condition Scale",
        description="controlnet conditioning scale for generalization",
        ge=0.3,
        le=0.9,
        default=0.9,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Describe what you do not want in your setting",
        default="low quality, out of frame, illustration, 3d, sepia, painting, cartoons, sketch, watermark, text, Logo, advertisement",
    )
    num_refine_steps: int = Field(
        title="Num Refine Steps",
        description="Number of steps to refine",
        ge=0.0,
        le=40.0,
        default=10,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps", description="Inference Steps", default=40
    )


class Kandinsky(ReplicateNode):
    """multilingual text2image latent diffusion model"""

    class Width(int, Enum):
        _384 = 384
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _960 = 960
        _1024 = 1024
        _1152 = 1152
        _1280 = 1280
        _1536 = 1536
        _1792 = 1792
        _2048 = 2048

    class Height(int, Enum):
        _384 = 384
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _960 = 960
        _1024 = 1024
        _1152 = 1152
        _1280 = 1280
        _1536 = 1536
        _1792 = 1792
        _2048 = 2048

    class Output_format(str, Enum):
        WEBP = "webp"
        JPEG = "jpeg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "ai-forever/kandinsky-2.2:ad9d7879fbffa2874e1d909d1d37d9bc682889cc65b31f7bb00d2362619f194a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/618e68d3-fba3-4fd0-a060-cdd46b2ab7cf/out-0_2.jpg",
            "created_at": "2023-07-12T21:53:29.439515Z",
            "description": "multilingual text2image latent diffusion model",
            "github_url": "https://github.com/chenxwh/Kandinsky-2/tree/v2.2",
            "license_url": "https://github.com/ai-forever/Kandinsky-2/blob/main/license",
            "name": "kandinsky-2.2",
            "owner": "ai-forever",
            "is_official": False,
            "paper_url": None,
            "run_count": 10028882,
            "url": "https://replicate.com/ai-forever/kandinsky-2.2",
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
    width: Width = Field(
        description="Width of output image. Lower the setting if hits memory limits.",
        default=512,
    )
    height: Height = Field(
        description="Height of output image. Lower the setting if hits memory limits.",
        default=512,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="A moss covered astronaut with a black background",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    output_format: Output_format = Field(
        description="Output image format", default="webp"
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Specify things to not see in the output",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=75,
    )
    num_inference_steps_prior: int = Field(
        title="Num Inference Steps Prior",
        description="Number of denoising steps for priors",
        ge=1.0,
        le=500.0,
        default=25,
    )


class StableDiffusionXLLightning(ReplicateNode):
    """SDXL-Lightning by ByteDance: a fast text-to-image model that makes high-quality images in 4 steps"""

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        HEUNDISCRETE = "HeunDiscrete"
        KARRASDPM = "KarrasDPM"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        K_EULER = "K_EULER"
        PNDM = "PNDM"
        DPM__2MSDE = "DPM++2MSDE"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "bytedance/sdxl-lightning-4step:6f7a773af6fc3e8de9d5a3c00be77c17308914bf67772726aff83496ba1e3bbe"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/779f3f58-c3db-4403-a01b-3ffed97a1449/out-0-1.jpg",
            "created_at": "2024-02-21T07:36:15.534380Z",
            "description": "SDXL-Lightning by ByteDance: a fast text-to-image model that makes high-quality images in 4 steps",
            "github_url": "https://github.com/lucataco/cog-sdxl-lightning-4step",
            "license_url": "https://huggingface.co/ByteDance/SDXL-Lightning/blob/main/LICENSE.md",
            "name": "sdxl-lightning-4step",
            "owner": "bytedance",
            "is_official": False,
            "paper_url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_report.pdf",
            "run_count": 1031625592,
            "url": "https://replicate.com/bytedance/sdxl-lightning-4step",
            "visibility": "public",
            "weights_url": "https://huggingface.co/ByteDance/SDXL-Lightning",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=0,
    )
    width: int = Field(
        title="Width",
        description="Width of output image. Recommended 1024 or 1280",
        ge=256.0,
        le=1280.0,
        default=1024,
    )
    height: int = Field(
        title="Height",
        description="Height of output image. Recommended 1024 or 1280",
        ge=256.0,
        le=1280.0,
        default=1024,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="self-portrait of a woman, lightning in the background",
    )
    scheduler: Scheduler = Field(description="scheduler", default="K_EULER")
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=0.0,
        le=50.0,
        default=0,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative Input prompt",
        default="worst quality, low quality",
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. 4 for best results",
        ge=1.0,
        le=10.0,
        default=4,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images",
        default=False,
    )


class PlaygroundV2(ReplicateNode):
    """Playground v2.5 is the state-of-the-art open-source model in aesthetic quality"""

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        HEUNDISCRETE = "HeunDiscrete"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        K_EULER = "K_EULER"
        PNDM = "PNDM"
        DPM__2MKARRAS = "DPM++2MKarras"
        DPMSOLVER = "DPMSolver++"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/b849582a-8699-4965-8016-3a51dc1da3d4/playground.jpeg",
            "created_at": "2024-02-27T22:20:16.107222Z",
            "description": "Playground v2.5 is the state-of-the-art open-source model in aesthetic quality",
            "github_url": "https://github.com/lucataco/cog-playground-v2.5-1024px-aesthetic",
            "license_url": "https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md",
            "name": "playground-v2.5-1024px-aesthetic",
            "owner": "playgroundai",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2206.00364",
            "run_count": 2852617,
            "url": "https://replicate.com/playgroundai/playground-v2.5-1024px-aesthetic",
            "visibility": "public",
            "weights_url": "https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: str | None = Field(
        title="Mask",
        description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
        default=None,
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    width: int = Field(
        title="Width",
        description="Width of output image",
        ge=256.0,
        le=1536.0,
        default=1024,
    )
    height: int = Field(
        title="Height",
        description="Height of output image",
        ge=256.0,
        le=1536.0,
        default=1024,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )
    scheduler: Scheduler = Field(
        description="Scheduler. DPMSolver++ or DPM++2MKarras is recommended for most cases",
        default="DPMSolver++",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=0.1,
        le=20.0,
        default=3,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative Input prompt",
        default="ugly, deformed, noisy, blurry, distorted",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=60.0,
        default=25,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See https://replicate.com/docs/how-does-replicate-work#safety",
        default=False,
    )


class Proteus_V_02(ReplicateNode):
    """Proteus v0.2 shows subtle yet significant improvements over Version 0.1. It demonstrates enhanced prompt understanding that surpasses MJ6, while also approaching its stylistic capabilities."""

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
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "datacte/proteus-v0.2:06775cd262843edbde5abab958abdbb65a0a6b58ca301c9fd78fa55c775fc019"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/pbxt/1nrcrEszpsb0Kpv0qNBJrtQjoefjHJ3xSh3whVOJcklSFxPSA/out-0.png",
            "created_at": "2024-01-24T17:45:49.361192Z",
            "description": "Proteus v0.2 shows subtle yet significant improvements over Version 0.1. It demonstrates enhanced prompt understanding that surpasses MJ6, while also approaching its stylistic capabilities.",
            "github_url": "https://github.com/lucataco/cog-proteus-v0.2",
            "license_url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/gpl-3.0.md",
            "name": "proteus-v0.2",
            "owner": "datacte",
            "is_official": False,
            "paper_url": None,
            "run_count": 11521152,
            "url": "https://replicate.com/datacte/proteus-v0.2",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    width: int = Field(title="Width", description="Width of output image", default=1024)
    height: int = Field(
        title="Height", description="Height of output image", default=1024
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="black fluffy gorgeous dangerous cat animal creature, large orange eyes, big fluffy ears, piercing gaze, full moon, dark ambiance, best quality, extremely detailed",
    )
    scheduler: Scheduler = Field(description="scheduler", default="KarrasDPM")
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance. Recommended 7-8",
        ge=1.0,
        le=50.0,
        default=7.5,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative Input prompt",
        default="worst quality, low quality",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. 20 to 35 steps for more detail, 20 steps for faster results.",
        ge=1.0,
        le=100.0,
        default=20,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See https://replicate.com/docs/how-does-replicate-work#safety",
        default=False,
    )


class Proteus_V_03(ReplicateNode):
    """ProteusV0.3: The Anime Update"""

    class Scheduler(str, Enum):
        DDIM = "DDIM"
        DPMSOLVERMULTISTEP = "DPMSolverMultistep"
        HEUNDISCRETE = "HeunDiscrete"
        KARRASDPM = "KarrasDPM"
        K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
        K_EULER = "K_EULER"
        PNDM = "PNDM"
        DPM__2MSDE = "DPM++2MSDE"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "datacte/proteus-v0.3:b28b79d725c8548b173b6a19ff9bffd16b9b80df5b18b8dc5cb9e1ee471bfa48"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/pbxt/C3LYYa30997dKRdeNDSXNjIK01CH5q8CSto12eWundnPPtWSA/out-0.png",
            "created_at": "2024-02-14T20:02:04.901849Z",
            "description": "ProteusV0.3: The Anime Update",
            "github_url": "https://github.com/lucataco/cog-proteus-v0.3",
            "license_url": "https://huggingface.co/models?license=license:gpl-3.0",
            "name": "proteus-v0.3",
            "owner": "datacte",
            "is_official": False,
            "paper_url": None,
            "run_count": 5145505,
            "url": "https://replicate.com/datacte/proteus-v0.3",
            "visibility": "public",
            "weights_url": "https://huggingface.co/dataautogpt3/ProteusV0.3",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img or inpaint mode"
    )
    width: int = Field(
        title="Width",
        description="Width of output image. Recommended 1024 or 1280",
        default=1024,
    )
    height: int = Field(
        title="Height",
        description="Height of output image. Recommended 1024 or 1280",
        default=1024,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="Anime full body portrait of a swordsman holding his weapon in front of him. He is facing the camera with a fierce look on his face. Anime key visual (best quality, HD, ~+~aesthetic~+~:1.2)",
    )
    scheduler: Scheduler = Field(description="scheduler", default="DPM++2MSDE")
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output.",
        ge=1.0,
        le=4.0,
        default=1,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance. Recommended 7-8",
        ge=1.0,
        le=50.0,
        default=7.5,
    )
    apply_watermark: bool = Field(
        title="Apply Watermark",
        description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        default=True,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative Input prompt",
        default="worst quality, low quality",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. 20 to 60 steps for more detail, 20 steps for faster results.",
        ge=1.0,
        le=100.0,
        default=20,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images. This feature is only available through the API. See https://replicate.com/docs/how-does-replicate-work#safety",
        default=False,
    )


class StickerMaker(ReplicateNode):
    """Make stickers with AI. Generates graphics with transparent backgrounds."""

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "steps", "width"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/sticker-maker:4acb778eb059772225ec213948f0660867b2e03f277448f18cf1800b96a65a1a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/fb7cf2ea-aacd-458d-9d19-76dda21f9748/sticker-maker.webp",
            "created_at": "2024-02-23T11:59:22.452180Z",
            "description": "Make stickers with AI. Generates graphics with transparent backgrounds.",
            "github_url": "https://github.com/fofr/cog-stickers",
            "license_url": "https://github.com/fofr/cog-stickers/blob/main/LICENSE",
            "name": "sticker-maker",
            "owner": "fofr",
            "is_official": False,
            "paper_url": None,
            "run_count": 1890531,
            "url": "https://replicate.com/fofr/sticker-maker",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Fix the random seed for reproducibility",
        default=None,
    )
    steps: int = Field(title="Steps", default=17)
    width: int = Field(title="Width", default=1152)
    height: int = Field(title="Height", default=1152)
    prompt: str = Field(title="Prompt", default="a cute cat")
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
        ge=0.0,
        le=100.0,
        default=90,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want in the image",
        default="",
    )
    number_of_images: int = Field(
        title="Number Of Images",
        description="Number of images to generate",
        ge=1.0,
        le=10.0,
        default=1,
    )


class StyleTransfer(ReplicateNode):
    """Transfer the style of one image to another"""

    class Model(str, Enum):
        FAST = "fast"
        HIGH_QUALITY = "high-quality"
        REALISTIC = "realistic"
        CINEMATIC = "cinematic"
        ANIMATED = "animated"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "model", "width"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/style-transfer:f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/fd0ac369-c6ac-4927-b882-ece29cffc45d/cover.webp",
            "created_at": "2024-04-17T20:34:49.861066Z",
            "description": "Transfer the style of one image to another",
            "github_url": "https://github.com/fofr/cog-style-transfer",
            "license_url": "https://github.com/fofr/cog-style-transfer/blob/main/LICENSE",
            "name": "style-transfer",
            "owner": "fofr",
            "is_official": False,
            "paper_url": None,
            "run_count": 1274398,
            "url": "https://replicate.com/fofr/style-transfer",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    model: Model = Field(description="Model to use for the generation", default="fast")
    width: int = Field(
        title="Width",
        description="Width of the output image (ignored if structure image given)",
        default=1024,
    )
    height: int = Field(
        title="Height",
        description="Height of the output image (ignored if structure image given)",
        default=1024,
    )
    prompt: str = Field(
        title="Prompt",
        description="Prompt for the image",
        default="An astronaut riding a unicorn",
    )
    style_image: types.ImageRef = Field(
        default=types.ImageRef(), description="Copy the style from this image"
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
        ge=0.0,
        le=100.0,
        default=80,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want to see in your image",
        default="",
    )
    structure_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="An optional image to copy structure from. Output images will use the same aspect ratio.",
    )
    number_of_images: int = Field(
        title="Number Of Images",
        description="Number of images to generate",
        ge=1.0,
        le=10.0,
        default=1,
    )
    structure_depth_strength: float = Field(
        title="Structure Depth Strength",
        description="Strength of the depth controlnet",
        ge=0.0,
        le=2.0,
        default=1,
    )
    structure_denoising_strength: float = Field(
        title="Structure Denoising Strength",
        description="How much of the original image (and colors) to preserve (0 is all, 1 is none, 0.65 is a good balance)",
        ge=0.0,
        le=1.0,
        default=0.65,
    )


class Illusions(ReplicateNode):
    """Create illusions with img2img and masking support"""

    class Sizing_strategy(str, Enum):
        WIDTH_HEIGHT = "width/height"
        INPUT_IMAGE = "input_image"
        CONTROL_IMAGE = "control_image"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "width"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/illusions:579b32db82b24584c3c6155fe3ae12e8fce50ba28b575c23e8a1f5f3a5e99ed8"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/iA3MUV1f533LXqi4sHVlAerJhr03uES9OlBHAHrqaeq92ArnA/output-0.png",
            "created_at": "2023-11-03T17:24:31.993569Z",
            "description": "Create illusions with img2img and masking support",
            "github_url": "https://github.com/fofr/cog-illusion",
            "license_url": None,
            "name": "illusions",
            "owner": "fofr",
            "is_official": False,
            "paper_url": None,
            "run_count": 61563,
            "url": "https://replicate.com/fofr/illusions",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(title="Seed", default=None)
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Optional img2img"
    )
    width: int = Field(title="Width", default=768)
    height: int = Field(title="Height", default=768)
    prompt: str = Field(title="Prompt", default="a painting of a 19th century town")
    mask_image: types.ImageRef = Field(
        default=types.ImageRef(), description="Optional mask for inpainting"
    )
    num_outputs: int = Field(
        title="Num Outputs", description="Number of outputs", default=1
    )
    control_image: types.ImageRef = Field(
        default=types.ImageRef(), description="Control image"
    )
    controlnet_end: float = Field(
        title="Controlnet End",
        description="When controlnet conditioning ends",
        default=1.0,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        default=7.5,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="The negative prompt to guide image generation.",
        default="ugly, disfigured, low quality, blurry, nsfw",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
        default=0.8,
    )
    sizing_strategy: Sizing_strategy = Field(
        description="Decide how to resize images – use width/height, resize based on input image or control image",
        default="width/height",
    )
    controlnet_start: float = Field(
        title="Controlnet Start",
        description="When controlnet conditioning starts",
        default=0.0,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps", description="Number of diffusion steps", default=40
    )
    controlnet_conditioning_scale: float = Field(
        title="Controlnet Conditioning Scale",
        description="How strong the controlnet conditioning is",
        default=0.75,
    )


class Ideogram_V2(ReplicateNode):
    """An excellent image model with state of the art inpainting, prompt comprehension and text rendering"""

    class Resolution(str, Enum):
        NONE = "None"
        _512X1536 = "512x1536"
        _576X1408 = "576x1408"
        _576X1472 = "576x1472"
        _576X1536 = "576x1536"
        _640X1344 = "640x1344"
        _640X1408 = "640x1408"
        _640X1472 = "640x1472"
        _640X1536 = "640x1536"
        _704X1152 = "704x1152"
        _704X1216 = "704x1216"
        _704X1280 = "704x1280"
        _704X1344 = "704x1344"
        _704X1408 = "704x1408"
        _704X1472 = "704x1472"
        _736X1312 = "736x1312"
        _768X1088 = "768x1088"
        _768X1216 = "768x1216"
        _768X1280 = "768x1280"
        _768X1344 = "768x1344"
        _832X960 = "832x960"
        _832X1024 = "832x1024"
        _832X1088 = "832x1088"
        _832X1152 = "832x1152"
        _832X1216 = "832x1216"
        _832X1248 = "832x1248"
        _864X1152 = "864x1152"
        _896X960 = "896x960"
        _896X1024 = "896x1024"
        _896X1088 = "896x1088"
        _896X1120 = "896x1120"
        _896X1152 = "896x1152"
        _960X832 = "960x832"
        _960X896 = "960x896"
        _960X1024 = "960x1024"
        _960X1088 = "960x1088"
        _1024X832 = "1024x832"
        _1024X896 = "1024x896"
        _1024X960 = "1024x960"
        _1024X1024 = "1024x1024"
        _1088X768 = "1088x768"
        _1088X832 = "1088x832"
        _1088X896 = "1088x896"
        _1088X960 = "1088x960"
        _1120X896 = "1120x896"
        _1152X704 = "1152x704"
        _1152X832 = "1152x832"
        _1152X864 = "1152x864"
        _1152X896 = "1152x896"
        _1216X704 = "1216x704"
        _1216X768 = "1216x768"
        _1216X832 = "1216x832"
        _1248X832 = "1248x832"
        _1280X704 = "1280x704"
        _1280X768 = "1280x768"
        _1280X800 = "1280x800"
        _1312X736 = "1312x736"
        _1344X640 = "1344x640"
        _1344X704 = "1344x704"
        _1344X768 = "1344x768"
        _1408X576 = "1408x576"
        _1408X640 = "1408x640"
        _1408X704 = "1408x704"
        _1472X576 = "1472x576"
        _1472X640 = "1472x640"
        _1472X704 = "1472x704"
        _1536X512 = "1536x512"
        _1536X576 = "1536x576"
        _1536X640 = "1536x640"

    class Style_type(str, Enum):
        NONE = "None"
        AUTO = "Auto"
        GENERAL = "General"
        REALISTIC = "Realistic"
        DESIGN = "Design"
        RENDER_3D = "Render 3D"
        ANIME = "Anime"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_10 = "16:10"
        _10_16 = "10:16"
        _3_1 = "3:1"
        _1_3 = "1:3"

    class Magic_prompt_option(str, Enum):
        AUTO = "Auto"
        ON = "On"
        OFF = "Off"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "ideogram-ai/ideogram-v2:3e6071946ab5319b3bcc37a4d00083e743dfdff5be386df6a2ff1f212fc7365b"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/71c982a3-27f0-42a6-ad6a-769f25097c08/replicate-prediction-s_VROPz1s.png",
            "created_at": "2024-10-22T09:26:23.607119Z",
            "description": "An excellent image model with state of the art inpainting, prompt comprehension and text rendering",
            "github_url": None,
            "license_url": "https://about.ideogram.ai/legal/api-tos",
            "name": "ideogram-v2",
            "owner": "ideogram-ai",
            "is_official": True,
            "paper_url": "https://ideogram.ai/",
            "run_count": 2627578,
            "url": "https://replicate.com/ideogram-ai/ideogram-v2",
            "visibility": "public",
            "weights_url": "https://ideogram.ai/",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="A black and white image. Black pixels are inpainted, white pixels are preserved. The mask will be resized to match the image size.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        le=2147483647.0,
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="An image file to use for inpainting. You must also use a mask.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    resolution: Resolution = Field(
        description="Resolution. Overrides aspect ratio. Ignored if an inpainting image is given.",
        default="None",
    )
    style_type: Style_type = Field(
        description="The styles help define the specific aesthetic of the image you want to generate.",
        default="None",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio. Ignored if a resolution or inpainting image is given.",
        default="1:1",
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Things you do not want to see in the generated image.",
        default=None,
    )
    magic_prompt_option: Magic_prompt_option = Field(
        description="Magic Prompt will interpret your prompt and optimize it to maximize variety and quality of the images generated. You can also use it to write prompts in different languages.",
        default="Auto",
    )


class Ideogram_V2_Turbo(ReplicateNode):
    """A fast image model with state of the art inpainting, prompt comprehension and text rendering."""

    class Resolution(str, Enum):
        NONE = "None"
        _512X1536 = "512x1536"
        _576X1408 = "576x1408"
        _576X1472 = "576x1472"
        _576X1536 = "576x1536"
        _640X1344 = "640x1344"
        _640X1408 = "640x1408"
        _640X1472 = "640x1472"
        _640X1536 = "640x1536"
        _704X1152 = "704x1152"
        _704X1216 = "704x1216"
        _704X1280 = "704x1280"
        _704X1344 = "704x1344"
        _704X1408 = "704x1408"
        _704X1472 = "704x1472"
        _736X1312 = "736x1312"
        _768X1088 = "768x1088"
        _768X1216 = "768x1216"
        _768X1280 = "768x1280"
        _768X1344 = "768x1344"
        _832X960 = "832x960"
        _832X1024 = "832x1024"
        _832X1088 = "832x1088"
        _832X1152 = "832x1152"
        _832X1216 = "832x1216"
        _832X1248 = "832x1248"
        _864X1152 = "864x1152"
        _896X960 = "896x960"
        _896X1024 = "896x1024"
        _896X1088 = "896x1088"
        _896X1120 = "896x1120"
        _896X1152 = "896x1152"
        _960X832 = "960x832"
        _960X896 = "960x896"
        _960X1024 = "960x1024"
        _960X1088 = "960x1088"
        _1024X832 = "1024x832"
        _1024X896 = "1024x896"
        _1024X960 = "1024x960"
        _1024X1024 = "1024x1024"
        _1088X768 = "1088x768"
        _1088X832 = "1088x832"
        _1088X896 = "1088x896"
        _1088X960 = "1088x960"
        _1120X896 = "1120x896"
        _1152X704 = "1152x704"
        _1152X832 = "1152x832"
        _1152X864 = "1152x864"
        _1152X896 = "1152x896"
        _1216X704 = "1216x704"
        _1216X768 = "1216x768"
        _1216X832 = "1216x832"
        _1248X832 = "1248x832"
        _1280X704 = "1280x704"
        _1280X768 = "1280x768"
        _1280X800 = "1280x800"
        _1312X736 = "1312x736"
        _1344X640 = "1344x640"
        _1344X704 = "1344x704"
        _1344X768 = "1344x768"
        _1408X576 = "1408x576"
        _1408X640 = "1408x640"
        _1408X704 = "1408x704"
        _1472X576 = "1472x576"
        _1472X640 = "1472x640"
        _1472X704 = "1472x704"
        _1536X512 = "1536x512"
        _1536X576 = "1536x576"
        _1536X640 = "1536x640"

    class Style_type(str, Enum):
        NONE = "None"
        AUTO = "Auto"
        GENERAL = "General"
        REALISTIC = "Realistic"
        DESIGN = "Design"
        RENDER_3D = "Render 3D"
        ANIME = "Anime"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_10 = "16:10"
        _10_16 = "10:16"
        _3_1 = "3:1"
        _1_3 = "1:3"

    class Magic_prompt_option(str, Enum):
        AUTO = "Auto"
        ON = "On"
        OFF = "Off"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "ideogram-ai/ideogram-v2-turbo:7cef9d520d672bb802588ad0d13151bc51aee9a408c270aebf25d6530045dd29"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8b1df940-d741-4446-beb2-0d72c66abb91/replicate-prediction-f_iX48w8f.png",
            "created_at": "2024-10-22T09:29:41.547244Z",
            "description": "A fast image model with state of the art inpainting, prompt comprehension and text rendering.",
            "github_url": None,
            "license_url": "https://about.ideogram.ai/legal/api-tos",
            "name": "ideogram-v2-turbo",
            "owner": "ideogram-ai",
            "is_official": True,
            "paper_url": "https://ideogram.ai/",
            "run_count": 2863440,
            "url": "https://replicate.com/ideogram-ai/ideogram-v2-turbo",
            "visibility": "public",
            "weights_url": "https://ideogram.ai/",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="A black and white image. Black pixels are inpainted, white pixels are preserved. The mask will be resized to match the image size.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        le=2147483647.0,
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="An image file to use for inpainting. You must also use a mask.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    resolution: Resolution = Field(
        description="Resolution. Overrides aspect ratio. Ignored if an inpainting image is given.",
        default="None",
    )
    style_type: Style_type = Field(
        description="The styles help define the specific aesthetic of the image you want to generate.",
        default="None",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio. Ignored if a resolution or inpainting image is given.",
        default="1:1",
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Things you do not want to see in the generated image.",
        default=None,
    )
    magic_prompt_option: Magic_prompt_option = Field(
        description="Magic Prompt will interpret your prompt and optimize it to maximize variety and quality of the images generated. You can also use it to write prompts in different languages.",
        default="Auto",
    )


class Ideogram_V2A(ReplicateNode):
    """Like Ideogram v2, but faster and cheaper"""

    class Resolution(str, Enum):
        NONE = "None"
        _512X1536 = "512x1536"
        _576X1408 = "576x1408"
        _576X1472 = "576x1472"
        _576X1536 = "576x1536"
        _640X1344 = "640x1344"
        _640X1408 = "640x1408"
        _640X1472 = "640x1472"
        _640X1536 = "640x1536"
        _704X1152 = "704x1152"
        _704X1216 = "704x1216"
        _704X1280 = "704x1280"
        _704X1344 = "704x1344"
        _704X1408 = "704x1408"
        _704X1472 = "704x1472"
        _736X1312 = "736x1312"
        _768X1088 = "768x1088"
        _768X1216 = "768x1216"
        _768X1280 = "768x1280"
        _768X1344 = "768x1344"
        _832X960 = "832x960"
        _832X1024 = "832x1024"
        _832X1088 = "832x1088"
        _832X1152 = "832x1152"
        _832X1216 = "832x1216"
        _832X1248 = "832x1248"
        _864X1152 = "864x1152"
        _896X960 = "896x960"
        _896X1024 = "896x1024"
        _896X1088 = "896x1088"
        _896X1120 = "896x1120"
        _896X1152 = "896x1152"
        _960X832 = "960x832"
        _960X896 = "960x896"
        _960X1024 = "960x1024"
        _960X1088 = "960x1088"
        _1024X832 = "1024x832"
        _1024X896 = "1024x896"
        _1024X960 = "1024x960"
        _1024X1024 = "1024x1024"
        _1088X768 = "1088x768"
        _1088X832 = "1088x832"
        _1088X896 = "1088x896"
        _1088X960 = "1088x960"
        _1120X896 = "1120x896"
        _1152X704 = "1152x704"
        _1152X832 = "1152x832"
        _1152X864 = "1152x864"
        _1152X896 = "1152x896"
        _1216X704 = "1216x704"
        _1216X768 = "1216x768"
        _1216X832 = "1216x832"
        _1248X832 = "1248x832"
        _1280X704 = "1280x704"
        _1280X768 = "1280x768"
        _1280X800 = "1280x800"
        _1312X736 = "1312x736"
        _1344X640 = "1344x640"
        _1344X704 = "1344x704"
        _1344X768 = "1344x768"
        _1408X576 = "1408x576"
        _1408X640 = "1408x640"
        _1408X704 = "1408x704"
        _1472X576 = "1472x576"
        _1472X640 = "1472x640"
        _1472X704 = "1472x704"
        _1536X512 = "1536x512"
        _1536X576 = "1536x576"
        _1536X640 = "1536x640"

    class Style_type(str, Enum):
        NONE = "None"
        AUTO = "Auto"
        GENERAL = "General"
        REALISTIC = "Realistic"
        DESIGN = "Design"
        RENDER_3D = "Render 3D"
        ANIME = "Anime"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _16_10 = "16:10"
        _10_16 = "10:16"
        _3_1 = "3:1"
        _1_3 = "1:3"

    class Magic_prompt_option(str, Enum):
        AUTO = "Auto"
        ON = "On"
        OFF = "Off"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "resolution"]

    @classmethod
    def replicate_model_id(cls):
        return "ideogram-ai/ideogram-v2a:8b85e4363b03c25f1d248d0f7e3e118503f2b33773a51bab414603bd52f6112d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/6e45e974-f381-435a-b9dd-23f3e6801c19/replicate-prediction-1yv65m0q.webp",
            "created_at": "2025-02-27T11:03:37.256216Z",
            "description": "Like Ideogram v2, but faster and cheaper",
            "github_url": None,
            "license_url": "https://ideogram.ai/legal/api-tos",
            "name": "ideogram-v2a",
            "owner": "ideogram-ai",
            "is_official": True,
            "paper_url": "https://ideogram.ai/",
            "run_count": 2008249,
            "url": "https://replicate.com/ideogram-ai/ideogram-v2a",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        le=2147483647.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    resolution: Resolution = Field(
        description="Resolution. Overrides aspect ratio. Ignored if an inpainting image is given.",
        default="None",
    )
    style_type: Style_type = Field(
        description="The styles help define the specific aesthetic of the image you want to generate.",
        default="None",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio. Ignored if a resolution or inpainting image is given.",
        default="1:1",
    )
    magic_prompt_option: Magic_prompt_option = Field(
        description="Magic Prompt will interpret your prompt and optimize it to maximize variety and quality of the images generated. You can also use it to write prompts in different languages.",
        default="Auto",
    )


class Imagen_3(ReplicateNode):
    """Google's highest quality text-to-image model, capable of generating images with detail, rich lighting and beauty"""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _9_16 = "9:16"
        _16_9 = "16:9"
        _3_4 = "3:4"
        _4_3 = "4:3"

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    class Safety_filter_level(str, Enum):
        BLOCK_LOW_AND_ABOVE = "block_low_and_above"
        BLOCK_MEDIUM_AND_ABOVE = "block_medium_and_above"
        BLOCK_ONLY_HIGH = "block_only_high"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "output_format"]

    @classmethod
    def replicate_model_id(cls):
        return "google/imagen-3:f01b3b9332c7b6ca2c6193268faff77052d4a13ed024ee18f85ec577e2b0da69"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/6e164365-9cab-422b-bf05-76d127abe3a2/replicate-prediction-_OX51bG7.webp",
            "created_at": "2025-02-05T12:56:07.610594Z",
            "description": "Google's highest quality text-to-image model, capable of generating images with detail, rich lighting and beauty",
            "github_url": None,
            "license_url": None,
            "name": "imagen-3",
            "owner": "google",
            "is_official": True,
            "paper_url": "https://deepmind.google/technologies/imagen-3/",
            "run_count": 1913332,
            "url": "https://replicate.com/google/imagen-3",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="1:1"
    )
    output_format: Output_format = Field(
        description="Format of the output image", default="jpg"
    )
    safety_filter_level: Safety_filter_level = Field(
        description="block_low_and_above is strictest, block_medium_and_above blocks some prompts, block_only_high is most permissive but some prompts will still be blocked",
        default="block_only_high",
    )


class Qwen_Image(ReplicateNode):
    """An image generation foundation model in the Qwen series that achieves significant advances in complex text rendering."""

    class Image_size(str, Enum):
        OPTIMIZE_FOR_QUALITY = "optimize_for_quality"
        OPTIMIZE_FOR_SPEED = "optimize_for_speed"

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "qwen/qwen-image:905e345fe1dfe10d628daac2140dd8dea471c0d99793ef0fdc46a15c688b62fb"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/62e062a1-f4e4-4192-9f0c-56408f092fec/replicate-prediction-97qabd8z.webp",
            "created_at": "2025-08-04T17:23:39.724770Z",
            "description": "An image generation foundation model in the Qwen series that achieves significant advances in complex text rendering.",
            "github_url": "https://github.com/QwenLM/Qwen-Image",
            "license_url": "https://choosealicense.com/licenses/apache-2.0/",
            "name": "qwen-image",
            "owner": "qwen",
            "is_official": True,
            "paper_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf",
            "run_count": 1382477,
            "url": "https://replicate.com/qwen/qwen-image",
            "visibility": "public",
            "weights_url": "https://huggingface.co/Qwen/Qwen-Image",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image for img2img pipeline"
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for generated image", default=None
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with additional optimizations.",
        default=True,
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance for generated image. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
        ge=0.0,
        le=10.0,
        default=3,
    )
    strength: float = Field(
        title="Strength",
        description="Strength for img2img pipeline",
        ge=0.0,
        le=1.0,
        default=0.9,
    )
    image_size: Image_size = Field(
        description="Image size for the generated image", default="optimize_for_quality"
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied.",
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image", default="16:9"
    )
    lora_weights: str | None = Field(
        title="Lora Weights",
        description="Load LoRA weights. Only works with text to image pipeline. Supports arbitrary .safetensors URLs, tar files, and zip files from the Internet (for example, 'https://huggingface.co/Viktor1717/scandinavian-interior-style1/resolve/main/my_first_flux_lora_v1.safetensors', 'https://example.com/lora_weights.tar.gz', or 'https://example.com/lora_weights.zip')",
        default=None,
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    enhance_prompt: bool = Field(
        title="Enhance Prompt",
        description="Enhance the prompt with positive magic.",
        default=False,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative prompt for generated image",
        default=" ",
    )
    replicate_weights: str | None = Field(
        title="Replicate Weights",
        description="Load LoRA weights from Replicate training. Only works with text to image pipeline. Supports arbitrary .safetensors URLs, tar files, and zip files from the Internet.",
        default=None,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps. Recommended range is 28-50, and lower number of steps produce lower quality outputs, faster.",
        ge=1.0,
        le=50.0,
        default=30,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Qwen_Image_Edit(ReplicateNode):
    """Edit images using a prompt. This model extends Qwen-Image’s unique text rendering capabilities to image editing tasks, enabling precise text editing"""

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "qwen/qwen-image-edit:a072e0d160ef0501120a390f602404655e467a6f591f6574f5742df0b67cbba7"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/29bbf095-fce5-4328-8117-9d79a41149c2/replicate-prediction-f28b7ty1.webp",
            "created_at": "2025-08-18T18:09:48.916882Z",
            "description": "Edit images using a prompt. This model extends Qwen-Image’s unique text rendering capabilities to image editing tasks, enabling precise text editing",
            "github_url": None,
            "license_url": "https://github.com/QwenLM/Qwen-Image/blob/main/LICENSE",
            "name": "qwen-image-edit",
            "owner": "qwen",
            "is_official": True,
            "paper_url": "https://arxiv.org/abs/2508.02324",
            "run_count": 1470160,
            "url": "https://replicate.com/qwen/qwen-image-edit",
            "visibility": "public",
            "weights_url": "https://huggingface.co/Qwen/Qwen-Image-Edit",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image to use as reference. Must be jpeg, png, gif, or webp.",
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Text instruction on how to edit the given image.",
        default=None,
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with additional optimizations.",
        default=True,
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Seedream_4(ReplicateNode):
    """Unified text-to-image generation and precise single-sentence editing at up to 4K resolution"""

    class Size(str, Enum):
        _1K = "1K"
        _2K = "2K"
        _4K = "4K"
        CUSTOM = "custom"

    class Aspect_ratio(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        _1_1 = "1:1"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _21_9 = "21:9"

    class Sequential_image_generation(str, Enum):
        DISABLED = "disabled"
        AUTO = "auto"

    @classmethod
    def get_basic_fields(cls):
        return ["size", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "bytedance/seedream-4:cf7d431991436f19d1c8dad83fe463c729c816d7a21056c5105e75c84a0aa7e9"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/53dda182-2998-4fde-b235-e1b2a09b0484/seedream4-sm.jpg",
            "created_at": "2025-09-09T11:23:42.672377Z",
            "description": "Unified text-to-image generation and precise single-sentence editing at up to 4K resolution",
            "github_url": None,
            "license_url": None,
            "name": "seedream-4",
            "owner": "bytedance",
            "is_official": True,
            "paper_url": None,
            "run_count": 22664322,
            "url": "https://replicate.com/bytedance/seedream-4",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    size: Size = Field(
        description="Image resolution: 1K (1024px), 2K (2048px), 4K (4096px), or 'custom' for specific dimensions.",
        default="2K",
    )
    width: int = Field(
        title="Width",
        description="Custom image width (only used when size='custom'). Range: 1024-4096 pixels.",
        ge=1024.0,
        le=4096.0,
        default=2048,
    )
    height: int = Field(
        title="Height",
        description="Custom image height (only used when size='custom'). Range: 1024-4096 pixels.",
        ge=1024.0,
        le=4096.0,
        default=2048,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    max_images: int = Field(
        title="Max Images",
        description="Maximum number of images to generate when sequential_image_generation='auto'. Range: 1-15. Total images (input + generated) cannot exceed 15.",
        ge=1.0,
        le=15.0,
        default=1,
    )
    image_input: list = Field(
        title="Image Input",
        description="Input image(s) for image-to-image generation. List of 1-10 images for single or multi-reference generation.",
        default=[],
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Image aspect ratio. Only used when size is not 'custom'. Use 'match_input_image' to automatically match the input image's aspect ratio.",
        default="match_input_image",
    )
    enhance_prompt: bool = Field(
        title="Enhance Prompt",
        description="Enable prompt enhancement for higher quality results, this will take longer to generate.",
        default=True,
    )
    sequential_image_generation: Sequential_image_generation = Field(
        description="Group image generation mode. 'disabled' generates a single image. 'auto' lets the model decide whether to generate multiple related images (e.g., story scenes, character variations).",
        default="disabled",
    )


class Imagen_4_Fast(ReplicateNode):
    """Use this fast version of Imagen 4 when speed and cost are more important than quality"""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _9_16 = "9:16"
        _16_9 = "16:9"
        _3_4 = "3:4"
        _4_3 = "4:3"

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    class Safety_filter_level(str, Enum):
        BLOCK_LOW_AND_ABOVE = "block_low_and_above"
        BLOCK_MEDIUM_AND_ABOVE = "block_medium_and_above"
        BLOCK_ONLY_HIGH = "block_only_high"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "output_format"]

    @classmethod
    def replicate_model_id(cls):
        return "google/imagen-4-fast:c0704c9b0a3c3f4853fd28d0dc24d4e820ebc7e4e4ecebd6ce55b4bbea0aa423"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/73c5af65-f578-4113-b62c-2a56971cff2f/replicate-prediction-trmpwr78.webp",
            "created_at": "2025-06-12T09:24:39.272587Z",
            "description": "Use this fast version of Imagen 4 when speed and cost are more important than quality",
            "github_url": None,
            "license_url": None,
            "name": "imagen-4-fast",
            "owner": "google",
            "is_official": True,
            "paper_url": None,
            "run_count": 3668254,
            "url": "https://replicate.com/google/imagen-4-fast",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image", default="1:1"
    )
    output_format: Output_format = Field(
        description="Format of the output image", default="jpg"
    )
    safety_filter_level: Safety_filter_level = Field(
        description="block_low_and_above is strictest, block_medium_and_above blocks some prompts, block_only_high is most permissive but some prompts will still be blocked",
        default="block_only_high",
    )


class Ideogram_V3_Turbo(ReplicateNode):
    """Turbo is the fastest and cheapest Ideogram v3. v3 creates images with stunning realism, creative designs, and consistent styles"""

    class Resolution(str, Enum):
        NONE = "None"
        _512X1536 = "512x1536"
        _576X1408 = "576x1408"
        _576X1472 = "576x1472"
        _576X1536 = "576x1536"
        _640X1344 = "640x1344"
        _640X1408 = "640x1408"
        _640X1472 = "640x1472"
        _640X1536 = "640x1536"
        _704X1152 = "704x1152"
        _704X1216 = "704x1216"
        _704X1280 = "704x1280"
        _704X1344 = "704x1344"
        _704X1408 = "704x1408"
        _704X1472 = "704x1472"
        _736X1312 = "736x1312"
        _768X1088 = "768x1088"
        _768X1216 = "768x1216"
        _768X1280 = "768x1280"
        _768X1344 = "768x1344"
        _800X1280 = "800x1280"
        _832X960 = "832x960"
        _832X1024 = "832x1024"
        _832X1088 = "832x1088"
        _832X1152 = "832x1152"
        _832X1216 = "832x1216"
        _832X1248 = "832x1248"
        _864X1152 = "864x1152"
        _896X960 = "896x960"
        _896X1024 = "896x1024"
        _896X1088 = "896x1088"
        _896X1120 = "896x1120"
        _896X1152 = "896x1152"
        _960X832 = "960x832"
        _960X896 = "960x896"
        _960X1024 = "960x1024"
        _960X1088 = "960x1088"
        _1024X832 = "1024x832"
        _1024X896 = "1024x896"
        _1024X960 = "1024x960"
        _1024X1024 = "1024x1024"
        _1088X768 = "1088x768"
        _1088X832 = "1088x832"
        _1088X896 = "1088x896"
        _1088X960 = "1088x960"
        _1120X896 = "1120x896"
        _1152X704 = "1152x704"
        _1152X832 = "1152x832"
        _1152X864 = "1152x864"
        _1152X896 = "1152x896"
        _1216X704 = "1216x704"
        _1216X768 = "1216x768"
        _1216X832 = "1216x832"
        _1248X832 = "1248x832"
        _1280X704 = "1280x704"
        _1280X768 = "1280x768"
        _1280X800 = "1280x800"
        _1312X736 = "1312x736"
        _1344X640 = "1344x640"
        _1344X704 = "1344x704"
        _1344X768 = "1344x768"
        _1408X576 = "1408x576"
        _1408X640 = "1408x640"
        _1408X704 = "1408x704"
        _1472X576 = "1472x576"
        _1472X640 = "1472x640"
        _1472X704 = "1472x704"
        _1536X512 = "1536x512"
        _1536X576 = "1536x576"
        _1536X640 = "1536x640"

    class Style_type(str, Enum):
        NONE = "None"
        AUTO = "Auto"
        GENERAL = "General"
        REALISTIC = "Realistic"
        DESIGN = "Design"

    class Aspect_ratio(str, Enum):
        _1_3 = "1:3"
        _3_1 = "3:1"
        _1_2 = "1:2"
        _2_1 = "2:1"
        _9_16 = "9:16"
        _16_9 = "16:9"
        _10_16 = "10:16"
        _16_10 = "16:10"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _1_1 = "1:1"

    class Style_preset(str, Enum):
        NONE = "None"
        _80S_ILLUSTRATION = "80s Illustration"
        _90S_NOSTALGIA = "90s Nostalgia"
        ABSTRACT_ORGANIC = "Abstract Organic"
        ANALOG_NOSTALGIA = "Analog Nostalgia"
        ART_BRUT = "Art Brut"
        ART_DECO = "Art Deco"
        ART_POSTER = "Art Poster"
        AURA = "Aura"
        AVANT_GARDE = "Avant Garde"
        BAUHAUS = "Bauhaus"
        BLUEPRINT = "Blueprint"
        BLURRY_MOTION = "Blurry Motion"
        BRIGHT_ART = "Bright Art"
        C4D_CARTOON = "C4D Cartoon"
        CHILDREN_S_BOOK = "Children's Book"
        COLLAGE = "Collage"
        COLORING_BOOK_I = "Coloring Book I"
        COLORING_BOOK_II = "Coloring Book II"
        CUBISM = "Cubism"
        DARK_AURA = "Dark Aura"
        DOODLE = "Doodle"
        DOUBLE_EXPOSURE = "Double Exposure"
        DRAMATIC_CINEMA = "Dramatic Cinema"
        EDITORIAL = "Editorial"
        EMOTIONAL_MINIMAL = "Emotional Minimal"
        ETHEREAL_PARTY = "Ethereal Party"
        EXPIRED_FILM = "Expired Film"
        FLAT_ART = "Flat Art"
        FLAT_VECTOR = "Flat Vector"
        FOREST_REVERIE = "Forest Reverie"
        GEO_MINIMALIST = "Geo Minimalist"
        GLASS_PRISM = "Glass Prism"
        GOLDEN_HOUR = "Golden Hour"
        GRAFFITI_I = "Graffiti I"
        GRAFFITI_II = "Graffiti II"
        HALFTONE_PRINT = "Halftone Print"
        HIGH_CONTRAST = "High Contrast"
        HIPPIE_ERA = "Hippie Era"
        ICONIC = "Iconic"
        JAPANDI_FUSION = "Japandi Fusion"
        JAZZY = "Jazzy"
        LONG_EXPOSURE = "Long Exposure"
        MAGAZINE_EDITORIAL = "Magazine Editorial"
        MINIMAL_ILLUSTRATION = "Minimal Illustration"
        MIXED_MEDIA = "Mixed Media"
        MONOCHROME = "Monochrome"
        NIGHTLIFE = "Nightlife"
        OIL_PAINTING = "Oil Painting"
        OLD_CARTOONS = "Old Cartoons"
        PAINT_GESTURE = "Paint Gesture"
        POP_ART = "Pop Art"
        RETRO_ETCHING = "Retro Etching"
        RIVIERA_POP = "Riviera Pop"
        SPOTLIGHT_80S = "Spotlight 80s"
        STYLIZED_RED = "Stylized Red"
        SURREAL_COLLAGE = "Surreal Collage"
        TRAVEL_POSTER = "Travel Poster"
        VINTAGE_GEO = "Vintage Geo"
        VINTAGE_POSTER = "Vintage Poster"
        WATERCOLOR = "Watercolor"
        WEIRD = "Weird"
        WOODBLOCK_PRINT = "Woodblock Print"

    class Magic_prompt_option(str, Enum):
        AUTO = "Auto"
        ON = "On"
        OFF = "Off"

    @classmethod
    def get_basic_fields(cls):
        return ["mask", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "ideogram-ai/ideogram-v3-turbo:d9b3748f95c0fe3e71f010f8cc5d80e8f5252acd0e74b1c294ee889eea52a47b"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/b55e9f9e-5f43-4cf8-99c8-c33cc8486f23/tmp0s1h52uw-1.webp",
            "created_at": "2025-04-30T13:21:08.936269Z",
            "description": "Turbo is the fastest and cheapest Ideogram v3. v3 creates images with stunning realism, creative designs, and consistent styles",
            "github_url": None,
            "license_url": "https://about.ideogram.ai/legal/api-tos",
            "name": "ideogram-v3-turbo",
            "owner": "ideogram-ai",
            "is_official": True,
            "paper_url": "https://about.ideogram.ai/3.0",
            "run_count": 6326689,
            "url": "https://replicate.com/ideogram-ai/ideogram-v3-turbo",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    mask: types.ImageRef = Field(
        default=types.ImageRef(),
        description="A black and white image. Black pixels are inpainted, white pixels are preserved. The mask will be resized to match the image size.",
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        le=2147483647.0,
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="An image file to use for inpainting. You must also use a mask.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    resolution: Resolution = Field(
        description="Resolution. Overrides aspect ratio. Ignored if an inpainting image is given.",
        default="None",
    )
    style_type: Style_type = Field(
        description="The styles help define the specific aesthetic of the image you want to generate.",
        default="None",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio. Ignored if a resolution or inpainting image is given.",
        default="1:1",
    )
    style_preset: Style_preset = Field(
        description="Apply a predefined artistic style to the generated image (V3 models only).",
        default="None",
    )
    magic_prompt_option: Magic_prompt_option = Field(
        description="Magic Prompt will interpret your prompt and optimize it to maximize variety and quality of the images generated. You can also use it to write prompts in different languages.",
        default="Auto",
    )
    style_reference_images: list | None = Field(
        title="Style Reference Images",
        description="A list of images to use as style references.",
        default=None,
    )


class Flux_Kontext_Pro(ReplicateNode):
    """A state-of-the-art text-based image editing model that delivers high-quality outputs with excellent prompt following and consistent results for transforming images through natural language"""

    class Aspect_ratio(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        _1_1 = "1:1"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _21_9 = "21:9"
        _9_21 = "9:21"
        _2_1 = "2:1"
        _1_2 = "1:2"

    class Output_format(str, Enum):
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "input_image"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-kontext-pro:897a70f5a7dbd8a0611413b3b98cf417b45f266bd595c571a22947619d9ae462"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/e74eecd6-daf1-4050-9f04-36313bd6f007/two-people-cropped.webp",
            "created_at": "2025-05-27T08:26:25.135215Z",
            "description": "A state-of-the-art text-based image editing model that delivers high-quality outputs with excellent prompt following and consistent results for transforming images through natural language",
            "github_url": None,
            "license_url": None,
            "name": "flux-kontext-pro",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 43170789,
            "url": "https://replicate.com/black-forest-labs/flux-kontext-pro",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Text description of what you want to generate, or the instruction on how to edit the given image.",
        default=None,
    )
    input_image: str | None = Field(
        title="Input Image",
        description="Image to use as reference. Must be jpeg, png, gif, or webp.",
        default=None,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the generated image. Use 'match_input_image' to match the aspect ratio of the input image.",
        default="match_input_image",
    )
    output_format: Output_format = Field(
        description="Output format for the generated image", default="png"
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 0 is most strict and 6 is most permissive. 2 is currently the maximum allowed when input images are used.",
        ge=0.0,
        le=6.0,
        default=2,
    )
    prompt_upsampling: bool = Field(
        title="Prompt Upsampling",
        description="Automatic prompt improvement",
        default=False,
    )


class Minimax_Image_01(ReplicateNode):
    """Minimax's first image model, with character reference support"""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _4_3 = "4:3"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _3_4 = "3:4"
        _9_16 = "9:16"
        _21_9 = "21:9"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "number_of_images"]

    @classmethod
    def replicate_model_id(cls):
        return "minimax/image-01:928f3bd6ac899108d0ab8cf7f91dfa39a03eda0175e94c9b4cd075776dececf0"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/926994db-2c8e-4b7d-934f-2f86b2480e55/43b05178-4b2a-42d9-9130-4fedae65.webp",
            "created_at": "2025-03-03T14:05:29.816962Z",
            "description": "Minimax's first image model, with character reference support",
            "github_url": None,
            "license_url": None,
            "name": "image-01",
            "owner": "minimax",
            "is_official": True,
            "paper_url": None,
            "run_count": 2455433,
            "url": "https://replicate.com/minimax/image-01",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    prompt: str | None = Field(
        title="Prompt", description="Text prompt for generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(description="Image aspect ratio", default="1:1")
    number_of_images: int = Field(
        title="Number Of Images",
        description="Number of images to generate",
        ge=1.0,
        le=9.0,
        default=1,
    )
    prompt_optimizer: bool = Field(
        title="Prompt Optimizer", description="Use prompt optimizer", default=True
    )
    subject_reference: str | None = Field(
        title="Subject Reference",
        description="An optional character reference image (human face) to use as the subject in the generated image(s).",
        default=None,
    )


class Flux_2_Pro(ReplicateNode):
    """High-quality image generation and editing with support for eight reference images"""

    class Resolution(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        _0_5_MP = "0.5 MP"
        _1_MP = "1 MP"
        _2_MP = "2 MP"
        _4_MP = "4 MP"

    class Aspect_ratio(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        CUSTOM = "custom"
        _1_1 = "1:1"
        _16_9 = "16:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _3_4 = "3:4"
        _4_3 = "4:3"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-2-pro:285631b5656a1839331cd9af0d82da820e2075db12046d1d061c681b2f206bc6"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/5a8b527e-f298-45db-b51a-4e1cc7d5f1eb/flux-2-pro-sm.jpg",
            "created_at": "2025-11-14T22:48:19.258717Z",
            "description": "High-quality image generation and editing with support for eight reference images",
            "github_url": None,
            "license_url": None,
            "name": "flux-2-pro",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 1454890,
            "url": "https://replicate.com/black-forest-labs/flux-2-pro",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    width: int | None = Field(
        title="Width",
        description="Width of the generated image. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32).",
        ge=256.0,
        le=2048.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of the generated image. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32).",
        ge=256.0,
        le=2048.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    resolution: Resolution = Field(
        description="Resolution in megapixels. Up to 4 MP is possible, but 2 MP or below is recommended. The maximum image size is 2048x2048, which means that high-resolution images may not respect the resolution if aspect ratio is not 1:1.\n\nResolution is not used when aspect_ratio is 'custom'. When aspect_ratio is 'match_input_image', use 'match_input_image' to match the input image's resolution (clamped to 0.5-4 MP).",
        default="1 MP",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. Use 'match_input_image' to match the first input image's aspect ratio.",
        default="1:1",
    )
    input_images: list = Field(
        title="Input Images",
        description="List of input images for image-to-image generation. Maximum 8 images. Must be jpeg, png, gif, or webp.",
        default=[],
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 5 is most permissive",
        ge=1.0,
        le=5.0,
        default=2,
    )


class Flux_2_Flex(ReplicateNode):
    """Max-quality image generation and editing with support for ten reference images"""

    class Resolution(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        _0_5_MP = "0.5 MP"
        _1_MP = "1 MP"
        _2_MP = "2 MP"
        _4_MP = "4 MP"

    class Aspect_ratio(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        CUSTOM = "custom"
        _1_1 = "1:1"
        _16_9 = "16:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _3_4 = "3:4"
        _4_3 = "4:3"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "steps", "width"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-2-flex:57df51f07c4bc4457b768277b1bb754a0c35b9f02c2ce48582a8b3a48fe0a2c3"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/DxBvdhGcpX6FLxoUikZhQdbYs87OMEv83563fspgBuXS7a2KA/tmp249z689p.webp",
            "created_at": "2025-11-25T10:38:39.577145Z",
            "description": "Max-quality image generation and editing with support for ten reference images",
            "github_url": None,
            "license_url": None,
            "name": "flux-2-flex",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 84403,
            "url": "https://replicate.com/black-forest-labs/flux-2-flex",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    steps: int = Field(
        title="Steps",
        description="Number of inference steps",
        ge=1.0,
        le=50.0,
        default=30,
    )
    width: int | None = Field(
        title="Width",
        description="Width of the generated image. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32).",
        ge=256.0,
        le=2048.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of the generated image. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32).",
        ge=256.0,
        le=2048.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    guidance: float = Field(
        title="Guidance",
        description="Guidance scale for generation. Controls how closely the output follows the prompt",
        ge=1.5,
        le=10.0,
        default=4.5,
    )
    resolution: Resolution = Field(
        description="Resolution in megapixels. Up to 4 MP is possible, but 2 MP or below is recommended. The maximum image size is 2048x2048, which means that high-resolution images may not respect the resolution if aspect ratio is not 1:1.\n\nResolution is not used when aspect_ratio is 'custom'. When aspect_ratio is 'match_input_image', use 'match_input_image' to match the input image's resolution (clamped to 0.5-4 MP).",
        default="1 MP",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. Use 'match_input_image' to match the first input image's aspect ratio.",
        default="1:1",
    )
    input_images: list = Field(
        title="Input Images",
        description="List of input images for image-to-image generation. Maximum 10 images. Must be jpeg, png, gif, or webp.",
        default=[],
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 5 is most permissive",
        ge=1.0,
        le=5.0,
        default=2,
    )
    prompt_upsampling: bool = Field(
        title="Prompt Upsampling",
        description="Automatically modify the prompt for more creative generation",
        default=True,
    )


class Flux_2_Klein_4B(ReplicateNode):
    """Very fast image generation and editing model. 4 steps distilled, sub-second inference for production and near real-time applications."""

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _16_9 = "16:9"
        _9_16 = "9:16"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _5_4 = "5:4"
        _4_5 = "4:5"
        _21_9 = "21:9"
        _9_21 = "9:21"
        MATCH_INPUT_IMAGE = "match_input_image"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    class Output_megapixels(str, Enum):
        _0_25 = "0.25"
        _0_5 = "0.5"
        _1 = "1"
        _2 = "2"
        _4 = "4"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "images", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-2-klein-4b:8e9c42d77b10a2a41af823ac4500f7545be6ebc4e745830fc3f3de10de200542"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/e32c60b0-a5e4-4846-b5e9-752df87ed6c5/replicate-klein-cover.jpg",
            "created_at": "2026-01-15T11:21:33.057246Z",
            "description": "Very fast image generation and editing model. 4 steps distilled, sub-second inference for production and near real-time applications.",
            "github_url": "https://github.com/black-forest-labs/flux2",
            "license_url": "https://github.com/black-forest-labs/flux2/blob/main/LICENSE.md",
            "name": "flux-2-klein-4b",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 7604,
            "url": "https://replicate.com/black-forest-labs/flux-2-klein-4b",
            "visibility": "public",
            "weights_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    images: list = Field(
        title="Images",
        description="List of input images for image-to-image generation. Maximum 5 images. Must be jpeg, png, gif, or webp.",
        default=[],
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation.", default=None
    )
    go_fast: bool = Field(
        title="Go Fast",
        description="Run faster predictions with additional optimizations.",
        default=False,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. Use 'match_input_image' to match the aspect ratio of the first input image.",
        default="1:1",
    )
    output_format: Output_format = Field(
        description="Format of the output images", default="jpg"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs.",
        ge=0.0,
        le=100.0,
        default=95,
    )
    output_megapixels: Output_megapixels = Field(
        description="Resolution of the output image in megapixels", default="1"
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class Flux_2_Max(ReplicateNode):
    """The highest fidelity image model from Black Forest Labs"""

    class Resolution(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        _0_5_MP = "0.5 MP"
        _1_MP = "1 MP"
        _2_MP = "2 MP"
        _4_MP = "4 MP"

    class Aspect_ratio(str, Enum):
        MATCH_INPUT_IMAGE = "match_input_image"
        CUSTOM = "custom"
        _1_1 = "1:1"
        _16_9 = "16:9"
        _3_2 = "3:2"
        _2_3 = "2:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _3_4 = "3:4"
        _4_3 = "4:3"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "black-forest-labs/flux-2-max:c9a020854ba37d5fe801ab712570d7e437b17c148843fe96dbcb7cadd160a8f7"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8147cff6-abe5-4842-9f6d-24fe06c626b9/flux-2-max-cover.jpg",
            "created_at": "2025-12-16T14:37:45.724046Z",
            "description": "The highest fidelity image model from Black Forest Labs",
            "github_url": None,
            "license_url": None,
            "name": "flux-2-max",
            "owner": "black-forest-labs",
            "is_official": True,
            "paper_url": None,
            "run_count": 194166,
            "url": "https://replicate.com/black-forest-labs/flux-2-max",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    width: int | None = Field(
        title="Width",
        description="Width of the generated image. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32).",
        ge=256.0,
        le=2048.0,
        default=None,
    )
    height: int | None = Field(
        title="Height",
        description="Height of the generated image. Only used when aspect_ratio=custom. Must be a multiple of 32 (if it's not, it will be rounded to nearest multiple of 32).",
        ge=256.0,
        le=2048.0,
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    resolution: Resolution = Field(
        description="Resolution in megapixels. Up to 4 MP is possible, but 2 MP or below is recommended. The maximum image size is 2048x2048, which means that high-resolution images may not respect the resolution if aspect ratio is not 1:1.\n\nResolution is not used when aspect_ratio is 'custom'. When aspect_ratio is 'match_input_image', use 'match_input_image' to match the input image's resolution (clamped to 0.5-4 MP).",
        default="1 MP",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio for the generated image. Use 'match_input_image' to match the first input image's aspect ratio.",
        default="1:1",
    )
    input_images: list = Field(
        title="Input Images",
        description="List of input images for image-to-image generation. Maximum 8 images. Must be jpeg, png, gif, or webp.",
        default=[],
    )
    output_format: Output_format = Field(
        description="Format of the output images.", default="webp"
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        ge=0.0,
        le=100.0,
        default=80,
    )
    safety_tolerance: int = Field(
        title="Safety Tolerance",
        description="Safety tolerance, 1 is most strict and 5 is most permissive",
        ge=1.0,
        le=5.0,
        default=2,
    )
