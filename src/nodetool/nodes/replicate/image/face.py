from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class FaceToMany(ReplicateNode):
    """Turn a face into 3D, emoji, pixel art, video game, claymation or toy"""

    class Style(str, Enum):
        _3D = "3D"
        EMOJI = "Emoji"
        VIDEO_GAME = "Video game"
        PIXELS = "Pixels"
        CLAY = "Clay"
        TOY = "Toy"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "style"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/face-to-many:a07f252abbbd832009640b27f063ea52d87d7a23a185ca165bec23b5adc8deaf"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/583bfb50-534d-4835-a856-80bd2abb332e/mona-list-emoji.webp",
            "created_at": "2024-03-05T13:01:03.163557Z",
            "description": "Turn a face into 3D, emoji, pixel art, video game, claymation or toy",
            "github_url": "https://github.com/fofr/cog-face-to-many",
            "license_url": "https://github.com/fofr/cog-face-to-many/blob/main/weights_licenses.md",
            "name": "face-to-many",
            "owner": "fofr",
            "paper_url": None,
            "run_count": 12626295,
            "url": "https://replicate.com/fofr/face-to-many",
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
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="An image of a person to be converted"
    )
    style: Style = Field(description="Style to convert to", default=Style("3D"))
    prompt: str = Field(title="Prompt", default="a person")
    lora_scale: float = Field(
        title="Lora Scale",
        description="How strong the LoRA will be",
        ge=0.0,
        le=1.0,
        default=1,
    )
    custom_lora_url: str | None = Field(
        title="Custom Lora Url",
        description="URL to a Replicate custom LoRA. Must be in the format https://replicate.delivery/pbxt/[id]/trained_model.tar or https://pbxt.replicate.delivery/[id]/trained_model.tar",
        default=None,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want in the image",
        default="",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Strength of the prompt. This is the CFG scale, higher numbers lead to stronger prompt, lower numbers will keep more of a likeness to the original.",
        ge=0.0,
        le=20.0,
        default=4.5,
    )
    denoising_strength: float = Field(
        title="Denoising Strength",
        description="How much of the original image to keep. 1 is the complete destruction of the original image, 0 is the original image",
        ge=0.0,
        le=1.0,
        default=0.65,
    )
    instant_id_strength: float = Field(
        title="Instant Id Strength",
        description="How strong the InstantID will be.",
        ge=0.0,
        le=1.0,
        default=1,
    )
    control_depth_strength: float = Field(
        title="Control Depth Strength",
        description="Strength of depth controlnet. The bigger this is, the more controlnet affects the output.",
        ge=0.0,
        le=1.0,
        default=0.8,
    )


class BecomeImage(ReplicateNode):
    """Adapt any picture of a face into another image"""

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/become-image:8d0b076a2aff3904dfcec3253c778e0310a68f78483c4699c7fd800f3051d2b3"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/b37fc7b7-0cef-4895-9176-bf5bb0cb7011/pearl-earring-1.webp",
            "created_at": "2024-03-11T11:16:22.168373Z",
            "description": "Adapt any picture of a face into another image",
            "github_url": "https://github.com/fofr/cog-become-image",
            "license_url": "https://github.com/fofr/cog-become-image/blob/main/weights_licenses.md",
            "name": "become-image",
            "owner": "fofr",
            "paper_url": None,
            "run_count": 474431,
            "url": "https://replicate.com/fofr/become-image",
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
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="An image of a person to be converted"
    )
    prompt: str = Field(title="Prompt", default="a person")
    image_to_become: types.ImageRef = Field(
        default=types.ImageRef(), description="Any image to convert the person to"
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want in the image",
        default="",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Strength of the prompt. This is the CFG scale, higher numbers lead to stronger prompt, lower numbers will keep more of a likeness to the original.",
        ge=0.0,
        le=3.0,
        default=2,
    )
    number_of_images: int = Field(
        title="Number Of Images",
        description="Number of images to generate",
        ge=1.0,
        le=10.0,
        default=2,
    )
    denoising_strength: float = Field(
        title="Denoising Strength",
        description="How much of the original image of the person to keep. 1 is the complete destruction of the original image, 0 is the original image",
        ge=0.0,
        le=1.0,
        default=1,
    )
    instant_id_strength: float = Field(
        title="Instant Id Strength",
        description="How strong the InstantID will be.",
        ge=0.0,
        le=1.0,
        default=1,
    )
    image_to_become_noise: float = Field(
        title="Image To Become Noise",
        description="How much noise to add to the style image before processing. An alternative way of controlling stength.",
        ge=0.0,
        le=1.0,
        default=0.3,
    )
    control_depth_strength: float = Field(
        title="Control Depth Strength",
        description="Strength of depth controlnet. The bigger this is, the more controlnet affects the output.",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images",
        default=False,
    )
    image_to_become_strength: float = Field(
        title="Image To Become Strength",
        description="How strong the style will be applied",
        ge=0.0,
        le=1.0,
        default=0.75,
    )


class PhotoMaker(ReplicateNode):
    """Create photos, paintings and avatars for anyone in any style within seconds."""

    class Style_name(str, Enum):
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        DISNEY_CHARACTOR = "Disney Charactor"
        DIGITAL_ART = "Digital Art"
        PHOTOGRAPHIC__DEFAULT = "Photographic (Default)"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        ENHANCE = "Enhance"
        COMIC_BOOK = "Comic book"
        LOWPOLY = "Lowpoly"
        LINE_ART = "Line art"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "num_steps"]

    @classmethod
    def replicate_model_id(cls):
        return "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/c10ac3c0-f86a-4249-9f3d-d7723cc93d45/photomaker.jpg",
            "created_at": "2024-01-16T15:42:17.882162Z",
            "description": "Create photos, paintings and avatars for anyone in any style within seconds.",
            "github_url": "https://github.com/datakami-models/PhotoMaker",
            "license_url": "https://github.com/TencentARC/PhotoMaker/blob/main/LICENSE",
            "name": "photomaker",
            "owner": "tencentarc",
            "paper_url": "https://huggingface.co/papers/2312.04461",
            "run_count": 5769183,
            "url": "https://replicate.com/tencentarc/photomaker",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Seed. Leave blank to use a random number",
        ge=0.0,
        le=2147483647.0,
        default=None,
    )
    prompt: str = Field(
        title="Prompt",
        description="Prompt. Example: 'a photo of a man/woman img'. The phrase 'img' is the trigger word.",
        default="A photo of a person img",
    )
    num_steps: int = Field(
        title="Num Steps",
        description="Number of sample steps",
        ge=1.0,
        le=100.0,
        default=20,
    )
    style_name: Style_name = Field(
        description="Style template. The style template will add a style-specific prompt and negative prompt to the user's prompt.",
        default=Style_name("Photographic (Default)"),
    )
    input_image: str | None = Field(
        title="Input Image",
        description="The input image, for example a photo of your face.",
        default=None,
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of output images",
        ge=1.0,
        le=4.0,
        default=1,
    )
    input_image2: str | None = Field(
        title="Input Image2",
        description="Additional input image (optional)",
        default=None,
    )
    input_image3: str | None = Field(
        title="Input Image3",
        description="Additional input image (optional)",
        default=None,
    )
    input_image4: str | None = Field(
        title="Input Image4",
        description="Additional input image (optional)",
        default=None,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale. A guidance scale of 1 corresponds to doing no classifier free guidance.",
        ge=1.0,
        le=10.0,
        default=5,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative Prompt. The negative prompt should NOT contain the trigger word.",
        default="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    )
    style_strength_ratio: float = Field(
        title="Style Strength Ratio",
        description="Style strength (%)",
        ge=15.0,
        le=50.0,
        default=20,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class PhotoMakerStyle(ReplicateNode):
    """Create photos, paintings and avatars for anyone in any style within seconds.  (Stylization version)"""

    class Style_name(str, Enum):
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        DISNEY_CHARACTOR = "Disney Charactor"
        DIGITAL_ART = "Digital Art"
        PHOTOGRAPHIC__DEFAULT = "Photographic (Default)"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        ENHANCE = "Enhance"
        COMIC_BOOK = "Comic book"
        LOWPOLY = "Lowpoly"
        LINE_ART = "Line art"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "num_steps"]

    @classmethod
    def replicate_model_id(cls):
        return "tencentarc/photomaker-style:467d062309da518648ba89d226490e02b8ed09b5abc15026e54e31c5a8cd0769"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/8e85a287-826f-4c21-9079-22eac106dd6b/output.0.png",
            "created_at": "2024-01-18T14:28:51.763369Z",
            "description": "Create photos, paintings and avatars for anyone in any style within seconds.  (Stylization version)",
            "github_url": "https://github.com/TencentARC/PhotoMaker",
            "license_url": "https://github.com/TencentARC/PhotoMaker/blob/main/LICENSE",
            "name": "photomaker-style",
            "owner": "tencentarc",
            "paper_url": "https://huggingface.co/papers/2312.04461",
            "run_count": 1181253,
            "url": "https://replicate.com/tencentarc/photomaker-style",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    seed: int | None = Field(
        title="Seed",
        description="Seed. Leave blank to use a random number",
        ge=0.0,
        le=2147483647.0,
        default=None,
    )
    prompt: str = Field(
        title="Prompt",
        description="Prompt. Example: 'a photo of a man/woman img'. The phrase 'img' is the trigger word.",
        default="A photo of a person img",
    )
    num_steps: int = Field(
        title="Num Steps",
        description="Number of sample steps",
        ge=1.0,
        le=100.0,
        default=20,
    )
    style_name: Style_name = Field(
        description="Style template. The style template will add a style-specific prompt and negative prompt to the user's prompt.",
        default=Style_name("(No style)"),
    )
    input_image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="The input image, for example a photo of your face.",
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of output images",
        ge=1.0,
        le=4.0,
        default=1,
    )
    input_image2: types.ImageRef = Field(
        default=types.ImageRef(), description="Additional input image (optional)"
    )
    input_image3: types.ImageRef = Field(
        default=types.ImageRef(), description="Additional input image (optional)"
    )
    input_image4: types.ImageRef = Field(
        default=types.ImageRef(), description="Additional input image (optional)"
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale. A guidance scale of 1 corresponds to doing no classifier free guidance.",
        ge=1.0,
        le=10.0,
        default=5,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative Prompt. The negative prompt should NOT contain the trigger word.",
        default="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    )
    style_strength_ratio: float = Field(
        title="Style Strength Ratio",
        description="Style strength (%)",
        ge=15.0,
        le=50.0,
        default=20,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images.",
        default=False,
    )


class FaceToSticker(ReplicateNode):
    """Turn a face into a sticker"""

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "steps"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/face-to-sticker:764d4827ea159608a07cdde8ddf1c6000019627515eb02b6b449695fd547e5ef"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/vvHSs2p5vPJtPRNW7ffCYt54fwP43r8I3kG4LPfoCCMaMBWPB/ComfyUI_00002_.png",
            "created_at": "2024-02-28T15:14:15.687345Z",
            "description": "Turn a face into a sticker",
            "github_url": "https://github.com/fofr/cog-face-to-sticker",
            "license_url": "https://github.com/fofr/cog-face-to-sticker/blob/main/weights_licenses.md",
            "name": "face-to-sticker",
            "owner": "fofr",
            "paper_url": None,
            "run_count": 1370085,
            "url": "https://replicate.com/fofr/face-to-sticker",
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
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="An image of a person to be converted to a sticker",
    )
    steps: int = Field(title="Steps", default=20)
    width: int = Field(title="Width", default=1024)
    height: int = Field(title="Height", default=1024)
    prompt: str = Field(title="Prompt", default="a person")
    upscale: bool = Field(
        title="Upscale", description="2x upscale the sticker", default=False
    )
    upscale_steps: int = Field(
        title="Upscale Steps", description="Number of steps to upscale", default=10
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want in the image",
        default="",
    )
    prompt_strength: float = Field(
        title="Prompt Strength",
        description="Strength of the prompt. This is the CFG scale, higher numbers lead to stronger prompt, lower numbers will keep more of a likeness to the original.",
        default=7,
    )
    ip_adapter_noise: float = Field(
        title="Ip Adapter Noise",
        description="How much noise is added to the IP adapter input",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    ip_adapter_weight: float = Field(
        title="Ip Adapter Weight",
        description="How much the IP adapter will influence the image",
        ge=0.0,
        le=1.0,
        default=0.2,
    )
    instant_id_strength: float = Field(
        title="Instant Id Strength",
        description="How strong the InstantID will be.",
        ge=0.0,
        le=1.0,
        default=1,
    )


class InstantId(ReplicateNode):
    """Make realistic images of real people instantly"""

    class Scheduler(str, Enum):
        DEISMULTISTEPSCHEDULER = "DEISMultistepScheduler"
        HEUNDISCRETESCHEDULER = "HeunDiscreteScheduler"
        EULERDISCRETESCHEDULER = "EulerDiscreteScheduler"
        DPMSOLVERMULTISTEPSCHEDULER = "DPMSolverMultistepScheduler"
        DPMSOLVERMULTISTEPSCHEDULER_KARRAS = "DPMSolverMultistepScheduler-Karras"
        DPMSOLVERMULTISTEPSCHEDULER_KARRAS_SDE = (
            "DPMSolverMultistepScheduler-Karras-SDE"
        )

    class Sdxl_weights(str, Enum):
        STABLE_DIFFUSION_XL_BASE_1_0 = "stable-diffusion-xl-base-1.0"
        JUGGERNAUT_XL_V8 = "juggernaut-xl-v8"
        AFRODITE_XL_V2 = "afrodite-xl-v2"
        ALBEDOBASE_XL_20 = "albedobase-xl-20"
        ALBEDOBASE_XL_V13 = "albedobase-xl-v13"
        ANIMAGINE_XL_30 = "animagine-xl-30"
        ANIME_ART_DIFFUSION_XL = "anime-art-diffusion-xl"
        ANIME_ILLUST_DIFFUSION_XL = "anime-illust-diffusion-xl"
        DREAMSHAPER_XL = "dreamshaper-xl"
        DYNAVISION_XL_V0610 = "dynavision-xl-v0610"
        GUOFENG4_XL = "guofeng4-xl"
        NIGHTVISION_XL_0791 = "nightvision-xl-0791"
        OMNIGEN_XL = "omnigen-xl"
        PONY_DIFFUSION_V6_XL = "pony-diffusion-v6-xl"
        PROTOVISION_XL_HIGH_FIDEL = "protovision-xl-high-fidel"
        REALVISXL_V3_0_TURBO = "RealVisXL_V3.0_Turbo"
        REALVISXL_V4_0_LIGHTNING = "RealVisXL_V4.0_Lightning"

    class Output_format(str, Enum):
        WEBP = "webp"
        JPG = "jpg"
        PNG = "png"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/3bb0b275-5996-4382-b73f-5bccfbddde92/instantidcover.jpg",
            "created_at": "2024-01-22T21:00:49.120905Z",
            "description": "Make realistic images of real people instantly",
            "github_url": "https://github.com/zsxkib/InstantID",
            "license_url": "https://github.com/zsxkib/InstantID/blob/main/LICENSE",
            "name": "instant-id",
            "owner": "zsxkib",
            "paper_url": "https://arxiv.org/abs/2401.07519",
            "run_count": 805604,
            "url": "https://replicate.com/zsxkib/instant-id",
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
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input face image"
    )
    prompt: str = Field(title="Prompt", description="Input prompt", default="a person")
    scheduler: Scheduler = Field(
        description="Scheduler", default=Scheduler("EulerDiscreteScheduler")
    )
    enable_lcm: bool = Field(
        title="Enable Lcm",
        description="Enable Fast Inference with LCM (Latent Consistency Models) - speeds up inference steps, trade-off is the quality of the generated image. Performs better with close-up portrait face images",
        default=False,
    )
    pose_image: types.ImageRef = Field(
        default=types.ImageRef(), description="(Optional) reference pose image"
    )
    num_outputs: int = Field(
        title="Num Outputs",
        description="Number of images to output",
        ge=1.0,
        le=8.0,
        default=1,
    )
    sdxl_weights: Sdxl_weights = Field(
        description="Pick which base weights you want to use",
        default=Sdxl_weights("stable-diffusion-xl-base-1.0"),
    )
    output_format: Output_format = Field(
        description="Format of the output images", default=Output_format("webp")
    )
    pose_strength: float = Field(
        title="Pose Strength",
        description="Openpose ControlNet strength, effective only if `enable_pose_controlnet` is true",
        ge=0.0,
        le=1.0,
        default=0.4,
    )
    canny_strength: float = Field(
        title="Canny Strength",
        description="Canny ControlNet strength, effective only if `enable_canny_controlnet` is true",
        ge=0.0,
        le=1.0,
        default=0.3,
    )
    depth_strength: float = Field(
        title="Depth Strength",
        description="Depth ControlNet strength, effective only if `enable_depth_controlnet` is true",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=7.5,
    )
    output_quality: int = Field(
        title="Output Quality",
        description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
        ge=0.0,
        le=100.0,
        default=80,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    ip_adapter_scale: float = Field(
        title="Ip Adapter Scale",
        description="Scale for image adapter strength (for detail)",
        ge=0.0,
        le=1.5,
        default=0.8,
    )
    lcm_guidance_scale: float = Field(
        title="Lcm Guidance Scale",
        description="Only used when `enable_lcm` is set to True, Scale for classifier-free guidance when using LCM",
        ge=1.0,
        le=20.0,
        default=1.5,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=30,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated images",
        default=False,
    )
    enable_pose_controlnet: bool = Field(
        title="Enable Pose Controlnet",
        description="Enable Openpose ControlNet, overrides strength if set to false",
        default=True,
    )
    enhance_nonface_region: bool = Field(
        title="Enhance Nonface Region",
        description="Enhance non-face region",
        default=True,
    )
    enable_canny_controlnet: bool = Field(
        title="Enable Canny Controlnet",
        description="Enable Canny ControlNet, overrides strength if set to false",
        default=False,
    )
    enable_depth_controlnet: bool = Field(
        title="Enable Depth Controlnet",
        description="Enable Depth ControlNet, overrides strength if set to false",
        default=False,
    )
    lcm_num_inference_steps: int = Field(
        title="Lcm Num Inference Steps",
        description="Only used when `enable_lcm` is set to True, Number of denoising steps when using LCM",
        ge=1.0,
        le=10.0,
        default=5,
    )
    face_detection_input_width: int = Field(
        title="Face Detection Input Width",
        description="Width of the input image for face detection",
        ge=640.0,
        le=4096.0,
        default=640,
    )
    face_detection_input_height: int = Field(
        title="Face Detection Input Height",
        description="Height of the input image for face detection",
        ge=640.0,
        le=4096.0,
        default=640,
    )
    controlnet_conditioning_scale: float = Field(
        title="Controlnet Conditioning Scale",
        description="Scale for IdentityNet strength (for fidelity)",
        ge=0.0,
        le=1.5,
        default=0.8,
    )


class Instant_ID_Photorealistic(ReplicateNode):
    """InstantID : Zero-shot Identity-Preserving Generation in Seconds. Using Juggernaut-XL v8 as the base model to encourage photorealism"""

    @classmethod
    def get_basic_fields(cls):
        return ["image", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "grandlineai/instant-id-photorealistic:03914a0c3326bf44383d0cd84b06822618af879229ce5d1d53bef38d93b68279"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/0c6bd74f-6129-4323-8125-beb65871a8de/Screenshot_2024-01-24_at_15.24.13.png",
            "created_at": "2024-01-24T07:43:55.954510Z",
            "description": "InstantID : Zero-shot Identity-Preserving Generation in Seconds. Using Juggernaut-XL v8 as the base model to encourage photorealism",
            "github_url": "https://github.com/GrandlineAI/InstantID",
            "license_url": "https://github.com/InstantID/InstantID/blob/main/LICENSE",
            "name": "instant-id-photorealistic",
            "owner": "grandlineai",
            "paper_url": None,
            "run_count": 39827,
            "url": "https://replicate.com/grandlineai/instant-id-photorealistic",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    image: types.ImageRef = Field(default=types.ImageRef(), description="Input image")
    width: int = Field(
        title="Width",
        description="Width of output image",
        ge=512.0,
        le=2048.0,
        default=640,
    )
    height: int = Field(
        title="Height",
        description="Height of output image",
        ge=512.0,
        le=2048.0,
        default=640,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality",
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=5,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    ip_adapter_scale: float = Field(
        title="Ip Adapter Scale",
        description="Scale for IP adapter",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=30,
    )
    controlnet_conditioning_scale: float = Field(
        title="Controlnet Conditioning Scale",
        description="Scale for ControlNet conditioning",
        ge=0.0,
        le=1.0,
        default=0.8,
    )


class Instant_ID_Artistic(ReplicateNode):
    """InstantID : Zero-shot Identity-Preserving Generation in Seconds. Using Dreamshaper-XL as the base model to encourage artistic generations"""

    @classmethod
    def get_basic_fields(cls):
        return ["image", "width", "height"]

    @classmethod
    def replicate_model_id(cls):
        return "grandlineai/instant-id-artistic:9cad10c7870bac9d6b587f406aef28208f964454abff5c4152f7dec9b0212a9a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/87b24249-b2c1-43f3-b3cf-a5005f23b21c/Screenshot_2024-01-24_at_15.24.13.png",
            "created_at": "2024-01-24T04:34:52.345779Z",
            "description": "InstantID : Zero-shot Identity-Preserving Generation in Seconds. Using Dreamshaper-XL as the base model to encourage artistic generations",
            "github_url": "https://github.com/GrandlineAI/InstantID",
            "license_url": "https://github.com/InstantID/InstantID/blob/main/LICENSE",
            "name": "instant-id-artistic",
            "owner": "grandlineai",
            "paper_url": None,
            "run_count": 5283,
            "url": "https://replicate.com/grandlineai/instant-id-artistic",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.ImageRef

    image: types.ImageRef = Field(default=types.ImageRef(), description="Input image")
    width: int = Field(
        title="Width",
        description="Width of output image",
        ge=512.0,
        le=2048.0,
        default=640,
    )
    height: int = Field(
        title="Height",
        description="Height of output image",
        ge=512.0,
        le=2048.0,
        default=640,
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality",
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Scale for classifier-free guidance",
        ge=1.0,
        le=50.0,
        default=5,
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Input Negative Prompt", default=""
    )
    ip_adapter_scale: float = Field(
        title="Ip Adapter Scale",
        description="Scale for IP adapter",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=30,
    )
    controlnet_conditioning_scale: float = Field(
        title="Controlnet Conditioning Scale",
        description="Scale for ControlNet conditioning",
        ge=0.0,
        le=1.0,
        default=0.8,
    )
