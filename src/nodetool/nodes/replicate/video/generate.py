from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class Ray(ReplicateNode):
    """Fast, high quality text-to-video and image-to-video (Also known as Dream Machine)"""

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
        return ["loop", "prompt", "end_image"]

    @classmethod
    def replicate_model_id(cls):
        return (
            "luma/ray:ace5984f3394f17c2a712644b0eb9983c4baaf94c6c30a0f94692d2c37bd8964"
        )

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/496f8ab2-3a87-4fe7-9867-5572460c2b5e/ray-cover.webp",
            "created_at": "2024-12-12T16:30:42.287210Z",
            "description": "Fast, high quality text-to-video and image-to-video (Also known as Dream Machine)",
            "github_url": None,
            "license_url": "https://lumalabs.ai/dream-machine/api/terms",
            "name": "ray",
            "owner": "luma",
            "is_official": True,
            "paper_url": "https://lumalabs.ai/dream-machine",
            "run_count": 69564,
            "url": "https://replicate.com/luma/ray",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    loop: bool = Field(
        title="Loop",
        description="Whether the video should loop, with the last frame matching the first frame for smooth, continuous playback. This input is ignored if end_image or end_video_id are set.",
        default=False,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for video generation", default=None
    )
    end_image: str | None = Field(
        title="End Image",
        description="An optional last frame of the video to use as the ending frame.",
        default=None,
    )
    start_image: str | None = Field(
        title="Start Image",
        description="An optional first frame of the video to use as the starting frame.",
        default=None,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the video. Ignored if a start frame, end frame or video ID is given.",
        default="16:9",
    )
    end_video_id: str | None = Field(
        title="End Video Id",
        description="Prepend a new video generation to the beginning of an existing one (Also called 'reverse extend'). You can combine this with start_image, or start_video_id.",
        default=None,
    )
    end_image_url: types.ImageRef = Field(
        default=types.ImageRef(), description="Deprecated: Use end_image instead"
    )
    start_video_id: str | None = Field(
        title="Start Video Id",
        description="Continue or extend a video generation with a new generation. You can combine this with end_image, or end_video_id.",
        default=None,
    )
    start_image_url: types.ImageRef = Field(
        default=types.ImageRef(), description="Deprecated: Use start_image instead"
    )


class HotshotXL(ReplicateNode):
    """😊 Hotshot-XL is an AI text-to-GIF model trained to work alongside Stable Diffusion XL"""

    class Width(int, Enum):
        _256 = 256
        _320 = 320
        _384 = 384
        _448 = 448
        _512 = 512
        _576 = 576
        _640 = 640
        _672 = 672
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Height(int, Enum):
        _256 = 256
        _320 = 320
        _384 = 384
        _448 = 448
        _512 = 512
        _576 = 576
        _640 = 640
        _672 = 672
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Scheduler(str, Enum):
        DDIMSCHEDULER = "DDIMScheduler"
        DPMSOLVERMULTISTEPSCHEDULER = "DPMSolverMultistepScheduler"
        HEUNDISCRETESCHEDULER = "HeunDiscreteScheduler"
        KARRASDPM = "KarrasDPM"
        EULERANCESTRALDISCRETESCHEDULER = "EulerAncestralDiscreteScheduler"
        EULERDISCRETESCHEDULER = "EulerDiscreteScheduler"
        PNDMSCHEDULER = "PNDMScheduler"

    @classmethod
    def get_basic_fields(cls):
        return ["mp4", "seed", "steps"]

    @classmethod
    def replicate_model_id(cls):
        return "lucataco/hotshot-xl:78b3a6257e16e4b241245d65c8b2b81ea2e1ff7ed4c55306b511509ddbfd327a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/70393e62-deab-4f95-ace7-eaeb8a9800db/compressed.gif",
            "created_at": "2023-10-05T04:09:21.646870Z",
            "description": "😊 Hotshot-XL is an AI text-to-GIF model trained to work alongside Stable Diffusion XL",
            "github_url": "https://github.com/lucataco/cog-hotshot-xl",
            "license_url": "https://github.com/hotshotco/Hotshot-XL/blob/main/LICENSE",
            "name": "hotshot-xl",
            "owner": "lucataco",
            "is_official": False,
            "paper_url": "https://huggingface.co/hotshotco/SDXL-512",
            "run_count": 893800,
            "url": "https://replicate.com/lucataco/hotshot-xl",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    mp4: bool = Field(
        title="Mp4", description="Save as mp4, False for GIF", default=False
    )
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    steps: int = Field(
        title="Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=30,
    )
    width: Width = Field(description="Width of the output", default=672)
    height: Height = Field(description="Height of the output", default=384)
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="a camel smoking a cigarette, hd, high quality",
    )
    scheduler: Scheduler = Field(
        description="Select a Scheduler", default="EulerAncestralDiscreteScheduler"
    )
    negative_prompt: str = Field(
        title="Negative Prompt", description="Negative prompt", default="blurry"
    )


class Zeroscope_V2_XL(ReplicateNode):
    """Zeroscope V2 XL & 576w"""

    class Model(str, Enum):
        XL = "xl"
        _576W = "576w"
        POTAT1 = "potat1"
        ANIMOV_512X = "animov-512x"

    @classmethod
    def get_basic_fields(cls):
        return ["fps", "seed", "model"]

    @classmethod
    def replicate_model_id(cls):
        return "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/d56e8888-a591-4edd-a9d3-2285b2ab66b4/1mrNnh8.jpg",
            "created_at": "2023-06-24T18:30:41.874899Z",
            "description": "Zeroscope V2 XL & 576w",
            "github_url": "https://github.com/anotherjesse/cog-text2video",
            "license_url": "https://github.com/anotherjesse/cog-text2video/blob/main/LICENSE",
            "name": "zeroscope-v2-xl",
            "owner": "anotherjesse",
            "is_official": False,
            "paper_url": "https://huggingface.co/cerspense/zeroscope_v2_576w",
            "run_count": 301932,
            "url": "https://replicate.com/anotherjesse/zeroscope-v2-xl",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    fps: int = Field(title="Fps", description="fps for the output video", default=8)
    seed: int | None = Field(
        title="Seed",
        description="Random seed. Leave blank to randomize the seed",
        default=None,
    )
    model: Model = Field(description="Model to use", default="xl")
    width: int = Field(
        title="Width", description="Width of the output video", ge=256.0, default=576
    )
    height: int = Field(
        title="Height", description="Height of the output video", ge=256.0, default=320
    )
    prompt: str = Field(
        title="Prompt",
        description="Input prompt",
        default="An astronaut riding a horse",
    )
    batch_size: int = Field(
        title="Batch Size", description="Batch size", ge=1.0, default=1
    )
    init_video: str | None = Field(
        title="Init Video",
        description="URL of the initial video (optional)",
        default=None,
    )
    num_frames: int = Field(
        title="Num Frames",
        description="Number of frames for the output video",
        default=24,
    )
    init_weight: float = Field(
        title="Init Weight", description="Strength of init_video", default=0.5
    )
    guidance_scale: float = Field(
        title="Guidance Scale",
        description="Guidance scale",
        ge=1.0,
        le=100.0,
        default=7.5,
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt", description="Negative prompt", default=None
    )
    remove_watermark: bool = Field(
        title="Remove Watermark", description="Remove watermark", default=False
    )
    num_inference_steps: int = Field(
        title="Num Inference Steps",
        description="Number of denoising steps",
        ge=1.0,
        le=500.0,
        default=50,
    )


class RobustVideoMatting(ReplicateNode):
    """extract foreground of a video"""

    class Output_type(str, Enum):
        GREEN_SCREEN = "green-screen"
        ALPHA_MASK = "alpha-mask"
        FOREGROUND_MASK = "foreground-mask"

    @classmethod
    def get_basic_fields(cls):
        return ["input_video", "output_type"]

    @classmethod
    def replicate_model_id(cls):
        return "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/1f92fd8f-2b90-4998-b5ae-1e23678ab004/showreel.gif",
            "created_at": "2022-11-25T14:06:18.152759Z",
            "description": "extract foreground of a video",
            "github_url": "https://github.com/PeterL1n/RobustVideoMatting",
            "license_url": "https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE",
            "name": "robust_video_matting",
            "owner": "arielreplicate",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2108.11515",
            "run_count": 73493,
            "url": "https://replicate.com/arielreplicate/robust_video_matting",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    input_video: types.VideoRef = Field(
        default=types.VideoRef(), description="Video to segment."
    )
    output_type: Output_type = Field(default="green-screen")


class AudioToWaveform(ReplicateNode):
    """Create a waveform video from audio"""

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "bg_color", "fg_alpha"]

    @classmethod
    def replicate_model_id(cls):
        return "fofr/audio-to-waveform:116cf9b97d0a117cfe64310637bf99ae8542cc35d813744c6ab178a3e134ff5a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/5d5cad9c-d4ba-44e1-8c4f-dc08648bbf5e/fofr_a_waveform_bar_chart_video_e.png",
            "created_at": "2023-06-13T15:26:38.672021Z",
            "description": "Create a waveform video from audio",
            "github_url": "https://github.com/fofr/audio-to-waveform",
            "license_url": "https://github.com/fofr/audio-to-waveform/blob/main/LICENSE",
            "name": "audio-to-waveform",
            "owner": "fofr",
            "is_official": False,
            "paper_url": "https://gradio.app/docs/#make_waveform",
            "run_count": 384020,
            "url": "https://replicate.com/fofr/audio-to-waveform",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    audio: types.AudioRef = Field(
        default=types.AudioRef(), description="Audio file to create waveform from"
    )
    bg_color: str = Field(
        title="Bg Color", description="Background color of waveform", default="#000000"
    )
    fg_alpha: float = Field(
        title="Fg Alpha", description="Opacity of foreground waveform", default=0.75
    )
    bar_count: int = Field(
        title="Bar Count", description="Number of bars in waveform", default=100
    )
    bar_width: float = Field(
        title="Bar Width",
        description="Width of bars in waveform. 1 represents full width, 0.5 represents half width, etc.",
        default=0.4,
    )
    bars_color: str = Field(
        title="Bars Color", description="Color of waveform bars", default="#ffffff"
    )
    caption_text: str = Field(
        title="Caption Text", description="Caption text for the video", default=""
    )


class Hunyuan_Video(ReplicateNode):
    """A state-of-the-art text-to-video generation model capable of creating high-quality videos with realistic motion from text descriptions"""

    @classmethod
    def get_basic_fields(cls):
        return ["fps", "seed", "width"]

    @classmethod
    def replicate_model_id(cls):
        return "tencent/hunyuan-video:6c9132aee14409cd6568d030453f1ba50f5f3412b844fe67f78a9eb62d55664f"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/3bebc89d-37c7-47ea-9a7b-a334b76eea87/hunyuan-featured.webp",
            "created_at": "2024-12-03T14:21:37.443615Z",
            "description": "A state-of-the-art text-to-video generation model capable of creating high-quality videos with realistic motion from text descriptions",
            "github_url": "https://github.com/zsxkib/HunyuanVideo/tree/replicate",
            "license_url": "https://huggingface.co/tencent/HunyuanVideo/blob/main/LICENSE",
            "name": "hunyuan-video",
            "owner": "tencent",
            "is_official": False,
            "paper_url": "https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf",
            "run_count": 116289,
            "url": "https://replicate.com/tencent/hunyuan-video",
            "visibility": "public",
            "weights_url": "https://huggingface.co/tencent/HunyuanVideo",
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    fps: int = Field(
        title="Fps",
        description="Frames per second of the output video",
        ge=1.0,
        default=24,
    )
    seed: int | None = Field(
        title="Seed", description="Random seed (leave empty for random)", default=None
    )
    width: int = Field(
        title="Width",
        description="Width of the video in pixels (must be divisible by 16)",
        ge=16.0,
        le=1280.0,
        default=864,
    )
    height: int = Field(
        title="Height",
        description="Height of the video in pixels (must be divisible by 16)",
        ge=16.0,
        le=1280.0,
        default=480,
    )
    prompt: str = Field(
        title="Prompt",
        description="The prompt to guide the video generation",
        default="A cat walks on the grass, realistic style",
    )
    infer_steps: int = Field(
        title="Infer Steps", description="Number of denoising steps", ge=1.0, default=50
    )
    video_length: int = Field(
        title="Video Length",
        description="Number of frames to generate (must be 4k+1, ex: 49 or 129)",
        ge=1.0,
        le=200.0,
        default=129,
    )
    embedded_guidance_scale: float = Field(
        title="Embedded Guidance Scale",
        description="Guidance scale",
        ge=1.0,
        le=10.0,
        default=6,
    )


class Video_01_Live(ReplicateNode):
    """An image-to-video (I2V) model specifically trained for Live2D and general animation use cases"""

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "prompt_optimizer", "first_frame_image"]

    @classmethod
    def replicate_model_id(cls):
        return "minimax/video-01-live:7574e16b8f1ad52c6332ecb264c0f132e555f46c222255a738131ec1bb614092"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/c202ad97-edd0-40b6-afaf-c99d71398d44/video-01-live-cover.webp",
            "created_at": "2024-12-16T20:27:52.715593Z",
            "description": "An image-to-video (I2V) model specifically trained for Live2D and general animation use cases",
            "github_url": None,
            "license_url": "https://intl.minimaxi.com/protocol/terms-of-service",
            "name": "video-01-live",
            "owner": "minimax",
            "is_official": True,
            "paper_url": None,
            "run_count": 173075,
            "url": "https://replicate.com/minimax/video-01-live",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    prompt: str | None = Field(
        title="Prompt", description="Text prompt for generation", default=None
    )
    prompt_optimizer: bool = Field(
        title="Prompt Optimizer", description="Use prompt optimizer", default=True
    )
    first_frame_image: str | None = Field(
        title="First Frame Image",
        description="First frame image for video generation. The output video will have the same aspect ratio as this image.",
        default=None,
    )


class Video_01(ReplicateNode):
    """Generate 6s videos with prompts or images. (Also known as Hailuo). Use a subject reference to make a video with a character and the S2V-01 model."""

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "prompt_optimizer", "first_frame_image"]

    @classmethod
    def replicate_model_id(cls):
        return "minimax/video-01:5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/b56c831c-4c68-4443-b69e-b71b105afe7f/minimax.webp",
            "created_at": "2024-11-26T14:40:21.652537Z",
            "description": "Generate 6s videos with prompts or images. (Also known as Hailuo). Use a subject reference to make a video with a character and the S2V-01 model.",
            "github_url": None,
            "license_url": "https://intl.minimaxi.com/protocol/terms-of-service",
            "name": "video-01",
            "owner": "minimax",
            "is_official": True,
            "paper_url": None,
            "run_count": 647422,
            "url": "https://replicate.com/minimax/video-01",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    prompt: str | None = Field(
        title="Prompt", description="Text prompt for generation", default=None
    )
    prompt_optimizer: bool = Field(
        title="Prompt Optimizer", description="Use prompt optimizer", default=True
    )
    first_frame_image: str | None = Field(
        title="First Frame Image",
        description="First frame image for video generation. The output video will have the same aspect ratio as this image.",
        default=None,
    )
    subject_reference: str | None = Field(
        title="Subject Reference",
        description="An optional character reference image to use as the subject in the generated video (this will use the S2V-01 model)",
        default=None,
    )


class Music_01(ReplicateNode):
    """Quickly generate up to 1 minute of music with lyrics and vocals in the style of a reference track"""

    class Bitrate(int, Enum):
        _32000 = 32000
        _64000 = 64000
        _128000 = 128000
        _256000 = 256000

    class Sample_rate(int, Enum):
        _16000 = 16000
        _24000 = 24000
        _32000 = 32000
        _44100 = 44100

    @classmethod
    def get_basic_fields(cls):
        return ["lyrics", "bitrate", "voice_id"]

    @classmethod
    def replicate_model_id(cls):
        return "minimax/music-01:0254c7e2f54315b667dbae03da7c155822ba29ffe0457be5bc246d564be486bd"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/1d2de931-15ff-48b0-9c4d-2f9200eb2913/music-01-cover.jpg",
            "created_at": "2024-12-17T12:40:30.320043Z",
            "description": "Quickly generate up to 1 minute of music with lyrics and vocals in the style of a reference track",
            "github_url": None,
            "license_url": "https://intl.minimaxi.com/protocol/terms-of-service",
            "name": "music-01",
            "owner": "minimax",
            "is_official": True,
            "paper_url": None,
            "run_count": 495367,
            "url": "https://replicate.com/minimax/music-01",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.AudioRef

    lyrics: str = Field(
        title="Lyrics",
        description="Lyrics with optional formatting. You can use a newline to separate each line of lyrics. You can use two newlines to add a pause between lines. You can use double hash marks (##) at the beginning and end of the lyrics to add accompaniment. Maximum 350 to 400 characters.",
        default="",
    )
    bitrate: Bitrate = Field(
        description="Bitrate for the generated music", default=256000
    )
    voice_id: str | None = Field(
        title="Voice Id",
        description="Reuse a previously uploaded voice ID",
        default=None,
    )
    song_file: types.AudioRef = Field(
        default=types.AudioRef(),
        description="Reference song, should contain music and vocals. Must be a .wav or .mp3 file longer than 15 seconds.",
    )
    voice_file: types.AudioRef = Field(
        default=types.AudioRef(),
        description="Voice reference. Must be a .wav or .mp3 file longer than 15 seconds. If only a voice reference is given, an a cappella vocal hum will be generated.",
    )
    sample_rate: Sample_rate = Field(
        description="Sample rate for the generated music", default=44100
    )
    instrumental_id: str | None = Field(
        title="Instrumental Id",
        description="Reuse a previously uploaded instrumental ID",
        default=None,
    )
    instrumental_file: str | None = Field(
        title="Instrumental File",
        description="Instrumental reference. Must be a .wav or .mp3 file longer than 15 seconds. If only an instrumental reference is given, a track without vocals will be generated.",
        default=None,
    )


class LTX_Video(ReplicateNode):
    """LTX-Video is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched."""

    class Model(str, Enum):
        _0_9_1 = "0.9.1"
        _0_9 = "0.9"

    class Length(int, Enum):
        _97 = 97
        _129 = 129
        _161 = 161
        _193 = 193
        _225 = 225
        _257 = 257

    class Target_size(int, Enum):
        _512 = 512
        _576 = 576
        _640 = 640
        _704 = 704
        _768 = 768
        _832 = 832
        _896 = 896
        _960 = 960
        _1024 = 1024

    class Aspect_ratio(str, Enum):
        _1_1 = "1:1"
        _1_2 = "1:2"
        _2_1 = "2:1"
        _2_3 = "2:3"
        _3_2 = "3:2"
        _3_4 = "3:4"
        _4_3 = "4:3"
        _4_5 = "4:5"
        _5_4 = "5:4"
        _9_16 = "9:16"
        _16_9 = "16:9"
        _9_21 = "9:21"
        _21_9 = "21:9"

    @classmethod
    def get_basic_fields(cls):
        return ["cfg", "seed", "image"]

    @classmethod
    def replicate_model_id(cls):
        return "lightricks/ltx-video:8c47da666861d081eeb4d1261853087de23923a268a69b63febdf5dc1dee08e4"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/184609cb-a0c5-47c9-8fec-987fb21cc977/replicate-prediction-_QTChmfY.webp",
            "created_at": "2024-11-29T14:15:01.460922Z",
            "description": "LTX-Video is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched.",
            "github_url": "https://github.com/Lightricks/LTX-Video",
            "license_url": "https://website.ltx.video/api-license-agreement",
            "name": "ltx-video",
            "owner": "lightricks",
            "is_official": False,
            "paper_url": None,
            "run_count": 164369,
            "url": "https://replicate.com/lightricks/ltx-video",
            "visibility": "public",
            "weights_url": "https://huggingface.co/Lightricks/LTX-Video",
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    cfg: float = Field(
        title="Cfg",
        description="How strongly the video follows the prompt",
        ge=1.0,
        le=20.0,
        default=3,
    )
    seed: int | None = Field(
        title="Seed",
        description="Set a seed for reproducibility. Random by default.",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Optional input image to use as the starting frame",
    )
    model: Model = Field(description="Model version to use", default="0.9.1")
    steps: int = Field(
        title="Steps", description="Number of steps", ge=1.0, le=50.0, default=30
    )
    length: Length = Field(
        description="Length of the output video in frames", default=97
    )
    prompt: str = Field(
        title="Prompt",
        description="Text prompt for the video. This model needs long descriptive prompts, if the prompt is too short the quality won't be good.",
        default="best quality, 4k, HDR, a tracking shot of a beautiful scene",
    )
    target_size: Target_size = Field(
        description="Target size for the output video", default=640
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the output video. Ignored if an image is provided.",
        default="3:2",
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want to see in your video",
        default="low quality, worst quality, deformed, distorted",
    )
    image_noise_scale: float = Field(
        title="Image Noise Scale",
        description="Lower numbers stick more closely to the input image",
        ge=0.0,
        le=1.0,
        default=0.15,
    )


class Wan_2_1_I2V_480p(ReplicateNode):
    """Accelerated inference for Wan 2.1 14B image to video, a comprehensive and open suite of video foundation models that pushes the boundaries of video generation."""

    class Fast_mode(str, Enum):
        OFF = "Off"
        BALANCED = "Balanced"
        FAST = "Fast"

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "wavespeedai/wan-2.1-i2v-480p:e2870aa4965fd9ddfd87c16a3c8ab952c18e745e63f3f3b123c2dc8b538ad2b5"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/75f0346d-ec4c-4078-bb40-6705578c0d21/replicate-prediction-br080xq9.webp",
            "created_at": "2025-02-26T20:09:34.365160Z",
            "description": "Accelerated inference for Wan 2.1 14B image to video, a comprehensive and open suite of video foundation models that pushes the boundaries of video generation.",
            "github_url": "https://github.com/Wan-Video/Wan2.1",
            "license_url": "https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt",
            "name": "wan-2.1-i2v-480p",
            "owner": "wavespeedai",
            "is_official": True,
            "paper_url": None,
            "run_count": 435128,
            "url": "https://replicate.com/wavespeedai/wan-2.1-i2v-480p",
            "visibility": "public",
            "weights_url": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B",
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(),
        description="Image for use as the initial frame of the video.",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for image generation", default=None
    )
    fast_mode: Fast_mode = Field(
        description="Speed up generation with different levels of acceleration. Faster modes may degrade quality somewhat. The speedup is dependent on the content, so different videos may see different speedups.",
        default="Balanced",
    )
    lora_scale: float = Field(
        title="Lora Scale",
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. You may still need to experiment to find the best value for your particular lora.",
        ge=0.0,
        le=4.0,
        default=1,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the output video.", default="16:9"
    )
    lora_weights: str | None = Field(
        title="Lora Weights",
        description="Load LoRA weights. Supports HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet.",
        default=None,
    )
    sample_shift: int = Field(
        title="Sample Shift",
        description="Flow shift parameter for video generation",
        ge=0.0,
        le=10.0,
        default=3,
    )
    sample_steps: int = Field(
        title="Sample Steps",
        description="Number of inference steps",
        ge=1.0,
        le=40.0,
        default=30,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative prompt to avoid certain elements",
        default="",
    )
    sample_guide_scale: float = Field(
        title="Sample Guide Scale",
        description="Guidance scale for generation",
        ge=1.0,
        le=10.0,
        default=5,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated videos",
        default=False,
    )


class Wan_2_1_1_3B(ReplicateNode):
    """Generate 5s 480p videos. Wan is an advanced and powerful visual generation model developed by Tongyi Lab of Alibaba Group"""

    class Frame_num(int, Enum):
        _17 = 17
        _33 = 33
        _49 = 49
        _65 = 65
        _81 = 81

    class Resolution(str, Enum):
        _480P = "480p"

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "frame_num"]

    @classmethod
    def replicate_model_id(cls):
        return "wan-video/wan-2.1-1.3b:121bbb762bf449889f090d36e3598c72c50c7a8cc2ce250433bc521a562aae61"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/a07f6849-fd97-4f6b-9166-887e84b0cb47/replicate-prediction-0s06z711.webp",
            "created_at": "2025-02-26T14:24:14.098215Z",
            "description": "Generate 5s 480p videos. Wan is an advanced and powerful visual generation model developed by Tongyi Lab of Alibaba Group",
            "github_url": "https://github.com/Wan-Video/Wan2.1",
            "license_url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "name": "wan-2.1-1.3b",
            "owner": "wan-video",
            "is_official": True,
            "paper_url": "https://wanxai.com/",
            "run_count": 46523,
            "url": "https://replicate.com/wan-video/wan-2.1-1.3b",
            "visibility": "public",
            "weights_url": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B",
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed for reproducible results (leave blank for random)",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="Text prompt describing what you want to generate",
        default=None,
    )
    frame_num: Frame_num = Field(
        description="Video duration in frames (based on standard 16fps playback)",
        default=81,
    )
    resolution: Resolution = Field(description="Video resolution", default="480p")
    aspect_ratio: Aspect_ratio = Field(description="Video aspect ratio", default="16:9")
    sample_shift: float = Field(
        title="Sample Shift",
        description="Sampling shift factor for flow matching (recommended range: 8-12)",
        ge=0.0,
        le=20.0,
        default=8,
    )
    sample_steps: int = Field(
        title="Sample Steps",
        description="Number of sampling steps (higher = better quality but slower)",
        ge=10.0,
        le=50.0,
        default=30,
    )
    sample_guide_scale: float = Field(
        title="Sample Guide Scale",
        description="Classifier free guidance scale (higher values strengthen prompt adherence)",
        ge=0.0,
        le=20.0,
        default=6,
    )


class Pixverse_V5(ReplicateNode):
    """Create 5s-8s videos with enhanced character movement, visual effects, and exclusive 1080p-8s support. Optimized for anime characters and complex actions"""

    class Effect(str, Enum):
        NONE = "None"
        LET_S_YMCA = "Let's YMCA!"
        SUBJECT_3_FEVER = "Subject 3 Fever"
        GHIBLI_LIVE = "Ghibli Live!"
        SUIT_SWAGGER = "Suit Swagger"
        MUSCLE_SURGE = "Muscle Surge"
        _360__MICROWAVE = "360° Microwave"
        WARMTH_OF_JESUS = "Warmth of Jesus"
        EMERGENCY_BEAT = "Emergency Beat"
        ANYTHING__ROBOT = "Anything, Robot"
        KUNGFU_CLUB = "Kungfu Club"
        MINT_IN_BOX = "Mint in Box"
        RETRO_ANIME_POP = "Retro Anime Pop"
        VOGUE_WALK = "Vogue Walk"
        MEGA_DIVE = "Mega Dive"
        EVIL_TRIGGER = "Evil Trigger"

    class Quality(str, Enum):
        _360P = "360p"
        _540P = "540p"
        _720P = "720p"
        _1080P = "1080p"

    class Duration(int, Enum):
        _5 = 5
        _8 = 8

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"
        _1_1 = "1:1"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "effect"]

    @classmethod
    def replicate_model_id(cls):
        return "pixverse/pixverse-v5:450181c56fcbf920d8d5ba9d7c5653537a009b626652c1a0a909924a785e3389"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/17daf77f-f059-4703-bbf8-284cfdf3a066/cover-tmp2sgncxf1.webp",
            "created_at": "2025-08-27T16:59:19.868043Z",
            "description": "Create 5s-8s videos with enhanced character movement, visual effects, and exclusive 1080p-8s support. Optimized for anime characters and complex actions",
            "github_url": None,
            "license_url": None,
            "name": "pixverse-v5",
            "owner": "pixverse",
            "is_official": True,
            "paper_url": None,
            "run_count": 752450,
            "url": "https://replicate.com/pixverse/pixverse-v5",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: str | None = Field(
        title="Image",
        description="Image to use for the first frame of the video",
        default=None,
    )
    effect: Effect = Field(
        description="Special effect to apply to the video. V5 supports effects. Does not work with last_frame_image.",
        default="None",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for video generation", default=None
    )
    quality: Quality = Field(
        description="Resolution of the video. 360p and 540p cost the same, but 720p and 1080p cost more. V5 supports 1080p with 8 second duration.",
        default="540p",
    )
    duration: Duration = Field(
        description="Duration of the video in seconds. 8 second videos cost twice as much as 5 second videos. V5 supports 1080p with 8 second duration.",
        default=5,
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of the video", default="16:9"
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Negative prompt to avoid certain elements in the video",
        default="",
    )
    last_frame_image: str | None = Field(
        title="Last Frame Image",
        description="Use to generate a video that transitions from the first image to the last image. Must be used with image.",
        default=None,
    )


class Gen4_Turbo(ReplicateNode):
    """Generate 5s and 10s 720p videos fast"""

    class Duration(int, Enum):
        _5 = 5
        _10 = 10

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _1_1 = "1:1"
        _21_9 = "21:9"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "runwayml/gen4-turbo:6257a44f7b6390e47eb18a1c11f55d221fc90ec056d9acfe490ec9924739533c"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/32f1975a-faa7-419b-8781-01c2f1593dc4/replicate-prediction-y3p6xca9r.mp4",
            "created_at": "2025-07-21T16:53:27.233701Z",
            "description": "Generate 5s and 10s 720p videos fast",
            "github_url": None,
            "license_url": None,
            "name": "gen4-turbo",
            "owner": "runwayml",
            "is_official": True,
            "paper_url": None,
            "run_count": 52472,
            "url": "https://replicate.com/runwayml/gen4-turbo",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    image: str | None = Field(
        title="Image",
        description="Initial image for video generation (first frame)",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for video generation", default=None
    )
    duration: Duration = Field(
        description="Duration of the output video in seconds", default=5
    )
    aspect_ratio: Aspect_ratio = Field(description="Video aspect ratio", default="16:9")


class Gen4_Aleph(ReplicateNode):
    """A new way to edit, transform and generate video"""

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"
        _4_3 = "4:3"
        _3_4 = "3:4"
        _1_1 = "1:1"
        _21_9 = "21:9"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "video", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "runwayml/gen4-aleph:68cabc3b111f47bd881cffaca63ad0b1e7834c77737e042cec6eca18962ce1d2"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/b8148ec5-c33f-4e04-a006-2d5c7d4d092a/combined_left_right.mp4",
            "created_at": "2025-08-05T09:25:34.423358Z",
            "description": "A new way to edit, transform and generate video",
            "github_url": None,
            "license_url": None,
            "name": "gen4-aleph",
            "owner": "runwayml",
            "is_official": True,
            "paper_url": "https://runwayml.com/research/introducing-runway-aleph",
            "run_count": 118641,
            "url": "https://replicate.com/runwayml/gen4-aleph",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Set for reproducible generation",
        default=None,
    )
    video: str | None = Field(
        title="Video",
        description="Input video to generate from. Videos must be less than 16MB. Only 5s of the input video will be used.",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for video generation", default=None
    )
    aspect_ratio: Aspect_ratio = Field(description="Video aspect ratio", default="16:9")
    reference_image: str | None = Field(
        title="Reference Image",
        description="Reference image to influence the style or content of the output.",
        default=None,
    )


class Kling_V2_1(ReplicateNode):
    """Use Kling v2.1 to generate 5s and 10s videos in 720p and 1080p resolution from a starting image (image-to-video)"""

    class Mode(str, Enum):
        STANDARD = "standard"
        PRO = "pro"

    class Duration(int, Enum):
        _5 = 5
        _10 = 10

    @classmethod
    def get_basic_fields(cls):
        return ["mode", "prompt", "duration"]

    @classmethod
    def replicate_model_id(cls):
        return "kwaivgi/kling-v2.1:daad218feb714b03e2a1ac445986aebb9d05243cd00da2af17be2e4049f48f69"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/a7690882-d1d2-44fb-b487-f41bd367adcf/replicate-prediction-2epyczsz.webp",
            "created_at": "2025-06-19T18:53:08.443897Z",
            "description": "Use Kling v2.1 to generate 5s and 10s videos in 720p and 1080p resolution from a starting image (image-to-video)",
            "github_url": None,
            "license_url": None,
            "name": "kling-v2.1",
            "owner": "kwaivgi",
            "is_official": True,
            "paper_url": None,
            "run_count": 3221755,
            "url": "https://replicate.com/kwaivgi/kling-v2.1",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    mode: Mode = Field(
        description="Standard has a resolution of 720p, pro is 1080p. Both are 24fps.",
        default="standard",
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for video generation", default=None
    )
    duration: Duration = Field(
        description="Duration of the video in seconds", default=5
    )
    end_image: str | None = Field(
        title="End Image",
        description="Last frame of the video (pro mode is required when this parameter is set)",
        default=None,
    )
    start_image: str | None = Field(
        title="Start Image",
        description="First frame of the video. You must use a start image with kling-v2.1.",
        default=None,
    )
    negative_prompt: str = Field(
        title="Negative Prompt",
        description="Things you do not want to see in the video",
        default="",
    )


class Kling_Lip_Sync(ReplicateNode):
    """Add lip-sync to any video with an audio file or text"""

    class Voice_id(str, Enum):
        EN_AOT = "en_AOT"
        EN_OVERSEA_MALE1 = "en_oversea_male1"
        EN_GIRLFRIEND_4_SPEECH02 = "en_girlfriend_4_speech02"
        EN_CHAT_0407_5_1 = "en_chat_0407_5-1"
        EN_UK_BOY1 = "en_uk_boy1"
        EN_PEPPAPIG_PLATFORM = "en_PeppaPig_platform"
        EN_AI_HUANGZHONG_712 = "en_ai_huangzhong_712"
        EN_CALM_STORY1 = "en_calm_story1"
        EN_UK_MAN2 = "en_uk_man2"
        EN_READER_EN_M_V1 = "en_reader_en_m-v1"
        EN_COMMERCIAL_LADY_EN_F_V1 = "en_commercial_lady_en_f-v1"
        ZH_GENSHIN_VINDI2 = "zh_genshin_vindi2"
        ZH_ZHINEN_XUESHENG = "zh_zhinen_xuesheng"
        ZH_TIYUXI_XUEDI = "zh_tiyuxi_xuedi"
        ZH_AI_SHATANG = "zh_ai_shatang"
        ZH_GENSHIN_KLEE2 = "zh_genshin_klee2"
        ZH_GENSHIN_KIRARA = "zh_genshin_kirara"
        ZH_AI_KAIYA = "zh_ai_kaiya"
        ZH_TIEXIN_NANYOU = "zh_tiexin_nanyou"
        ZH_AI_CHENJIAHAO_712 = "zh_ai_chenjiahao_712"
        ZH_GIRLFRIEND_1_SPEECH02 = "zh_girlfriend_1_speech02"
        ZH_CHAT1_FEMALE_NEW_3 = "zh_chat1_female_new-3"
        ZH_GIRLFRIEND_2_SPEECH02 = "zh_girlfriend_2_speech02"
        ZH_CARTOON_BOY_07 = "zh_cartoon-boy-07"
        ZH_CARTOON_GIRL_01 = "zh_cartoon-girl-01"
        ZH_AI_HUANGYAOSHI_712 = "zh_ai_huangyaoshi_712"
        ZH_YOU_PINGJING = "zh_you_pingjing"
        ZH_AI_LAOGUOWANG_712 = "zh_ai_laoguowang_712"
        ZH_CHENGSHU_JIEJIE = "zh_chengshu_jiejie"
        ZH_ZHUXI_SPEECH02 = "zh_zhuxi_speech02"
        ZH_UK_OLDMAN3 = "zh_uk_oldman3"
        ZH_LAOPOPO_SPEECH02 = "zh_laopopo_speech02"
        ZH_HEAINAINAI_SPEECH02 = "zh_heainainai_speech02"
        ZH_DONGBEILAOTIE_SPEECH02 = "zh_dongbeilaotie_speech02"
        ZH_CHONGQINGXIAOHUO_SPEECH02 = "zh_chongqingxiaohuo_speech02"
        ZH_CHUANMEIZI_SPEECH02 = "zh_chuanmeizi_speech02"
        ZH_CHAOSHANDASHU_SPEECH02 = "zh_chaoshandashu_speech02"
        ZH_AI_TAIWAN_MAN2_SPEECH02 = "zh_ai_taiwan_man2_speech02"
        ZH_XIANZHANGGUI_SPEECH02 = "zh_xianzhanggui_speech02"
        ZH_TIANJINJIEJIE_SPEECH02 = "zh_tianjinjiejie_speech02"
        ZH_DIYINNANSANG_DB_CN_M_04_V2 = "zh_diyinnansang_DB_CN_M_04-v2"
        ZH_YIZHIPIANNAN_V1 = "zh_yizhipiannan-v1"
        ZH_GUANXIAOFANG_V2 = "zh_guanxiaofang-v2"
        ZH_TIANMEIXUEMEI_V1 = "zh_tianmeixuemei-v1"
        ZH_DAOPIANYANSANG_V1 = "zh_daopianyansang-v1"
        ZH_MENGWA_V1 = "zh_mengwa-v1"

    @classmethod
    def get_basic_fields(cls):
        return ["text", "video_id", "voice_id"]

    @classmethod
    def replicate_model_id(cls):
        return "kwaivgi/kling-lip-sync:8311467f07043d4b3feb44584d2586bfa2fc70203eca612ed26f84d0b55df3ce"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/92rRwTYlfo0BLSIIcirPaRwPtJhN0lrl4ww79omyef38rCdpA/tmp2ni84f_5.mp4",
            "created_at": "2025-05-18T20:54:54.551885Z",
            "description": "Add lip-sync to any video with an audio file or text",
            "github_url": None,
            "license_url": None,
            "name": "kling-lip-sync",
            "owner": "kwaivgi",
            "is_official": True,
            "paper_url": None,
            "run_count": 27998,
            "url": "https://replicate.com/kwaivgi/kling-lip-sync",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    text: str | None = Field(
        title="Text",
        description="Text content for lip sync (if not using audio)",
        default=None,
    )
    video_id: str | None = Field(
        title="Video Id",
        description="ID of a video generated by Kling. Cannot be used with video_url.",
        default=None,
    )
    voice_id: Voice_id = Field(
        description="Voice ID for speech synthesis (if using text and not audio)",
        default="en_AOT",
    )
    video_url: str | None = Field(
        title="Video Url",
        description="URL of a video for lip syncing. It can be an .mp4 or .mov file, should be less than 100MB, with a duration of 2-10 seconds, and a resolution of 720p-1080p (720-1920px dimensions). Cannot be used with video_id.",
        default=None,
    )
    audio_file: str | None = Field(
        title="Audio File",
        description="Audio file for lip sync. Must be .mp3, .wav, .m4a, or .aac and less than 5MB.",
        default=None,
    )
    voice_speed: float = Field(
        title="Voice Speed",
        description="Speech rate (only used if using text and not audio)",
        ge=0.8,
        le=2.0,
        default=1,
    )


class Hailuo_02(ReplicateNode):
    """Hailuo 2 is a text-to-video and image-to-video model that can make 6s or 10s videos at 768p (standard) or 1080p (pro). It excels at real world physics."""

    class Duration(int, Enum):
        _6 = 6
        _10 = 10

    class Resolution(str, Enum):
        _512P = "512p"
        _768P = "768p"
        _1080P = "1080p"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "resolution"]

    @classmethod
    def replicate_model_id(cls):
        return "minimax/hailuo-02:baaadb886e09b1e711387e270d841930e8253f08775bc6cb176580658f0f2fd9"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/e2953b9d-ef83-4a4a-b90f-1ef168223bf2/tmpbufkswx1.mp4",
            "created_at": "2025-07-02T11:02:01.844243Z",
            "description": "Hailuo 2 is a text-to-video and image-to-video model that can make 6s or 10s videos at 768p (standard) or 1080p (pro). It excels at real world physics.",
            "github_url": None,
            "license_url": None,
            "name": "hailuo-02",
            "owner": "minimax",
            "is_official": True,
            "paper_url": None,
            "run_count": 307221,
            "url": "https://replicate.com/minimax/hailuo-02",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    prompt: str | None = Field(
        title="Prompt", description="Text prompt for generation", default=None
    )
    duration: Duration = Field(
        description="Duration of the video in seconds. 10 seconds is only available for 768p resolution.",
        default=6,
    )
    resolution: Resolution = Field(
        description="Pick between standard 512p, 768p, or pro 1080p resolution. The pro model is not just high resolution, it is also higher quality.",
        default="1080p",
    )
    last_frame_image: str | None = Field(
        title="Last Frame Image",
        description="Last frame image for video generation. The final frame of the output video will match this image.",
        default=None,
    )
    prompt_optimizer: bool = Field(
        title="Prompt Optimizer", description="Use prompt optimizer", default=True
    )
    first_frame_image: str | None = Field(
        title="First Frame Image",
        description="First frame image for video generation. The output video will have the same aspect ratio as this image.",
        default=None,
    )


class Wan_2_2_T2V_Fast(ReplicateNode):
    """A very fast and cheap PrunaAI optimized version of Wan 2.2 A14B text-to-video"""

    class Resolution(str, Enum):
        _480P = "480p"
        _720P = "720p"

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "prompt", "go_fast"]

    @classmethod
    def replicate_model_id(cls):
        return "wan-video/wan-2.2-t2v-fast:c483b1f7b892065bc58ebadb6381abf557f6b1f517d2ff0febb3fb635cf49b4d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/08195778-ca4f-401d-b1eb-fecc242b97c0/output.mp4",
            "created_at": "2025-07-30T11:03:52.081987Z",
            "description": "A very fast and cheap PrunaAI optimized version of Wan 2.2 A14B text-to-video",
            "github_url": None,
            "license_url": None,
            "name": "wan-2.2-t2v-fast",
            "owner": "wan-video",
            "is_official": True,
            "paper_url": None,
            "run_count": 164438,
            "url": "https://replicate.com/wan-video/wan-2.2-t2v-fast",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed", description="Random seed. Leave blank for random", default=None
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for video generation", default=None
    )
    go_fast: bool = Field(title="Go Fast", description="Go fast", default=True)
    num_frames: int = Field(
        title="Num Frames",
        description="Number of video frames. 81 frames give the best results",
        ge=81.0,
        le=121.0,
        default=81,
    )
    resolution: Resolution = Field(
        description="Resolution of video. 16:9 corresponds to 832x480px, and 9:16 is 480x832px",
        default="480p",
    )
    aspect_ratio: Aspect_ratio = Field(
        description="Aspect ratio of video. 16:9 corresponds to 832x480px, and 9:16 is 480x832px",
        default="16:9",
    )
    sample_shift: float = Field(
        title="Sample Shift",
        description="Sample shift factor",
        ge=1.0,
        le=20.0,
        default=12,
    )
    optimize_prompt: bool = Field(
        title="Optimize Prompt",
        description="Translate prompt to Chinese before generation",
        default=False,
    )
    frames_per_second: int = Field(
        title="Frames Per Second",
        description="Frames per second. Note that the pricing of this model is based on the video duration at 16 fps",
        ge=5.0,
        le=30.0,
        default=16,
    )
    interpolate_output: bool = Field(
        title="Interpolate Output",
        description="Interpolate the generated video to 30 FPS using ffmpeg",
        default=True,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated video.",
        default=False,
    )
    lora_scale_transformer: float = Field(
        title="Lora Scale Transformer",
        description="Determines how strongly the transformer LoRA should be applied.",
        default=1,
    )
    lora_scale_transformer_2: float = Field(
        title="Lora Scale Transformer 2",
        description="Determines how strongly the transformer_2 LoRA should be applied.",
        default=1,
    )
    lora_weights_transformer: str | None = Field(
        title="Lora Weights Transformer",
        description="Load LoRA weights for transformer. Supports arbitrary .safetensors URLs from the Internet (for example, 'https://huggingface.co/Viktor1717/scandinavian-interior-style1/resolve/main/my_first_flux_lora_v1.safetensors')",
        default=None,
    )
    lora_weights_transformer_2: str | None = Field(
        title="Lora Weights Transformer 2",
        description="Load LoRA weights for transformer_2. Supports arbitrary .safetensors URLs from the Internet. Can be different from transformer LoRA.",
        default=None,
    )


class Wan_2_2_I2V_Fast(ReplicateNode):
    """A very fast and cheap PrunaAI optimized version of Wan 2.2 A14B image-to-video"""

    class Resolution(str, Enum):
        _480P = "480p"
        _720P = "720p"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "wan-video/wan-2.2-i2v-fast:4eaf2b01d3bf70d8a2e00b219efeb7cb415855ad18b7dacdc4cae664a73a6eea"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/e04b60c1-dbc9-4606-8b37-6da8c7e27f5e/replicate-prediction-xd6kc1a96.mp4",
            "created_at": "2025-07-30T18:13:03.381058Z",
            "description": "A very fast and cheap PrunaAI optimized version of Wan 2.2 A14B image-to-video",
            "github_url": None,
            "license_url": None,
            "name": "wan-2.2-i2v-fast",
            "owner": "wan-video",
            "is_official": True,
            "paper_url": None,
            "run_count": 6181679,
            "url": "https://replicate.com/wan-video/wan-2.2-i2v-fast",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed", description="Random seed. Leave blank for random", default=None
    )
    image: types.ImageRef = Field(
        default=types.ImageRef(), description="Input image to generate video from."
    )
    prompt: str | None = Field(
        title="Prompt", description="Prompt for video generation", default=None
    )
    go_fast: bool = Field(title="Go Fast", description="Go fast", default=True)
    last_image: str | None = Field(
        title="Last Image",
        description="Optional last image to condition the video generation. If provided, creates smoother transitions between frames.",
        default=None,
    )
    num_frames: int = Field(
        title="Num Frames",
        description="Number of video frames. 81 frames give the best results",
        ge=81.0,
        le=121.0,
        default=81,
    )
    resolution: Resolution = Field(
        description="Resolution of video. 16:9 corresponds to 832x480px, and 9:16 is 480x832px",
        default="480p",
    )
    sample_shift: float = Field(
        title="Sample Shift",
        description="Sample shift factor",
        ge=1.0,
        le=20.0,
        default=12,
    )
    frames_per_second: int = Field(
        title="Frames Per Second",
        description="Frames per second. Note that the pricing of this model is based on the video duration at 16 fps",
        ge=5.0,
        le=30.0,
        default=16,
    )
    interpolate_output: bool = Field(
        title="Interpolate Output",
        description="Interpolate the generated video to 30 FPS using ffmpeg",
        default=False,
    )
    disable_safety_checker: bool = Field(
        title="Disable Safety Checker",
        description="Disable safety checker for generated video.",
        default=False,
    )
    lora_scale_transformer: float = Field(
        title="Lora Scale Transformer",
        description="Determines how strongly the transformer LoRA should be applied.",
        default=1,
    )
    lora_scale_transformer_2: float = Field(
        title="Lora Scale Transformer 2",
        description="Determines how strongly the transformer_2 LoRA should be applied.",
        default=1,
    )
    lora_weights_transformer: str | None = Field(
        title="Lora Weights Transformer",
        description="Load LoRA weights for the HIGH transformer. Supports arbitrary .safetensors URLs from the Internet (for example, 'https://huggingface.co/TheRaf7/instagirl-v2/resolve/main/Instagirlv2.0_hinoise.safetensors')",
        default=None,
    )
    lora_weights_transformer_2: str | None = Field(
        title="Lora Weights Transformer 2",
        description="Load LoRA weights for the LOW transformer_2. Supports arbitrary .safetensors URLs from the Internet. Can be different from transformer LoRA. (for example, 'https://huggingface.co/TheRaf7/instagirl-v2/resolve/main/Instagirlv2.0_lownoise.safetensors')",
        default=None,
    )


class Lipsync_2(ReplicateNode):
    """Generate realistic lipsyncs with Sync Labs' 2.0 model"""

    class Sync_mode(str, Enum):
        LOOP = "loop"
        BOUNCE = "bounce"
        CUT_OFF = "cut_off"
        SILENCE = "silence"
        REMAP = "remap"

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "video", "sync_mode"]

    @classmethod
    def replicate_model_id(cls):
        return "sync/lipsync-2:3190ef7dc0cbca29458d0032c032ef140a840087141cf10333e8d19a213f9194"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/wJSCHDW1zvqcBtInTLFK9rW6N52db0y5FImHrJlyjYi12gQF/tmpr6cxkcvr.mp4",
            "created_at": "2025-07-15T14:19:02.814147Z",
            "description": "Generate realistic lipsyncs with Sync Labs' 2.0 model",
            "github_url": None,
            "license_url": None,
            "name": "lipsync-2",
            "owner": "sync",
            "is_official": True,
            "paper_url": None,
            "run_count": 16137,
            "url": "https://replicate.com/sync/lipsync-2",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    audio: types.AudioRef = Field(
        default=types.AudioRef(), description="Input audio file (.wav)"
    )
    video: types.VideoRef = Field(
        default=types.VideoRef(), description="Input video file (.mp4)"
    )
    sync_mode: Sync_mode = Field(
        description="Lipsync mode when audio and video durations are out of sync",
        default="loop",
    )
    temperature: float = Field(
        title="Temperature",
        description="How expressive lipsync can be (0-1)",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    active_speaker: bool = Field(
        title="Active Speaker",
        description="Whether to detect active speaker (i.e. whoever is speaking in the clip will be used for lipsync)",
        default=False,
    )


class Lipsync_2_Pro(ReplicateNode):
    """Studio-grade lipsync in minutes, not weeks"""

    class Sync_mode(str, Enum):
        LOOP = "loop"
        BOUNCE = "bounce"
        CUT_OFF = "cut_off"
        SILENCE = "silence"
        REMAP = "remap"

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "video", "sync_mode"]

    @classmethod
    def replicate_model_id(cls):
        return "sync/lipsync-2-pro:eaad6bceea4938d05f5d984b22897e5a7d389d4fff9a70888af5718502b57d39"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://replicate.delivery/xezq/boKOqiuyz9omClm432RN1vzB3wejNr36em4lmJ2SC6If6KnqA/tmp47bgpsi_.mp4",
            "created_at": "2025-08-27T19:58:34.559850Z",
            "description": "Studio-grade lipsync in minutes, not weeks",
            "github_url": None,
            "license_url": None,
            "name": "lipsync-2-pro",
            "owner": "sync",
            "is_official": True,
            "paper_url": None,
            "run_count": 10209,
            "url": "https://replicate.com/sync/lipsync-2-pro",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    audio: types.AudioRef = Field(
        default=types.AudioRef(), description="Input audio file (.wav)"
    )
    video: types.VideoRef = Field(
        default=types.VideoRef(), description="Input video file (.mp4)"
    )
    sync_mode: Sync_mode = Field(
        description="Lipsync mode when audio and video durations are out of sync",
        default="loop",
    )
    temperature: float = Field(
        title="Temperature",
        description="How expressive lipsync can be (0-1)",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    active_speaker: bool = Field(
        title="Active Speaker",
        description="Whether to detect active speaker (i.e. whoever is speaking in the clip will be used for lipsync)",
        default=False,
    )


class Veo_3_1(ReplicateNode):
    """New and improved version of Veo 3, with higher-fidelity video, context-aware audio, reference image and last frame support"""

    class Duration(int, Enum):
        _4 = 4
        _6 = 6
        _8 = 8

    class Resolution(str, Enum):
        _720P = "720p"
        _1080P = "1080p"

    class Aspect_ratio(str, Enum):
        _16_9 = "16:9"
        _9_16 = "9:16"

    @classmethod
    def get_basic_fields(cls):
        return ["seed", "image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "google/veo-3.1:ed5b1767b711dd15d954b162af1e890d27882680f463a85e94f02d604012b972"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/7600d853-2847-4c46-afc9-faef04fea2c5/veo3.1-sm.mp4",
            "created_at": "2025-10-10T12:09:23.785590Z",
            "description": "New and improved version of Veo 3, with higher-fidelity video, context-aware audio, reference image and last frame support",
            "github_url": None,
            "license_url": None,
            "name": "veo-3.1",
            "owner": "google",
            "is_official": True,
            "paper_url": None,
            "run_count": 298485,
            "url": "https://replicate.com/google/veo-3.1",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return types.VideoRef

    seed: int | None = Field(
        title="Seed",
        description="Random seed. Omit for random generations",
        default=None,
    )
    image: str | None = Field(
        title="Image",
        description="Input image to start generating from. Ideal images are 16:9 or 9:16 and 1280x720 or 720x1280, depending on the aspect ratio you choose.",
        default=None,
    )
    prompt: str | None = Field(
        title="Prompt", description="Text prompt for video generation", default=None
    )
    duration: Duration = Field(description="Video duration in seconds", default=8)
    last_frame: str | None = Field(
        title="Last Frame",
        description="Ending image for interpolation. When provided with an input image, creates a transition between the two images.",
        default=None,
    )
    resolution: Resolution = Field(
        description="Resolution of the generated video", default="1080p"
    )
    aspect_ratio: Aspect_ratio = Field(description="Video aspect ratio", default="16:9")
    generate_audio: bool = Field(
        title="Generate Audio",
        description="Generate audio with the video",
        default=True,
    )
    negative_prompt: str | None = Field(
        title="Negative Prompt",
        description="Description of what to exclude from the generated video",
        default=None,
    )
    reference_images: list = Field(
        title="Reference Images",
        description="1 to 3 reference images for subject-consistent generation (reference-to-video, or R2V). Reference images only work with 16:9 aspect ratio and 8-second duration. Last frame is ignored if reference images are provided.",
        default=[],
    )
