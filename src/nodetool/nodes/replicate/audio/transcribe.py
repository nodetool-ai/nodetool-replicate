from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class IncrediblyFastWhisper(ReplicateNode):
    """whisper-large-v3, incredibly fast, powered by Hugging Face Transformers! 🤗"""

    class Task(str, Enum):
        TRANSCRIBE = "transcribe"
        TRANSLATE = "translate"

    class Language(str, Enum):
        NONE = "None"
        AFRIKAANS = "afrikaans"
        ALBANIAN = "albanian"
        AMHARIC = "amharic"
        ARABIC = "arabic"
        ARMENIAN = "armenian"
        ASSAMESE = "assamese"
        AZERBAIJANI = "azerbaijani"
        BASHKIR = "bashkir"
        BASQUE = "basque"
        BELARUSIAN = "belarusian"
        BENGALI = "bengali"
        BOSNIAN = "bosnian"
        BRETON = "breton"
        BULGARIAN = "bulgarian"
        CANTONESE = "cantonese"
        CATALAN = "catalan"
        CHINESE = "chinese"
        CROATIAN = "croatian"
        CZECH = "czech"
        DANISH = "danish"
        DUTCH = "dutch"
        ENGLISH = "english"
        ESTONIAN = "estonian"
        FAROESE = "faroese"
        FINNISH = "finnish"
        FRENCH = "french"
        GALICIAN = "galician"
        GEORGIAN = "georgian"
        GERMAN = "german"
        GREEK = "greek"
        GUJARATI = "gujarati"
        HAITIAN_CREOLE = "haitian creole"
        HAUSA = "hausa"
        HAWAIIAN = "hawaiian"
        HEBREW = "hebrew"
        HINDI = "hindi"
        HUNGARIAN = "hungarian"
        ICELANDIC = "icelandic"
        INDONESIAN = "indonesian"
        ITALIAN = "italian"
        JAPANESE = "japanese"
        JAVANESE = "javanese"
        KANNADA = "kannada"
        KAZAKH = "kazakh"
        KHMER = "khmer"
        KOREAN = "korean"
        LAO = "lao"
        LATIN = "latin"
        LATVIAN = "latvian"
        LINGALA = "lingala"
        LITHUANIAN = "lithuanian"
        LUXEMBOURGISH = "luxembourgish"
        MACEDONIAN = "macedonian"
        MALAGASY = "malagasy"
        MALAY = "malay"
        MALAYALAM = "malayalam"
        MALTESE = "maltese"
        MAORI = "maori"
        MARATHI = "marathi"
        MONGOLIAN = "mongolian"
        MYANMAR = "myanmar"
        NEPALI = "nepali"
        NORWEGIAN = "norwegian"
        NYNORSK = "nynorsk"
        OCCITAN = "occitan"
        PASHTO = "pashto"
        PERSIAN = "persian"
        POLISH = "polish"
        PORTUGUESE = "portuguese"
        PUNJABI = "punjabi"
        ROMANIAN = "romanian"
        RUSSIAN = "russian"
        SANSKRIT = "sanskrit"
        SERBIAN = "serbian"
        SHONA = "shona"
        SINDHI = "sindhi"
        SINHALA = "sinhala"
        SLOVAK = "slovak"
        SLOVENIAN = "slovenian"
        SOMALI = "somali"
        SPANISH = "spanish"
        SUNDANESE = "sundanese"
        SWAHILI = "swahili"
        SWEDISH = "swedish"
        TAGALOG = "tagalog"
        TAJIK = "tajik"
        TAMIL = "tamil"
        TATAR = "tatar"
        TELUGU = "telugu"
        THAI = "thai"
        TIBETAN = "tibetan"
        TURKISH = "turkish"
        TURKMEN = "turkmen"
        UKRAINIAN = "ukrainian"
        URDU = "urdu"
        UZBEK = "uzbek"
        VIETNAMESE = "vietnamese"
        WELSH = "welsh"
        YIDDISH = "yiddish"
        YORUBA = "yoruba"

    class Timestamp(str, Enum):
        CHUNK = "chunk"
        WORD = "word"

    @classmethod
    def get_basic_fields(cls):
        return ["task", "audio", "hf_token"]

    @classmethod
    def replicate_model_id(cls):
        return "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/4c5d637c-c441-4857-9791-7c11111b38b4/52ebbd85-50a7-4741-b398-30e31.webp",
            "created_at": "2023-11-13T13:28:53.689979Z",
            "description": "whisper-large-v3, incredibly fast, powered by Hugging Face Transformers! 🤗",
            "github_url": "https://github.com/chenxwh/insanely-fast-whisper",
            "license_url": "https://github.com/Vaibhavs10/insanely-fast-whisper/blob/main/LICENSE",
            "name": "incredibly-fast-whisper",
            "owner": "vaibhavs10",
            "is_official": False,
            "paper_url": None,
            "run_count": 15309490,
            "url": "https://replicate.com/vaibhavs10/incredibly-fast-whisper",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    task: Task = Field(
        description="Task to perform: transcribe or translate to another language.",
        default="transcribe",
    )
    audio: types.AudioRef = Field(default=types.AudioRef(), description="Audio file")
    hf_token: str | None = Field(
        title="Hf Token",
        description="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips. You need to agree to the terms in 'https://huggingface.co/pyannote/speaker-diarization-3.1' and 'https://huggingface.co/pyannote/segmentation-3.0' first.",
        default=None,
    )
    language: Language = Field(
        description="Language spoken in the audio, specify 'None' to perform language detection.",
        default="None",
    )
    timestamp: Timestamp = Field(
        description="Whisper supports both chunked as well as word level timestamps.",
        default="chunk",
    )
    batch_size: int = Field(
        title="Batch Size",
        description="Number of parallel batches you want to compute. Reduce if you face OOMs.",
        default=24,
    )
    diarise_audio: bool = Field(
        title="Diarise Audio",
        description="Use Pyannote.audio to diarise the audio clips. You will need to provide hf_token below too.",
        default=False,
    )


class GPT4o_Transcribe(ReplicateNode):
    """A speech-to-text model that uses GPT-4o to transcribe audio"""

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "language", "audio_file"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-4o-transcribe:1302d9240d7caa85f014a03f1e901b878ec6df3a37dac854699ddd6f560a4518"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/b1aa75b9-0353-401e-a38f-5e129bee9658/4o-transcribe.webp",
            "created_at": "2025-05-20T13:56:11.066606Z",
            "description": "A speech-to-text model that uses GPT-4o to transcribe audio",
            "github_url": None,
            "license_url": "https://openai.com/policies/",
            "name": "gpt-4o-transcribe",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 9029,
            "url": "https://replicate.com/openai/gpt-4o-transcribe",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    prompt: str | None = Field(
        title="Prompt",
        description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.",
        default=None,
    )
    language: str | None = Field(
        title="Language",
        description="The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.",
        default=None,
    )
    audio_file: str | None = Field(
        title="Audio File",
        description="The audio file to transcribe. Supported formats: mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm",
        default=None,
    )
    temperature: float = Field(
        title="Temperature",
        description="Sampling temperature between 0 and 1",
        ge=0.0,
        le=1.0,
        default=0,
    )
