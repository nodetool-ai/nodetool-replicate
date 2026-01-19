from pydantic import BaseModel, Field
import typing
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
from nodetool.nodes.replicate.replicate_node import ReplicateNode
from enum import Enum


class Llama3_8B(ReplicateNode):
    """Base version of Llama 3, an 8 billion parameter language model from Meta."""

    @classmethod
    def get_basic_fields(cls):
        return ["top_k", "top_p", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/meta-llama-3-8b:9a9e68fc8695f5847ce944a5cecf9967fd7c64d0fb8c8af1d5bdcc71f03c5e47"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/dd9ac11a-edda-4d33-b413-6a721c44dfb0/meta-logo.png",
            "created_at": "2024-04-17T18:04:26.049832Z",
            "description": "Base version of Llama 3, an 8 billion parameter language model from Meta.",
            "github_url": "https://github.com/meta-llama/llama3",
            "license_url": "https://github.com/meta-llama/llama3/blob/main/LICENSE",
            "name": "meta-llama-3-8b",
            "owner": "meta",
            "is_official": True,
            "paper_url": None,
            "run_count": 51165556,
            "url": "https://replicate.com/meta/meta-llama-3-8b",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_k: int = Field(
        title="Top K",
        description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
        default=50,
    )
    top_p: float = Field(
        title="Top P",
        description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
        default=0.9,
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=512,
    )
    min_tokens: int = Field(
        title="Min Tokens",
        description="The minimum number of tokens the model should generate as output.",
        default=0,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.6,
    )
    prompt_template: str = Field(
        title="Prompt Template",
        description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
        default="{prompt}",
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=1.15
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0.2
    )


class Llama3_8B_Instruct(ReplicateNode):
    """An 8 billion parameter language model from Meta, fine tuned for chat completions"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_k", "top_p", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/meta-llama-3-8b-instruct:5a6809ca6288247d06daf6365557e5e429063f32a21146b2a807c682652136b8"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/68b7dc1a-4767-4353-b066-212b0126b5de/meta-logo.png",
            "created_at": "2024-04-17T21:44:58.480057Z",
            "description": "An 8 billion parameter language model from Meta, fine tuned for chat completions",
            "github_url": "https://github.com/meta-llama/llama3",
            "license_url": "https://github.com/meta-llama/llama3/blob/main/LICENSE",
            "name": "meta-llama-3-8b-instruct",
            "owner": "meta",
            "is_official": True,
            "paper_url": None,
            "run_count": 396800564,
            "url": "https://replicate.com/meta/meta-llama-3-8b-instruct",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_k: int = Field(
        title="Top K",
        description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
        default=50,
    )
    top_p: float = Field(
        title="Top P",
        description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
        default=0.9,
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=512,
    )
    min_tokens: int = Field(
        title="Min Tokens",
        description="The minimum number of tokens the model should generate as output.",
        default=0,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.6,
    )
    prompt_template: str = Field(
        title="Prompt Template",
        description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
        default="{prompt}",
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=1.15
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0.2
    )


class Llama3_70B(ReplicateNode):
    """Base version of Llama 3, a 70 billion parameter language model from Meta."""

    @classmethod
    def get_basic_fields(cls):
        return ["top_k", "top_p", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/meta-llama-3-70b:83c5bdea9941e83be68480bd06ad792f3f295612a24e4678baed34083083a87f"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/8e044b4c-0b20-4717-83bd-a94d89fb0dbe/meta-logo.png",
            "created_at": "2024-04-17T18:05:18.044746Z",
            "description": "Base version of Llama 3, a 70 billion parameter language model from Meta.",
            "github_url": None,
            "license_url": "https://github.com/meta-llama/llama3/blob/main/LICENSE",
            "name": "meta-llama-3-70b",
            "owner": "meta",
            "is_official": True,
            "paper_url": None,
            "run_count": 853111,
            "url": "https://replicate.com/meta/meta-llama-3-70b",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_k: int = Field(
        title="Top K",
        description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
        default=50,
    )
    top_p: float = Field(
        title="Top P",
        description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
        default=0.9,
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=512,
    )
    min_tokens: int = Field(
        title="Min Tokens",
        description="The minimum number of tokens the model should generate as output.",
        default=0,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.6,
    )
    prompt_template: str = Field(
        title="Prompt Template",
        description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
        default="{prompt}",
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=1.15
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0.2
    )


class Llama3_8B_Instruct(ReplicateNode):
    """An 8 billion parameter language model from Meta, fine tuned for chat completions"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_k", "top_p", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/meta-llama-3-8b-instruct:5a6809ca6288247d06daf6365557e5e429063f32a21146b2a807c682652136b8"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/68b7dc1a-4767-4353-b066-212b0126b5de/meta-logo.png",
            "created_at": "2024-04-17T21:44:58.480057Z",
            "description": "An 8 billion parameter language model from Meta, fine tuned for chat completions",
            "github_url": "https://github.com/meta-llama/llama3",
            "license_url": "https://github.com/meta-llama/llama3/blob/main/LICENSE",
            "name": "meta-llama-3-8b-instruct",
            "owner": "meta",
            "is_official": True,
            "paper_url": None,
            "run_count": 396800564,
            "url": "https://replicate.com/meta/meta-llama-3-8b-instruct",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_k: int = Field(
        title="Top K",
        description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
        default=50,
    )
    top_p: float = Field(
        title="Top P",
        description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
        default=0.9,
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=512,
    )
    min_tokens: int = Field(
        title="Min Tokens",
        description="The minimum number of tokens the model should generate as output.",
        default=0,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.6,
    )
    prompt_template: str = Field(
        title="Prompt Template",
        description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
        default="{prompt}",
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=1.15
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0.2
    )


class Llama3_70B_Instruct(ReplicateNode):
    """A 70 billion parameter language model from Meta, fine tuned for chat completions"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_k", "top_p", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/meta-llama-3-70b-instruct:fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/3dcb020b-1fad-4101-84cf-88af9b20ac21/meta-logo.png",
            "created_at": "2024-04-17T21:44:13.482460Z",
            "description": "A 70 billion parameter language model from Meta, fine tuned for chat completions",
            "github_url": "https://github.com/meta-llama/llama3",
            "license_url": "https://github.com/meta-llama/llama3/blob/main/LICENSE",
            "name": "meta-llama-3-70b-instruct",
            "owner": "meta",
            "is_official": True,
            "paper_url": None,
            "run_count": 164561944,
            "url": "https://replicate.com/meta/meta-llama-3-70b-instruct",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_k: int = Field(
        title="Top K",
        description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
        default=50,
    )
    top_p: float = Field(
        title="Top P",
        description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
        default=0.9,
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=512,
    )
    min_tokens: int = Field(
        title="Min Tokens",
        description="The minimum number of tokens the model should generate as output.",
        default=0,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.6,
    )
    prompt_template: str = Field(
        title="Prompt Template",
        description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
        default="{prompt}",
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=1.15
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0.2
    )


class Llama3_1_405B_Instruct(ReplicateNode):
    """Meta's flagship 405 billion parameter language model, fine-tuned for chat completions"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_k", "top_p", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/meta-llama-3.1-405b-instruct:4ff591d23f09abef843c126a3c526bffb037a4e854e0af5af133a4d0f4243181"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/81ca001f-6a0a-4bef-b2f1-32466887df20/meta-logo.png",
            "created_at": "2024-07-22T20:40:30.648238Z",
            "description": "Meta's flagship 405 billion parameter language model, fine-tuned for chat completions",
            "github_url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1",
            "license_url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
            "name": "meta-llama-3.1-405b-instruct",
            "owner": "meta",
            "is_official": True,
            "paper_url": None,
            "run_count": 6996736,
            "url": "https://replicate.com/meta/meta-llama-3.1-405b-instruct",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_k: int = Field(
        title="Top K",
        description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
        default=50,
    )
    top_p: float = Field(
        title="Top P",
        description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
        default=0.9,
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=512,
    )
    min_tokens: int = Field(
        title="Min Tokens",
        description="The minimum number of tokens the model should generate as output.",
        default=0,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.6,
    )
    system_prompt: str = Field(
        title="System Prompt",
        description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Ignored for non-chat models.",
        default="You are a helpful assistant.",
    )
    stop_sequences: str = Field(
        title="Stop Sequences",
        description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
        default="",
    )
    prompt_template: str = Field(
        title="Prompt Template",
        description="A template to format the prompt with. If not provided, the default prompt template will be used.",
        default="",
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=0
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0
    )


class LlamaGuard_3_11B_Vision(ReplicateNode):
    """A Llama-3.2-11B pretrained model, fine-tuned for content safety classification"""

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/llama-guard-3-11b-vision:21d9a2579c40ab00a401cd487c6fab3b3053ef582eb5c9ca06920c1c76bdebf1"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/d7d7e254-bf5a-458f-9754-791a7db8ba44/replicate-prediction-m8j_JCyBlXR.webp",
            "created_at": "2024-12-23T20:39:23.769654Z",
            "description": "A Llama-3.2-11B pretrained model, fine-tuned for content safety classification",
            "github_url": "https://github.com/lucataco/cog-Llama-Guard-3-11B-Vision",
            "license_url": "https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt",
            "name": "llama-guard-3-11b-vision",
            "owner": "meta",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2312.06674",
            "run_count": 1506,
            "url": "https://replicate.com/meta/llama-guard-3-11b-vision",
            "visibility": "public",
            "weights_url": "https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision",
        }

    @classmethod
    def return_type(cls):
        return str

    image: str | None = Field(
        title="Image", description="Image to moderate", default=None
    )
    prompt: str = Field(
        title="Prompt",
        description="User message to moderate",
        default="Which one should I buy?",
    )


class LlamaGuard_3_8B(ReplicateNode):
    """A Llama-3.1-8B pretrained model, fine-tuned for content safety classification"""

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "assistant"]

    @classmethod
    def replicate_model_id(cls):
        return "meta/llama-guard-3-8b:146d1220d447cdcc639bc17c5f6137416042abee6ae153a2615e6ef5749205c8"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/b59edf5b-6571-4673-8cd8-87488501f5b7/replicate-prediction-d2c_9x54OXs.webp",
            "created_at": "2024-12-21T00:37:41.039448Z",
            "description": "A Llama-3.1-8B pretrained model, fine-tuned for content safety classification",
            "github_url": "https://github.com/lucataco/cog-Llama-Guard-3-8B",
            "license_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/LICENSE",
            "name": "llama-guard-3-8b",
            "owner": "meta",
            "is_official": False,
            "paper_url": "https://arxiv.org/abs/2407.21783",
            "run_count": 360040,
            "url": "https://replicate.com/meta/llama-guard-3-8b",
            "visibility": "public",
            "weights_url": "https://huggingface.co/meta-llama/Llama-Guard-3-8B",
        }

    @classmethod
    def return_type(cls):
        return str

    prompt: str = Field(
        title="Prompt",
        description="User message to moderate",
        default="I forgot how to kill a process in Linux, can you help?",
    )
    assistant: str | None = Field(
        title="Assistant", description="Assistant response to classify", default=None
    )


class Snowflake_Arctic_Instruct(ReplicateNode):
    """An efficient, intelligent, and truly open-source language model"""

    @classmethod
    def get_basic_fields(cls):
        return ["name", "name_file"]

    @classmethod
    def replicate_model_id(cls):
        return "snowflake/snowflake-arctic-instruct:081f548e9a59c93b8355abe28ca52680c8305bc8f4a186a3de62ea41b25db8dd"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/793e32b4-913c-4036-a847-4afb38e42fc1/Snowflake_Arctic_Opengraph_120.png",
            "created_at": "2024-04-24T00:08:29.300675Z",
            "description": "An efficient, intelligent, and truly open-source language model",
            "github_url": "https://github.com/Snowflake-Labs/snowflake-arctic",
            "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
            "name": "snowflake-arctic-instruct",
            "owner": "snowflake",
            "is_official": True,
            "paper_url": None,
            "run_count": 1996678,
            "url": "https://replicate.com/snowflake/snowflake-arctic-instruct",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    name: str | None = Field(title="Name", default=None)
    name_file: str | None = Field(title="Name File", default=None)


class Claude_3_7_Sonnet(ReplicateNode):
    """The most intelligent Claude model and the first hybrid reasoning model on the market (claude-3-7-sonnet-20250219)"""

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "max_tokens"]

    @classmethod
    def replicate_model_id(cls):
        return "anthropic/claude-3.7-sonnet:81a891bd00c339f3565bda15b255b372eb8bf6c669fe996b66eea5d677454a46"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/56aed331-fb30-4e82-9708-b63b2fa90699/claude-3.7-logo.webp",
            "created_at": "2025-02-25T15:21:57.270034Z",
            "description": "The most intelligent Claude model and the first hybrid reasoning model on the market (claude-3-7-sonnet-20250219)",
            "github_url": None,
            "license_url": "https://www.anthropic.com/legal/consumer-terms",
            "name": "claude-3.7-sonnet",
            "owner": "anthropic",
            "is_official": True,
            "paper_url": None,
            "run_count": 3628064,
            "url": "https://replicate.com/anthropic/claude-3.7-sonnet",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    image: str | None = Field(
        title="Image",
        description="Optional input image. Images are priced as (width px * height px)/750 input tokens",
        default=None,
    )
    prompt: str | None = Field(title="Prompt", description="Input prompt", default=None)
    max_tokens: int = Field(
        title="Max Tokens",
        description="Maximum number of output tokens",
        ge=1024.0,
        le=64000.0,
        default=8192,
    )
    system_prompt: str = Field(
        title="System Prompt", description="System prompt", default=""
    )
    max_image_resolution: float = Field(
        title="Max Image Resolution",
        description="Maximum image resolution in megapixels. Scales down image before sending it to Claude, to save time and money.",
        ge=0.001,
        le=2.0,
        default=0.5,
    )


class Deepseek_R1(ReplicateNode):
    """A reasoning model trained with reinforcement learning, on par with OpenAI o1"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_p", "prompt", "max_tokens"]

    @classmethod
    def replicate_model_id(cls):
        return "deepseek-ai/deepseek-r1:fec99f91f58ce5302af6d4cfd8638846925f47e0cd39a4554637806e8379766d"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/302182ab-af74-4963-97f2-6121a80c61d7/deepseek-r1-cover.webp",
            "created_at": "2025-01-29T02:46:11.806773Z",
            "description": "A reasoning model trained with reinforcement learning, on par with OpenAI o1",
            "github_url": "https://github.com/deepseek-ai/DeepSeek-R1",
            "license_url": "https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE",
            "name": "deepseek-r1",
            "owner": "deepseek-ai",
            "is_official": True,
            "paper_url": None,
            "run_count": 2168449,
            "url": "https://replicate.com/deepseek-ai/deepseek-r1",
            "visibility": "public",
            "weights_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1",
        }

    @classmethod
    def return_type(cls):
        return str

    top_p: float = Field(
        title="Top P", description="Top-p (nucleus) sampling", default=1
    )
    prompt: str = Field(title="Prompt", description="Prompt", default="")
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=2048,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.1,
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=0
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0
    )


class GPT_5_Structured(ReplicateNode):
    """GPT-5 with support for structured outputs, web search and custom tools"""

    class Model(str, Enum):
        GPT_5 = "gpt-5"
        GPT_5_MINI = "gpt-5-mini"
        GPT_5_NANO = "gpt-5-nano"

    class Verbosity(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Reasoning_effort(str, Enum):
        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @classmethod
    def get_basic_fields(cls):
        return ["model", "tools", "prompt"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-5-structured:4e32e66191d7bdfcabf4398892aa77f0352964520ae6bc545ed60add03090d91"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/60ebe2fc-dcd1-4185-90b2-bf0cc34c9688/replicate-prediction-qhzgvzbn7.jpg",
            "created_at": "2025-08-14T15:32:22.469348Z",
            "description": "GPT-5 with support for structured outputs, web search and custom tools",
            "github_url": None,
            "license_url": None,
            "name": "gpt-5-structured",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 355541,
            "url": "https://replicate.com/openai/gpt-5-structured",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    model: Model = Field(description="GPT-5 model to use.", default="gpt-5")
    tools: list = Field(
        title="Tools",
        description="Tools to make available to the model. Should be a JSON object containing a list of tool definitions.",
        default=[],
    )
    prompt: str | None = Field(
        title="Prompt",
        description="A simple text input to the model, equivalent to a text input with the user role. Ignored if input_item_list is provided.",
        default=None,
    )
    verbosity: Verbosity = Field(
        description="Constrains the verbosity of the model's response. Lower values will result in more concise responses, while higher values will result in more verbose responses. Currently supported values are low, medium, and high. GPT-5 supports this parameter to help control whether answers are short and to the point or long and comprehensive.",
        default="medium",
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    json_schema: dict = Field(
        title="Json Schema",
        description="A JSON schema that the response must conform to. For simple data structures we recommend using `simple_text_format_schema` which will be converted to a JSON schema for you.",
        default={},
    )
    instructions: str | None = Field(
        title="Instructions",
        description="A system (or developer) message inserted into the model's context. When using along with previous_response_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.",
        default=None,
    )
    simple_schema: list = Field(
        title="Simple Schema",
        description="Create a JSON schema for the output to conform to. The schema will be created from a simple list of field specifications. Strings: 'thing' (defaults to string), 'thing:str', 'thing:string'. Booleans: 'is_a_thing:bool' or 'is_a_thing:boolean'. Numbers: 'count:number', 'count:int'. Lists: 'things:list' (defaults to list of strings), 'things:list:str', 'number_things:list:number', etc. Nested objects are not supported, use `json_schema` instead.",
        default=[],
    )
    input_item_list: list = Field(
        title="Input Item List",
        description="A list of one or many input items to the model, containing different content types. This parameter corresponds with the `input` OpenAI API parameter. For more details see: https://platform.openai.com/docs/api-reference/responses/create#responses_create-input. Similar to the `messages` parameter, but with more flexibility in the content types.",
        default=[],
    )
    reasoning_effort: Reasoning_effort = Field(
        description="Constrains effort on reasoning for GPT-5 models. Currently supported values are minimal, low, medium, and high. The minimal value gets answers back faster without extensive reasoning first. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default="minimal",
    )
    enable_web_search: bool = Field(
        title="Enable Web Search",
        description="Allow GPT-5 to use web search for the response.",
        default=False,
    )
    max_output_tokens: int | None = Field(
        title="Max Output Tokens",
        description="Maximum number of completion tokens to generate. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default=None,
    )
    previous_response_id: str | None = Field(
        title="Previous Response Id",
        description="The ID of a previous response to continue from.",
        default=None,
    )


class GPT_5(ReplicateNode):
    """OpenAI's new model excelling at coding, writing, and reasoning."""

    class Verbosity(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Reasoning_effort(str, Enum):
        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "messages", "verbosity"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-5:e66760af5e83560f7d8d71e3420dce9362cea9ac41a492e5ba41e40405b62c55"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/93d32638-644c-4926-8d07-ad01eec112fb/gpt-5-sm.jpg",
            "created_at": "2025-08-07T01:46:29.933808Z",
            "description": "OpenAI's new model excelling at coding, writing, and reasoning.",
            "github_url": None,
            "license_url": None,
            "name": "gpt-5",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 981927,
            "url": "https://replicate.com/openai/gpt-5",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    prompt: str | None = Field(
        title="Prompt",
        description="The prompt to send to the model. Do not use if using messages.",
        default=None,
    )
    messages: list = Field(
        title="Messages",
        description='A JSON string representing a list of messages. For example: [{"role": "user", "content": "Hello, how are you?"}]. If provided, prompt and system_prompt are ignored.',
        default=[],
    )
    verbosity: Verbosity = Field(
        description="Constrains the verbosity of the model's response. Lower values will result in more concise responses, while higher values will result in more verbose responses. Currently supported values are low, medium, and high. GPT-5 supports this parameter to help control whether answers are short and to the point or long and comprehensive.",
        default="medium",
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    system_prompt: str | None = Field(
        title="System Prompt",
        description="System prompt to set the assistant's behavior",
        default=None,
    )
    reasoning_effort: Reasoning_effort = Field(
        description="Constrains effort on reasoning for GPT-5 models. Currently supported values are minimal, low, medium, and high. The minimal value gets answers back faster without extensive reasoning first. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default="minimal",
    )
    max_completion_tokens: int | None = Field(
        title="Max Completion Tokens",
        description="Maximum number of completion tokens to generate. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default=None,
    )


class GPT_5_Mini(ReplicateNode):
    """Faster version of OpenAI's flagship GPT-5 model"""

    class Verbosity(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Reasoning_effort(str, Enum):
        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "messages", "verbosity"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-5-mini:ea9e381ae5a1370344caf7103b2efd367cc37f30e42b7acc6c3bcb2b140182e1"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/518903fa-a7de-4876-a79e-aac7fdeae577/Screenshot_2025-08-07_at_1.04..png",
            "created_at": "2025-08-07T01:46:38.852666Z",
            "description": "Faster version of OpenAI's flagship GPT-5 model",
            "github_url": None,
            "license_url": None,
            "name": "gpt-5-mini",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 736565,
            "url": "https://replicate.com/openai/gpt-5-mini",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    prompt: str | None = Field(
        title="Prompt",
        description="The prompt to send to the model. Do not use if using messages.",
        default=None,
    )
    messages: list = Field(
        title="Messages",
        description='A JSON string representing a list of messages. For example: [{"role": "user", "content": "Hello, how are you?"}]. If provided, prompt and system_prompt are ignored.',
        default=[],
    )
    verbosity: Verbosity = Field(
        description="Constrains the verbosity of the model's response. Lower values will result in more concise responses, while higher values will result in more verbose responses. Currently supported values are low, medium, and high. GPT-5 supports this parameter to help control whether answers are short and to the point or long and comprehensive.",
        default="medium",
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    system_prompt: str | None = Field(
        title="System Prompt",
        description="System prompt to set the assistant's behavior",
        default=None,
    )
    reasoning_effort: Reasoning_effort = Field(
        description="Constrains effort on reasoning for GPT-5 models. Currently supported values are minimal, low, medium, and high. The minimal value gets answers back faster without extensive reasoning first. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default="minimal",
    )
    max_completion_tokens: int | None = Field(
        title="Max Completion Tokens",
        description="Maximum number of completion tokens to generate. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default=None,
    )


class GPT_5_Nano(ReplicateNode):
    """Fastest, most cost-effective GPT-5 model from OpenAI"""

    class Verbosity(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class Reasoning_effort(str, Enum):
        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "messages", "verbosity"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-5-nano:7ac1cc959145e65a06f2931cc378602226a85b286c00431baa32905550501923"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_cover_image/fbb1068e-ae55-4d4f-9ee8-3e3da859f69f/Screenshot_2025-08-07_at_1.04.57P.png",
            "created_at": "2025-08-07T01:46:49.288485Z",
            "description": "Fastest, most cost-effective GPT-5 model from OpenAI",
            "github_url": None,
            "license_url": None,
            "name": "gpt-5-nano",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 3834399,
            "url": "https://replicate.com/openai/gpt-5-nano",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    prompt: str | None = Field(
        title="Prompt",
        description="The prompt to send to the model. Do not use if using messages.",
        default=None,
    )
    messages: list = Field(
        title="Messages",
        description='A JSON string representing a list of messages. For example: [{"role": "user", "content": "Hello, how are you?"}]. If provided, prompt and system_prompt are ignored.',
        default=[],
    )
    verbosity: Verbosity = Field(
        description="Constrains the verbosity of the model's response. Lower values will result in more concise responses, while higher values will result in more verbose responses. Currently supported values are low, medium, and high. GPT-5 supports this parameter to help control whether answers are short and to the point or long and comprehensive.",
        default="medium",
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    system_prompt: str | None = Field(
        title="System Prompt",
        description="System prompt to set the assistant's behavior",
        default=None,
    )
    reasoning_effort: Reasoning_effort = Field(
        description="Constrains effort on reasoning for GPT-5 models. Currently supported values are minimal, low, medium, and high. The minimal value gets answers back faster without extensive reasoning first. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default="minimal",
    )
    max_completion_tokens: int | None = Field(
        title="Max Completion Tokens",
        description="Maximum number of completion tokens to generate. For higher reasoning efforts you may need to increase your max_completion_tokens to avoid empty responses (where all the tokens are used on reasoning).",
        default=None,
    )


class GPT_4_1(ReplicateNode):
    """OpenAI's Flagship GPT model for complex tasks."""

    @classmethod
    def get_basic_fields(cls):
        return ["top_p", "prompt", "messages"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-4.1:12500eb28df96f9b9a30ae89f02652414f9d692ad391cea2c326015aa719e1a2"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8dbaab0b-d3e7-4bae-8772-d7e1f879d537/gpt-4.1.webp",
            "created_at": "2025-05-01T04:55:34.442201Z",
            "description": "OpenAI's Flagship GPT model for complex tasks.",
            "github_url": None,
            "license_url": "https://openai.com/policies/",
            "name": "gpt-4.1",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 273884,
            "url": "https://replicate.com/openai/gpt-4.1",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_p: float = Field(
        title="Top P",
        description="Nucleus sampling parameter - the model considers the results of the tokens with top_p probability mass. (0.1 means only the tokens comprising the top 10% probability mass are considered.)",
        ge=0.0,
        le=1.0,
        default=1,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="The prompt to send to the model. Do not use if using messages.",
        default=None,
    )
    messages: list = Field(
        title="Messages",
        description='A JSON string representing a list of messages. For example: [{"role": "user", "content": "Hello, how are you?"}]. If provided, prompt and system_prompt are ignored.',
        default=[],
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    temperature: float = Field(
        title="Temperature",
        description="Sampling temperature between 0 and 2",
        ge=0.0,
        le=2.0,
        default=1,
    )
    system_prompt: str | None = Field(
        title="System Prompt",
        description="System prompt to set the assistant's behavior",
        default=None,
    )
    presence_penalty: float = Field(
        title="Presence Penalty",
        description="Presence penalty parameter - positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
        ge=-2.0,
        le=2.0,
        default=0,
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty",
        description="Frequency penalty parameter - positive values penalize the repetition of tokens.",
        ge=-2.0,
        le=2.0,
        default=0,
    )
    max_completion_tokens: int = Field(
        title="Max Completion Tokens",
        description="Maximum number of completion tokens to generate",
        default=4096,
    )


class GPT_4_1_Mini(ReplicateNode):
    """Fast, affordable version of GPT-4.1"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_p", "prompt", "messages"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-4.1-mini:029d04e27c11b0898c24e0d8ae12c93dee8edbf3ff59a839e8a4a896691b733a"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/6f1609f1-ba45-4513-bfd6-7dfcb95efad8/Screenshot_2025-05-01_at_12.03.png",
            "created_at": "2025-05-01T06:57:43.806670Z",
            "description": "Fast, affordable version of GPT-4.1",
            "github_url": None,
            "license_url": "https://openai.com/policies/",
            "name": "gpt-4.1-mini",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 1395191,
            "url": "https://replicate.com/openai/gpt-4.1-mini",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_p: float = Field(
        title="Top P",
        description="Nucleus sampling parameter - the model considers the results of the tokens with top_p probability mass. (0.1 means only the tokens comprising the top 10% probability mass are considered.)",
        ge=0.0,
        le=1.0,
        default=1,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="The prompt to send to the model. Do not use if using messages.",
        default=None,
    )
    messages: list = Field(
        title="Messages",
        description='A JSON string representing a list of messages. For example: [{"role": "user", "content": "Hello, how are you?"}]. If provided, prompt and system_prompt are ignored.',
        default=[],
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    temperature: float = Field(
        title="Temperature",
        description="Sampling temperature between 0 and 2",
        ge=0.0,
        le=2.0,
        default=1,
    )
    system_prompt: str | None = Field(
        title="System Prompt",
        description="System prompt to set the assistant's behavior",
        default=None,
    )
    presence_penalty: float = Field(
        title="Presence Penalty",
        description="Presence penalty parameter - positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
        ge=-2.0,
        le=2.0,
        default=0,
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty",
        description="Frequency penalty parameter - positive values penalize the repetition of tokens.",
        ge=-2.0,
        le=2.0,
        default=0,
    )
    max_completion_tokens: int = Field(
        title="Max Completion Tokens",
        description="Maximum number of completion tokens to generate",
        default=4096,
    )


class GPT_4_1_Nano(ReplicateNode):
    """Fastest, most cost-effective GPT-4.1 model from OpenAI"""

    @classmethod
    def get_basic_fields(cls):
        return ["top_p", "prompt", "messages"]

    @classmethod
    def replicate_model_id(cls):
        return "openai/gpt-4.1-nano:d16a8857696f4bb42006b2f3799b590111a7365280f9b7f0c898f8e2ee3b8ea2"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/f2c12cca-5859-407a-9189-0509526e4757/Screenshot_2025-05-01_at_12.29.png",
            "created_at": "2025-05-01T07:26:10.557033Z",
            "description": "Fastest, most cost-effective GPT-4.1 model from OpenAI",
            "github_url": None,
            "license_url": "https://openai.com/policies/",
            "name": "gpt-4.1-nano",
            "owner": "openai",
            "is_official": True,
            "paper_url": None,
            "run_count": 1024497,
            "url": "https://replicate.com/openai/gpt-4.1-nano",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_p: float = Field(
        title="Top P",
        description="Nucleus sampling parameter - the model considers the results of the tokens with top_p probability mass. (0.1 means only the tokens comprising the top 10% probability mass are considered.)",
        ge=0.0,
        le=1.0,
        default=1,
    )
    prompt: str | None = Field(
        title="Prompt",
        description="The prompt to send to the model. Do not use if using messages.",
        default=None,
    )
    messages: list = Field(
        title="Messages",
        description='A JSON string representing a list of messages. For example: [{"role": "user", "content": "Hello, how are you?"}]. If provided, prompt and system_prompt are ignored.',
        default=[],
    )
    image_input: list = Field(
        title="Image Input",
        description="List of images to send to the model",
        default=[],
    )
    temperature: float = Field(
        title="Temperature",
        description="Sampling temperature between 0 and 2",
        ge=0.0,
        le=2.0,
        default=1,
    )
    system_prompt: str | None = Field(
        title="System Prompt",
        description="System prompt to set the assistant's behavior",
        default=None,
    )
    presence_penalty: float = Field(
        title="Presence Penalty",
        description="Presence penalty parameter - positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
        ge=-2.0,
        le=2.0,
        default=0,
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty",
        description="Frequency penalty parameter - positive values penalize the repetition of tokens.",
        ge=-2.0,
        le=2.0,
        default=0,
    )
    max_completion_tokens: int = Field(
        title="Max Completion Tokens",
        description="Maximum number of completion tokens to generate",
        default=4096,
    )


class Deepseek_V3_1(ReplicateNode):
    """Latest hybrid thinking model from Deepseek"""

    class Thinking(str, Enum):
        MEDIUM = "medium"
        NONE = "None"

    @classmethod
    def get_basic_fields(cls):
        return ["top_p", "prompt", "thinking"]

    @classmethod
    def replicate_model_id(cls):
        return "deepseek-ai/deepseek-v3.1:279f6b0991efaba468503a13d7726829cd76ea12076af7521a41578c2f19f581"

    @classmethod
    def get_hardware(cls):
        return "None"

    @classmethod
    def get_model_info(cls):
        return {
            "cover_image_url": "https://tjzk.replicate.delivery/models_models_featured_image/8d444a48-2b5e-46ba-80dd-bfbdae41ffb8/tmp2xmj7b2x.jpg",
            "created_at": "2025-08-25T18:41:12.219732Z",
            "description": "Latest hybrid thinking model from Deepseek",
            "github_url": None,
            "license_url": None,
            "name": "deepseek-v3.1",
            "owner": "deepseek-ai",
            "is_official": True,
            "paper_url": None,
            "run_count": 252670,
            "url": "https://replicate.com/deepseek-ai/deepseek-v3.1",
            "visibility": "public",
            "weights_url": None,
        }

    @classmethod
    def return_type(cls):
        return str

    top_p: float = Field(
        title="Top P", description="Top-p (nucleus) sampling", default=1
    )
    prompt: str = Field(
        title="Prompt",
        description="Prompt",
        default="Why are you better than Deepseek v3?",
    )
    thinking: Thinking = Field(
        description="Reasoning effort level for DeepSeek models. Use 'medium' for enhanced reasoning or leave as None for default behavior.",
        default="None",
    )
    max_tokens: int = Field(
        title="Max Tokens",
        description="The maximum number of tokens the model should generate as output.",
        default=1024,
    )
    temperature: float = Field(
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
        default=0.1,
    )
    presence_penalty: float = Field(
        title="Presence Penalty", description="Presence penalty", default=0
    )
    frequency_penalty: float = Field(
        title="Frequency Penalty", description="Frequency penalty", default=0
    )
