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
            "paper_url": None,
            "run_count": 50806621,
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
            "paper_url": None,
            "run_count": 325600516,
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
            "paper_url": None,
            "run_count": 820484,
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
            "paper_url": None,
            "run_count": 325600516,
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
            "paper_url": None,
            "run_count": 145312032,
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
        return "meta/meta-llama-3.1-405b-instruct:e6cb7fc3ed90eae2c879c48deda8f49152391ad66349fe7694be24089c29f71c"

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
            "paper_url": None,
            "run_count": 4954794,
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
    stop_sequences: str | None = Field(
        title="Stop Sequences",
        description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
        default=None,
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
            "paper_url": "https://arxiv.org/abs/2312.06674",
            "run_count": 50,
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
            "paper_url": "https://arxiv.org/abs/2407.21783",
            "run_count": 30683,
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
            "paper_url": None,
            "run_count": 1971050,
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
            "paper_url": None,
            "run_count": 19522,
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
        return "deepseek-ai/deepseek-r1:d0426501ce0499770ac3d3494e9f3c7c03f9b10e569212f6148a4fd337ec3ece"

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
            "paper_url": None,
            "run_count": 498955,
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
        default=20480,
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
