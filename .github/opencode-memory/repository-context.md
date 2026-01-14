# Repository Context

## Overview

`nodetool-replicate` provides Replicate integration nodes for the [NodeTool](https://github.com/nodetool-ai/nodetool) project. It depends on [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Directory Structure

```
src/nodetool/
├── nodes/replicate/           # All Replicate node implementations
│   ├── gencode.py             # Node definition list and code generation runner
│   ├── code_generation.py     # Code generation logic
│   ├── replicate_node.py      # Base class for all Replicate nodes
│   ├── audio/                 # Audio processing nodes
│   │   ├── generate.py        # Audio generation (TTS, music)
│   │   ├── transcribe.py      # Speech-to-text
│   │   ├── enhance.py         # Audio enhancement
│   │   └── separate.py        # Source separation (Demucs)
│   ├── image/                 # Image processing nodes
│   │   ├── generate.py        # Image generation (SDXL, Flux, etc.)
│   │   ├── analyze.py         # Image analysis (CLIP, BLIP)
│   │   ├── face.py            # Face manipulation
│   │   ├── enhance.py         # Image enhancement
│   │   ├── upscale.py         # Image upscaling
│   │   ├── ocr.py             # OCR nodes
│   │   └── process.py         # General image processing
│   ├── video/                 # Video processing nodes
│   │   ├── generate.py        # Video generation
│   │   └── enhance.py         # Video enhancement
│   └── text/                  # Text/LLM nodes
│       └── generate.py        # Text generation (Llama, GPT, etc.)
├── dsl/                       # Generated DSL code
└── package_metadata/          # Generated package metadata
```

## Key Files

### `gencode.py`
Contains the `replicate_nodes` list - a Python list of dictionaries defining all Replicate nodes to generate. Each dictionary specifies:
- `node_name`: Class name for the node
- `model_id`: Replicate model identifier (e.g., "black-forest-labs/flux-schnell")
- `namespace`: Output location (e.g., "image.generate")
- `return_type`: Output type (ImageRef, AudioRef, VideoRef, str, etc.)
- `overrides`: Field type overrides for input parameters (e.g., `{"image": ImageRef}`)

### `code_generation.py`
Code generation logic that:
1. Fetches model metadata from Replicate API
2. Parses OpenAPI schema from model
3. Generates Python source code for each node

### `replicate_node.py`
Base class `ReplicateNode` that all nodes inherit from. Handles:
- Running predictions on Replicate
- Converting outputs to appropriate types
- Model metadata registration

## Namespaces and Return Types

| Namespace | Purpose | Common Return Type |
|-----------|---------|-------------------|
| `image.generate` | Image generation models | `ImageRef` |
| `image.analyze` | Image analysis/captioning | `str` |
| `image.face` | Face manipulation | `ImageRef` |
| `image.enhance` | Image enhancement | `ImageRef` |
| `image.upscale` | Image upscaling | `ImageRef` |
| `image.process` | General image processing | `ImageRef` |
| `image.ocr` | Optical character recognition | `str` |
| `video.generate` | Video generation | `VideoRef` |
| `video.enhance` | Video enhancement | `VideoRef` |
| `audio.generate` | Audio/music generation | `AudioRef` |
| `audio.transcribe` | Speech-to-text | `str` |
| `audio.enhance` | Audio enhancement | `AudioRef` |
| `audio.separate` | Audio source separation | `dict` of `AudioRef` |
| `text.generate` | LLM text generation | `str` |

## Commands

```bash
# Generate nodes from gencode.py
REPLICATE_API_TOKEN=<token> python -m nodetool.nodes.replicate.gencode

# Generate for specific namespace only
REPLICATE_API_TOKEN=<token> python -m nodetool.nodes.replicate.gencode --namespace image.generate

# Generate metadata and DSL
nodetool package scan
nodetool codegen

# Linting and formatting
ruff check .
black --check .
```

## Code Style

- Python 3.11+ syntax
- Nodes inherit from `BaseNode` (via `ReplicateNode`)
- Node attributes use `pydantic.Field`
- Each node has a docstring from the model description
- Include `get_basic_fields` class method listing key fields
