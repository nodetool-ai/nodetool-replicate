# Replicate Node Code Generation Guide

This document explains how to add new Replicate models and generate node classes.

## Overview

The `src/nodetool/nodes/replicate/gencode.py` file contains definitions for all Replicate models that are exposed as nodes in nodetool. The code generation process:

1. Reads model definitions from `gencode.py`
2. Fetches OpenAPI schemas from Replicate's API for each model
3. Generates Python node classes with proper types and fields
4. Creates namespace-specific files (e.g., `image/generate.py`, `video/generate.py`)

## Adding New Models

### 1. Find Missing Models

Explore https://replicate.com/explore to find popular models that aren't yet in the repository.

### 2. Add Model Definition

Edit `src/nodetool/nodes/replicate/gencode.py` and add a new entry to the `replicate_nodes` list:

```python
{
    "model_id": "owner/model-name",      # Replicate model ID
    "node_name": "ModelName",             # Python class name
    "namespace": "image.generate",        # Category namespace
    "return_type": ImageRef,              # Return type (ImageRef, VideoRef, AudioRef, str)
    "overrides": {"image": ImageRef},     # Optional: field type overrides
}
```

### Available Namespaces

- **Image**: `image.generate`, `image.process`, `image.analyze`, `image.enhance`, `image.upscale`, `image.face`, `image.ocr`, `image.3d`
- **Video**: `video.generate`, `video.enhance`
- **Audio**: `audio.generate`, `audio.transcribe`, `audio.separate`, `audio.enhance`
- **Text**: `text.generate`

### Return Types

Import from `nodetool.metadata.types`:
- `ImageRef` - For image outputs
- `VideoRef` - For video outputs
- `AudioRef` - For audio outputs
- `SVGRef` - For SVG outputs
- `str` - For text outputs
- `dict` - For complex JSON outputs

### Field Overrides

Some models have generic field names like "image" or "audio" that need type overrides to properly handle file references:

```python
"overrides": {
    "image": ImageRef,           # Single image input
    "mask": ImageRef,            # Mask image input
    "video": VideoRef,           # Video input
    "audio": AudioRef,           # Audio input
}
```

## Running Code Generation

### Prerequisites

1. Install dependencies:
```bash
pip install -e .
pip install -r requirements-dev.txt
```

2. Set up your Replicate API token:
```bash
export REPLICATE_API_TOKEN="your-token-here"
```

Get your token from https://replicate.com/account/api-tokens

### Generate All Nodes

Run the gencode script to generate all node classes:

```bash
cd src/nodetool/nodes/replicate
python gencode.py
```

This will:
- Fetch model schemas from Replicate API
- Generate node classes in namespace-specific files
- Create proper Python types and Field definitions
- Add docstrings from model descriptions

### Generate Specific Namespace

To regenerate only a specific namespace:

```bash
python gencode.py --namespace image.generate
```

### After Generation

1. Run package scan to update metadata:
```bash
nodetool package scan
```

2. Generate DSL code:
```bash
nodetool codegen
```

3. Run linting and formatting:
```bash
ruff check .
black .
```

4. Run tests:
```bash
pytest -q
```

## Example: Adding a New Model

Let's add a new image generation model:

1. **Add definition to gencode.py:**
```python
{
    "model_id": "stability-ai/stable-cascade",
    "node_name": "StableCascade",
    "namespace": "image.generate",
    "return_type": ImageRef,
    "overrides": {"image": ImageRef},
},
```

2. **Run code generation:**
```bash
export REPLICATE_API_TOKEN="your-token"
cd src/nodetool/nodes/replicate
python gencode.py --namespace image.generate
```

3. **Generate metadata and DSL:**
```bash
nodetool package scan
nodetool codegen
```

4. **Verify the changes:**
```bash
ruff check .
black --check .
pytest -q
```

## Troubleshooting

### API Rate Limits

If you hit rate limits, the script includes exponential backoff retry logic. You may need to wait between runs.

### Missing Fields

If a field is missing type information, add it to the `overrides` dict in the model definition.

### Invalid Schemas

Some models may have invalid or incomplete OpenAPI schemas. Check the Replicate model page to ensure the model is properly configured.

## Recent Changes

- Added 23 new model definitions including:
  - 3D generation models (InstantMesh, SplatterImage, Flux_3D)
  - Advanced image models (InstructPix2Pix, ControlNet_Scribble)
  - Video processing (StableVideoDiffusion, VideoToVideo)
  - Code generation (CodeLlama models)
  - Face manipulation (FaceToAll, LivePortrait)

Total nodes: 177 (as of latest update)
