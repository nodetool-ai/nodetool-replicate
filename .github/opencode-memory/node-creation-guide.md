# Node Creation Guide

This guide explains how to add new Replicate models to the nodetool-replicate repository.

## Step 1: Discover New Models on Replicate

Visit https://replicate.com/explore to find models that are:
- **Popular** - High run counts indicate user demand
- **Official/Featured** - From Replicate's curated collections
- **Well-maintained** - Recent updates and good documentation
- **Not yet covered** - Check `gencode.py` for existing `model_id` values

## Step 2: Determine Model Information

For each model, gather:

1. **Model ID**: Found in the URL, e.g., `black-forest-labs/flux-schnell`
2. **Category**: What does the model do?
   - Text-to-image → `image.generate`
   - Image analysis/captioning → `image.analyze`
   - Face manipulation → `image.face`
   - Image enhancement → `image.enhance`
   - Image upscaling → `image.upscale`
   - General image processing → `image.process`
   - OCR → `image.ocr`
   - Video generation → `video.generate`
   - Video enhancement → `video.enhance`
   - Audio/music generation → `audio.generate`
   - Speech-to-text → `audio.transcribe`
   - Audio enhancement → `audio.enhance`
   - Source separation → `audio.separate`
   - LLM/text generation → `text.generate`
3. **Return Type**: What does the model output?
   - Images → `ImageRef`
   - Videos → `VideoRef`
   - Audio → `AudioRef`
   - Text → `str`
   - Structured data → `dict` or specific type
4. **Input Overrides**: Which parameters are file inputs?
   - Look at the model's API inputs
   - Parameters accepting images should use `ImageRef`
   - Parameters accepting audio should use `AudioRef`
   - Parameters accepting video should use `VideoRef`

## Step 3: Create Node Name

Convert the model ID to a valid Python class name:

```
Model ID               → Node Name
black-forest-labs/flux-schnell → Flux_Schnell
stability-ai/stable-diffusion  → StableDiffusion
meta/meta-llama-3-70b          → Llama3_70B
```

Rules:
- Use PascalCase (capitalize each word)
- Replace hyphens with underscores
- Remove organization prefix redundancy
- Keep it recognizable

## Step 4: Add Entry to gencode.py

Add a dictionary to the `replicate_nodes` list in `src/nodetool/nodes/replicate/gencode.py`:

```python
{
    "model_id": "owner/model-name",
    "node_name": "ModelName",
    "namespace": "category.subcategory",
    "return_type": ImageRef,  # or VideoRef, AudioRef, str, etc.
    "overrides": {"input_image": ImageRef},  # Optional: type overrides
},
```

### Examples

**Image Generation Model:**
```python
{
    "model_id": "black-forest-labs/flux-schnell",
    "node_name": "Flux_Schnell",
    "namespace": "image.generate",
    "return_type": ImageRef,
},
```

**Image-to-Image Model:**
```python
{
    "model_id": "fofr/style-transfer",
    "node_name": "StyleTransfer",
    "namespace": "image.generate",
    "return_type": ImageRef,
    "overrides": {"structure_image": ImageRef, "style_image": ImageRef},
},
```

**Video Generation Model:**
```python
{
    "model_id": "luma/ray",
    "node_name": "Ray",
    "namespace": "video.generate",
    "return_type": VideoRef,
    "overrides": {"start_image_url": ImageRef, "end_image_url": ImageRef},
},
```

**Audio Source Separation:**
```python
{
    "model_id": "ryan5453/demucs",
    "node_name": "Demucs",
    "namespace": "audio.separate",
    "overrides": {"audio": AudioRef},
    "return_type": {
        "vocals": AudioRef,
        "drums": AudioRef,
        "bass": AudioRef,
        "other": AudioRef,
    },
},
```

**LLM/Text Model:**
```python
{
    "model_id": "meta/meta-llama-3-70b-instruct",
    "node_name": "Llama3_70B_Instruct",
    "namespace": "text.generate",
    "return_type": str,
},
```

## Step 5: Regenerate Node Code

After adding entries to `gencode.py`, regenerate the nodes:

```bash
# Set your Replicate API token (for manual execution)
export REPLICATE_API_TOKEN=your_token_here

# Regenerate all nodes
python -m nodetool.nodes.replicate.gencode

# Or regenerate just one namespace
python -m nodetool.nodes.replicate.gencode --namespace image.generate
```

Note: In CI workflows, the token is provided via `${{ secrets.REPLICATE_API_TOKEN }}`.

## Step 6: Regenerate Metadata and DSL

```bash
nodetool package scan
nodetool codegen
```

## Step 7: Verify and Format

```bash
ruff check .
black .
```

## Common Issues

### No latest version found
Some models don't have a stable version. Skip these or wait for an update.

### Missing API token
Ensure `REPLICATE_API_TOKEN` environment variable is set.

### Schema parsing errors
Some models have malformed OpenAPI schemas. Skip these.

### Import errors after generation
Run `black .` to fix formatting issues.

## Checklist for New Models

- [ ] Model exists and is active on Replicate
- [ ] Model ID is correct (check URL)
- [ ] Node name follows naming conventions
- [ ] Namespace matches model category
- [ ] Return type matches model output
- [ ] Input overrides cover all file inputs
- [ ] Code regenerated successfully
- [ ] Linting passes (`ruff check .`)
- [ ] Formatting passes (`black --check .`)
