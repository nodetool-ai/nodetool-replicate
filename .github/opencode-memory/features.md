# Features Log

This file tracks nodes and features added by the automated Claude Code agent.

## Format

Each entry should follow this format:
```
## YYYY-MM-DD - Feature/Node Name
- Model ID: `owner/model-name`
- Namespace: `category.subcategory`
- Description: Brief description of what was added
```

---

## Initial Setup - 2025-01-14

Repository configured with Claude Code memory and automated sync workflow.

---

## 2026-01-18 - Featured Models Added

- Model ID: `openai/gpt-image-1.5`
- Namespace: `image.generate`
- Description: OpenAI's latest image generation model with better instruction following and adherence to prompts, 1.3M runs

- Model ID: `black-forest-labs/flux-2-max`
- Namespace: `image.generate`
- Description: Black Forest Labs' highest fidelity image model with superior prompt adherence and editing consistency, 198.8K runs

- Model ID: `google/lyria-2`
- Namespace: `audio.generate`
- Description: Google's music generation model producing 48kHz stereo audio from text prompts, 55.1K runs

---

## 2026-03-06 - New Models Added

- Model ID: `microsoft/phi-3-mini-4k-instruct`
- Namespace: `text.generate`
- Description: Microsoft's Phi-3-Mini-4K-Instruct, a 3.8B parameter lightweight open model trained with Phi-3 datasets

- Model ID: `deforum-art/deforum-stable-diffusion`
- Namespace: `video.generate`
- Description: Deforum Stable Diffusion for creating animated videos with Stable Diffusion

---

## 2026-03-07 - New Models Added

- Model ID: `fofr/sdxl-controlnet-lora`
- Namespace: `image.generate`
- Description: Multi-controlnet SDXL with LoRA loading, img2img, and inpainting capabilities

- Model ID: `lucataco/realistic-vision-v5.1`
- Namespace: `image.generate`
- Description: Implementation of Realistic Vision v5.1 with VAE for photorealistic image generation

- Model ID: `prompthero/openjourney-v4`
- Namespace: `image.generate`
- Description: SD 1.5 trained with +124k Midjourney v4 images by PromptHero
