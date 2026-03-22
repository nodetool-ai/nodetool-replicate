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

## 2026-03-22 - New Models Added

- Model ID: `microsoft/phi-3-mini-128k-instruct`
- Namespace: `text.generate`
- Description: Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets

- Model ID: `fofr/sdxl-controlnet-lora`
- Namespace: `image.generate`
- Description: Multi-controlnet, lora loading, img2img, inpainting for SDXL

- Model ID: `lucataco/animate-diff`
- Namespace: `video.generate`
- Description: Animate Your Personalized Text-to-Image Diffusion Models for video generation

---

## 2026-03-22 - Models Disabled

- Model ID: `luma/ray`
- Reason: Model returns 404 Not Found - no longer available on Replicate

- Model ID: `runwayml/upscale-v1`
- Reason: Model returns 404 Not Found - no longer available on Replicate
