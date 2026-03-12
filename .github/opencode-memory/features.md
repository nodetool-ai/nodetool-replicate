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

## 2026-03-12 - New Video, Audio, and Image Models

### Video Generation Models

- Model ID: `vidu/q3-pro`
- Namespace: `video.generate`
- Description: High-fidelity video generation with text-to-video, image-to-video, and start-end-to-video modes up to 16 seconds

- Model ID: `vidu/q3-turbo`
- Namespace: `video.generate`
- Description: Fast video generation with text-to-video, image-to-video, and start-end-to-video modes up to 16 seconds

- Model ID: `lightricks/ltx-2-pro`
- Namespace: `video.generate`
- Description: High visual fidelity video generation with fast turnaround for marketing and content creation

- Model ID: `lightricks/ltx-2-fast`
- Namespace: `video.generate`
- Description: Fast video generation ideal for rapid ideation and mobile workflows with instant feedback

### Text-to-Speech Models

- Model ID: `inworld/tts-1.5-max`
- Namespace: `audio.generate`
- Description: Highest-quality text-to-speech with <200ms latency, emotion control, and 15-language support

- Model ID: `inworld/tts-1.5-mini`
- Namespace: `audio.generate`
- Description: Lightweight text-to-speech with fast inference and multi-language support

### Image Processing Models

- Model ID: `recraft-ai/recraft-vectorize`
- Namespace: `image.process`
- Description: Convert raster images to high-quality SVG format with precision and clean vector paths

- Model ID: `recraft-ai/recraft-remove-background`
- Namespace: `image.process`
- Description: Automated background removal tuned for AI-generated content, product photos, and portraits

- Model ID: `recraft-ai/recraft-creative-upscale`
- Namespace: `image.upscale`
- Description: Creative image upscaling with artistic enhancement and detail preservation

- Model ID: `recraft-ai/recraft-crisp-upscale`
- Namespace: `image.upscale`
- Description: Crisp image upscaling maintaining sharp edges and fine details

### Image Generation with Transparency

- Model ID: `jide/nano-banana-2-transparent`
- Namespace: `image.generate`
- Description: Generates images with real RGBA transparency using triangular mesh technology
