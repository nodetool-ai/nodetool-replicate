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

## 2026-03-28 - Latest Featured Models Added

- Model ID: `google/gemini-3-flash`
- Namespace: `text.generate`
- Description: Google's most intelligent model built for speed with frontier intelligence, superior search, and grounding, 645.4K runs

- Model ID: `black-forest-labs/flux-2-klein-4b`
- Namespace: `image.generate`
- Description: Very fast image generation and editing model. 4 steps distilled, sub-second inference for production and near real-time applications, 7.9M runs

- Model ID: `google/nano-banana-2`
- Namespace: `image.generate`
- Description: Google's fast image generation model with conversational editing, multi-image fusion, and character consistency, 3.7M runs

- Model ID: `inworld/tts-1.5-max`
- Namespace: `audio.generate`
- Description: Highest-quality text-to-speech with <200ms latency, emotion control, and 15-language support, 12.3K runs

**Notes:**
- Temporarily disabled models with 404 errors: `luma/ray`, `meta/meta-llama-3.1-405b-instruct`, `runwayml/upscale-v1`, `prunaai/p-video`
- Total node count increased from previous run to 158 nodes
