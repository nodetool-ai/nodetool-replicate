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

---

## 2026-03-29 - Latest Popular Models Added

- Model ID: `google/veo-3.1-fast`
- Namespace: `video.generate`
- Description: New and improved version of Veo 3 Fast with higher-fidelity video, context-aware audio and last frame support, 531,249 runs

- Model ID: `qwen/qwen3-tts`
- Namespace: `audio.generate`
- Description: A unified Text-to-Speech model featuring three powerful modes: Voice, Clone and Design, 202,446 runs

- Model ID: `bytedance/seedream-4.5`
- Namespace: `image.generate`
- Description: Upgraded Bytedance image model with stronger spatial understanding and world knowledge, 5,864,554 runs

**Notes:**
- Total node count increased from 158 to 161 nodes

---

## 2026-04-02 - Latest Popular Models Added

- Model ID: `bytedance/seedream-5-lite`
- Namespace: `image.generate`
- Description: ByteDance's latest image generation model with built-in reasoning, example-based editing, and deep domain knowledge. Generate up to 3K resolution images, edit with reference examples, and create batch image sets, 580,900+ runs

- Model ID: `runwayml/gen-4.5`
- Namespace: `video.generate`
- Description: State-of-the-art video generation model delivering unprecedented visual fidelity, cinematic realism, and precise creative control. Ranked #1 in Artificial Analysis Text-to-Video Leaderboard with 1,247 Elo score, 64,100+ runs

- Model ID: `xai/grok-imagine-video`
- Namespace: `video.generate`
- Description: xAI's image-to-video model that animates still images into short videos with synchronized audio. Uses Aurora autoregressive mixture-of-experts architecture for realistic motion, object interactions, and automatically generated sound, 342,500+ runs

**Notes:**
- Total node count increased from 161 to 164 nodes

---

## 2026-04-05 - Latest Featured Models Added

- Model ID: `recraft-ai/recraft-v4`
- Namespace: `image.generate`
- Description: Recraft AI's latest image generation model with brand design, vector art, and realistic photography capabilities. Features text-to-image, image-to-image, and natural image editing with consistent style and brand alignment, 210.4K+ runs

- Model ID: `google/lyria-3-pro`
- Namespace: `audio.generate`
- Description: Google's most advanced music generation model producing high-quality instrumental music with improved coherence, structure, and audio fidelity. Generates diverse musical styles and genres from text prompts, very recent release with growing adoption

- Model ID: `inworld/tts-1.5-mini`
- Namespace: `audio.generate`
- Description: InWorld's lightweight text-to-speech model optimized for speed and efficiency while maintaining high voice quality. Features low latency speech synthesis with emotion control and multi-language support, 5.7K+ runs

**Notes:**
- Total node count increased from 164 to 167 nodes
