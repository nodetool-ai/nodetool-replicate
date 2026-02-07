# Replicate Node for Nodetool

The Replicate node enables seamless integration with [Replicate](https://replicate.com)'s API for running machine learning models in your workflows. This node allows you to leverage Replicate's extensive collection of pre-trained models directly within your Nodetool pipelines.

## Features

- Direct integration with Replicate's API
- Real-time model status monitoring
- Support for various model types and inputs
- Automatic error handling and status reporting

## Audio Processing Nodes

The Replicate node includes several specialized audio processing nodes:

### Audio Enhancement

- **AudioSR**: Upscale and enhance audio quality using AI super-resolution
- Supports configurable parameters like seed, steps, and guidance scale
- Ideal for improving audio fidelity and clarity

### Audio Generation

- **Riffusion**: Generate music using stable diffusion
- **MusicGen**: Create music from text prompts or existing melodies
- **TortoiseTTS**: Text-to-speech with voice cloning capabilities
- **StyleTTS2**: Advanced text-to-speech generation
- **MMAudio**: Add AI-generated sound to videos
- **RealisticVoiceCloning**: Create song covers with AI voice models

### Audio Separation

- **Demucs**: Professional audio source separation
  - Separate tracks into vocals, drums, bass, and other instruments
  - Multiple model options (htdemucs, mdx_q, etc.)
  - Support for various output formats (MP3, WAV, FLAC)

### Audio Transcription

- **IncrediblyFastWhisper**: Fast audio transcription and translation
  - Support for 90+ languages
  - Word-level and chunk-level timestamp options
  - Optional speaker diarization

## Image Processing Nodes

The Replicate node includes several specialized image processing nodes:

### Image Analysis

- **CLIP Interrogator**: Generate text prompts that match images
  - Multiple modes: best, fast, classic, negative
  - Support for different CLIP models
- **Moondream2**: Efficient vision language model for edge devices
- **BLIP/BLIP2**: Generate image captions and answer questions about images
- **LLaVA-13b**: Visual instruction model with GPT-4 level capabilities
- **NSFW Detection**: Detect inappropriate content in images
- **CLIP Features**: Extract features from images using CLIP models

### Image Enhancement

- **CodeFormer**: Restore old photos and AI-generated faces
  - Face upsampling and background enhancement
  - Adjustable quality/fidelity balance
- **Night Enhancement**: Improve low-light images
- **SUPIR**: Photo-realistic image restoration
  - Multiple versions (v0Q, v0F) for different quality levels
  - Advanced color correction and enhancement
- **MAXIM**: Multi-purpose image processing
  - Denoising, deblurring, deraining
  - Indoor/outdoor dehazing
  - Low-light enhancement
- **Old Photos Restoration**: Specialized restoration for vintage photos

### Face Manipulation

- **FaceToMany**: Transform faces into different styles
  - 3D, emoji, pixel art, video game, claymation, toy styles
  - Adjustable strength and control parameters
- **BecomeImage**: Adapt faces to match reference images
- **PhotoMaker**: Create stylized portraits
  - Multiple style presets (Disney, Digital Art, Comic Book, etc.)
  - Support for multiple input images
- **FaceToSticker**: Convert faces into sticker-style images
- **InstantID**: Advanced face transformation
  - Multiple variants (Photorealistic, Artistic)
  - Extensive control over generation parameters
  - Support for pose control and face enhancement

## Setup

1. Sign up for a Replicate account at https://replicate.com
2. Get your API token from the Replicate dashboard
3. Configure your API token in the Nodetool app: `REPLICATE_API_TOKEN`

## Usage

1. Add a Replicate node to your workflow
2. Select the desired model from Replicate's collection
3. Configure the model inputs according to the model's requirements
4. Connect the node to your workflow's input and output nodes
5. Run your workflow

The node will automatically handle:

- API authentication
- Model inference requests
- Status monitoring
- Result processing

## Status Indicators

The Replicate node includes a status indicator that shows:

- 🟢 **Online**: Model is available and ready
- 🔴 **Offline**: Model is currently unavailable

## Error Handling

The node includes built-in error handling for common issues:

- Invalid API tokens
- Model availability issues
- Input validation errors
- Network connectivity problems

## New Features

### Recently Added Models (177+ total nodes)

#### 3D Generation
- **InstantMesh**: Generate 3D meshes from images
- **SplatterImage**: Create 3D gaussian splats from single images
- **Flux_3D**: 3D-aware image generation

#### Advanced Image Processing
- **InstructPix2Pix**: Edit images using text instructions
- **ControlNet_Scribble**: Generate images from scribble sketches
- **AnimateDiff**: Create animations from images
- **GroundedSegmentAnything**: Advanced segmentation with text prompts
- **SAM_HQ**: High-quality segmentation with Segment Anything Model

#### Video Processing
- **StableVideoDiffusion**: Generate videos from images
- **VideoToVideo**: Transform videos with AI

#### Code Generation
- **CodeLlama**: 13B and 34B instruction-tuned models for code generation

#### Face & Portrait
- **FaceToAll**: Transform faces into various artistic styles
- **LivePortrait**: Animate portraits with driving videos

#### Advanced Features
- **AnyComfyUI Workflow**: Run custom ComfyUI workflows
- **MusicGen_Melody**: Generate music with melody conditioning
- **RecognizeAnything**: Detect and recognize objects in images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For information on adding new Replicate models, see [CODEGEN.md](CODEGEN.md).

## License

AGPL
