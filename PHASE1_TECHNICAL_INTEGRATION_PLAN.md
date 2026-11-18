# AUTEUR STUDIO — PHASE 1 TECHNICAL INTEGRATION PLAN
## Next-Generation Neural Image Intelligence Upgrade

> Historical architecture snapshot. Use [IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md](./IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md) for the current repository state and revised plan.

**Status:** AUDIT COMPLETE — READY FOR REVIEW  
**Date:** 2026-11-16  
**Branch:** main (commit d559854)  
**Author:** Kilo / Auteur Studio Team

---

## 1. EXISTING ARCHITECTURE ASSESSMENT

### 1.1 Application Stack
| Layer | Technology | Location |
|-------|------------|----------|
| **Frontend** | React 19 + TypeScript + Vite + Tailwind CSS 4 | `src/` |
| **Backend/API** | Express + TypeScript (tsx) | `server.ts` |
| **Local Inference** | onnxruntime-node (CPU) | `src/engine/` |
| **Audio** | Web Audio API (OfflineAudioContext) | `src/utils/audioMixer.ts` |
| **Timeline/NLE** | Custom engine (Canvas 2D) | `src/utils/timelineEngine.ts`, `src/utils/canvasRenderer.ts` |
| **External AI** | Google GenAI (Gemini) for prompt enhancement, video, music, voice, subtitles | `server.ts` |

### 1.2 Current Image Generation Pipeline (SDXS-512)
**File:** `src/engine/localDiffusionPipeline.ts` (361 lines)

**Architecture:** 4-ingredient ONNX pipeline running entirely on CPU via onnxruntime-node
```
CLIP BPE Tokenizer (vocab.json + merges.txt)
    ↓
CLIP ViT-L/14 Text Encoder (text_encoder.onnx) → [1, 77, 768] embeddings
    ↓
SDXS-512 UNet Brain (unet.onnx) → 1-step Euler denoising at σ=14.6
    ↓
TAESD VAE Decoder (vae_decoder.onnx) → 512×512 RGB
    ↓ (optional Img2Img)
SD VAE Encoder (vae_encoder.onnx) → latent encoding for reference images
```

**Model Source:** `skillsafeai/sdxs-512-dreamshaper-onnx-webgpu` (Hugging Face)  
**VAE Encoder Source:** `jdp8/sd-turbo-onnx`  
**Weights Location:** `.cache/models/sdxs-512/` (auto-downloaded on first run)  
**VRAM/Compute:** CPU-only, ~2.5s per image, ~2GB RAM for model weights  
**Resolution:** Fixed 512×512 latent → cropped to 512×288 (16:9), 288×512 (9:16), 512×512 (1:1)  
**Generation Modes:**
- `local-sdxs` — pure distilled (CFG=0), fastest
- `local-sdxs-cfg` — dual-pass CFG 1.05–1.35 with negative prompt
- `local-sdxs-img2img` — VAE encode reference → noise at strength 0.35–0.95 → denoise

**Tokenizer:** Custom CLIP BPE implementation (`src/engine/clipTokenizer.ts`) — matches OpenAI CLIP vocab/merges  
**Sampler:** Euler trailing schedule (`src/engine/diffusionSampler.ts`) — 1 step (SDXS distilled) or configurable steps

### 1.3 API Endpoints (server.ts)
| Endpoint | Purpose | Model |
|----------|---------|-------|
| `POST /api/prompts/enhance` | Prompt enhancement + audio transcription | Gemini 4 Hydrogen / 3.5 Transcribe |
| `POST /api/characters/analyze` | Character trait extraction from image | Gemini 4 Hydrogen Vision |
| `POST /api/media/image` | **Local diffusion** (SDXS-512) | ONNX CPU |
| `POST /api/media/video` | Video generation | Gemini Omni Flash |
| `POST /api/media/music` | Music generation | Lyria 3 |
| `POST /api/media/voice` | TTS (single/dual) | Gemini TTS |
| `POST /api/subtitles/generate` | Subtitle cues from audio/text | Gemini 4 Hydrogen / 3.5 Transcribe |

### 1.4 Frontend Architecture
- **Single-page studio** with left dock (6 tabs), center program monitor (Canvas 2D), right inspector, bottom timeline
- **GeneratePanel** — unified UI for image/video/restyle with engine selector, character lock, sampler controls
- **CharacterCastPanel** — character management with trait extraction, upload, overlay insertion
- **TimelineDock** — multi-track (V1, V2/overlay, A1/voice, A2/music, S1/subtitle) with snap, transitions, Ken Burns
- **ExportView** — renders final composition to video/WebM via Canvas capture

### 1.5 Data Types (src/types/studio.ts)
```typescript
ImageEngineId = 'local-sdxs' | 'local-sdxs-cfg' | 'local-sdxs-img2img'
VisualStyle = 'cinematic' | 'cartoon2d' | 'anime' | 'claymation' | 'noir' | 'pixar3d' | 'watercolor' | 'original'
TrackId = 'video' | 'overlay' | 'voice' | 'music' | 'subtitle'
```

---

## 2. GAP ANALYSIS VS. TARGET ARCHITECTURE

| Target Capability | Current State | Gap |
|-------------------|---------------|-----|
| **Multi-model generation** | Single SDXS-512 only | Need model registry + adapter pattern |
| **FLUX.2 / Qwen / MOSS / Z-Image** | Not integrated | Need Python inference service (PyTorch/Diffusers) |
| **Visual Intelligence Engine** | Gemini-only (cloud) | Need local VLM (Qwen-VL, LLaVA, etc.) |
| **ControlNet / IP-Adapter / LoRA** | Not supported | Need Diffusers pipeline integration |
| **Character consistency (IP-Adapter + LoRA)** | VAE Img2Img only (weak) | Need proper reference conditioning |
| **SAM 2 segmentation** | Not integrated | Need Python service |
| **Real-ESRGAN upscaling** | Not integrated | Need Python service |
| **Quality assessment (VLM evaluator)** | Not implemented | Need VLM comparison pipeline |
| **Hardware-aware model selection** | Fixed SDXS only | Need GPU detection, VRAM estimation |
| **ComfyUI backend option** | Not integrated | Need API client + workflow templates |
| **Model licensing audit** | Not documented | Need license registry per model |

---

## 3. RECOMMENDED MODEL INTEGRATIONS (PHASE 1 SCOPE)

### 3.1 Priority 1: Core Generation Models (Apache 2.0 / Commercially Usable)

| Model | Hugging Face ID | License | VRAM (FP16) | Use Case | Integration Complexity |
|-------|-----------------|---------|-------------|----------|------------------------|
| **FLUX.1-dev** (or **FLUX.1-schnell**) | `black-forest-labs/FLUX.1-dev` | Apache 2.0 | ~12-24GB | Primary high-quality generation | High (transformer + VAE + T5) |
| **FLUX.1-schnell** (distilled 4-step) | `black-forest-labs/FLUX.1-schnell` | Apache 2.0 | ~12GB | Fast generation | High |
| **Qwen-Image** | `Qwen/Qwen-Image` | Apache 2.0 | ~10-15GB | Detailed composition, text rendering | Medium (Diffusers native) |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | Apache 2.0 | ~10-15GB | Instruction-based editing | Medium |
| **MOSS-Image** | `OpenGVLab/MOSS-Image` | Apache 2.0 | ~8-12GB | Typography, layout | Medium |

> **Note:** FLUX.2 Klein 4B mentioned in spec is not yet released as of audit date. FLUX.1-dev/schnell are the current Apache 2.0 releases. Will monitor for FLUX.2.

### 3.2 Priority 2: Visual Intelligence & Control

| Component | Candidate | License | Integration |
|-----------|-----------|---------|-------------|
| **Vision-Language Model** | `Qwen/Qwen2.5-VL-7B-Instruct` or `microsoft/Phi-3.5-vision-instruct` | Apache 2.0 / MIT | Python service, prompt understanding + quality eval |
| **ControlNet** | `diffusers/controlnet-canny-sdxl-1.0`, `depth`, `openpose` | Apache 2.0 | Diffusers pipeline |
| **IP-Adapter** | `h94/IP-Adapter` (Plus/FaceID) | MIT | Diffusers pipeline, character consistency |
| **LoRA** | CivitAI / Hugging Face Hub loaders | Varies | Diffusers `load_lora_weights` |
| **SAM 2** | `facebook/sam2-hiera-base-plus` | Apache 2.0 | Python service, object masking |
| **Real-ESRGAN** | `xinntao/Real-ESRGAN` (x4plus) | BSD-3-Clause | Python service, upscaling |

### 3.3 Priority 3: Orchestration Backend

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **ComfyUI API** | Native workflow execution via REST | Supports all models, visual workflows, queue management | Extra process, learning curve |
| **Custom FastAPI + Diffusers** | Tailored Python service | Full control, optimized for Auteur | More engineering |
| **Hybrid** | ComfyUI for complex workflows, custom API for simple | Best of both | Two backends |

**Recommendation:** Start with **Custom FastAPI + Diffusers** for core generation (simpler, tighter integration), evaluate ComfyUI for Phase 2+ complex multi-model workflows.

---

## 4. PROPOSED ARCHITECTURE (PHASE 1 IMPLEMENTATION)

### 4.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTEUR STUDIO (Node.js/TypeScript)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Generate    │  │  Character   │  │  Timeline    │  │  Export      │   │
│  │  Panel       │  │  Cast Panel  │  │  Dock        │  │  View        │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │            │
│         └─────────────────┼─────────────────┼─────────────────┘            │
│                           ▼                                                 │
│              ┌────────────────────────┐                                   │
│              │   Unified Inference    │                                   │
│              │   API Client (TS)      │                                   │
│              └───────────┬────────────┘                                   │
└──────────────────────────┼────────────────────────────────────────────────┘
                           │ HTTP/WS (localhost:8000)
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PYTHON INFERENCE SERVICE (FastAPI)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  Model Registry │  │  Job Queue      │  │  Hardware       │            │
│  │  (JSON/DB)      │  │  (Redis/In-mem) │  │  Detection      │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                     │
│           ▼                    ▼                    ▼                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                   GENERATION ADAPTERS                            │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │     │
│  │  │ FLUX     │ │ Qwen     │ │ MOSS     │ │ SDXS     │           │     │
│  │  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ (ONNX)   │           │     │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│           │                    │                    │                     │
│           ▼                    ▼                    ▼                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  ControlNet     │  │  IP-Adapter     │  │  VLM Engine     │          │
│  │  Pipeline       │  │  Pipeline       │  │  (Qwen-VL)      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│           │                    │                    │                     │
│           ▼                    ▼                    ▼                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                    SHARED DIFFUSERS PIPELINES                    │     │
│  │  (model caching, CPU offload, FP16/BF16, attention slicing)     │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Model Registry Schema (JSON)

```json
{
  "models": [
    {
      "id": "flux-1-dev",
      "name": "FLUX.1-dev",
      "version": "1.0",
      "source": "huggingface",
      "repoId": "black-forest-labs/FLUX.1-dev",
      "license": "Apache-2.0",
      "commercialUse": true,
      "architecture": "flux",
      "components": ["transformer", "vae", "text_encoder", "text_encoder_2"],
      "vramGbFp16": 24,
      "vramGbInt8": 14,
      "vramGbNf4": 10,
      "supportedModes": ["txt2img", "img2img", "controlnet", "ip_adapter", "lora"],
      "maxResolution": [1024, 1024],
      "defaultSteps": 28,
      "quantizationOptions": ["fp16", "bf16", "int8", "nf4"],
      "downloadStatus": "not_downloaded",
      "installPath": null,
      "enabled": true
    },
    {
      "id": "qwen-image",
      "name": "Qwen-Image",
      "version": "1.0",
      "source": "huggingface",
      "repoId": "Qwen/Qwen-Image",
      "license": "Apache-2.0",
      "commercialUse": true,
      "architecture": "qwen_image",
      "components": ["transformer", "vae", "text_encoder"],
      "vramGbFp16": 15,
      "vramGbInt8": 9,
      "supportedModes": ["txt2img", "img2img", "controlnet", "ip_adapter"],
      "maxResolution": [1024, 1024],
      "defaultSteps": 20,
      "quantizationOptions": ["fp16", "bf16", "int8"],
      "downloadStatus": "not_downloaded",
      "enabled": true
    },
    {
      "id": "sdxs-512",
      "name": "SDXS-512 DreamShaper",
      "version": "1.0",
      "source": "huggingface",
      "repoId": "skillsafeai/sdxs-512-dreamshaper-onnx-webgpu",
      "license": "Apache-2.0",
      "commercialUse": true,
      "architecture": "sdxs_onnx",
      "components": ["tokenizer", "text_encoder", "unet", "vae_decoder", "vae_encoder"],
      "vramGbFp16": 0,
      "ramGb": 2,
      "supportedModes": ["txt2img", "img2img"],
      "maxResolution": [512, 512],
      "defaultSteps": 1,
      "quantizationOptions": ["fp32_onnx"],
      "downloadStatus": "downloaded",
      "installPath": ".cache/models/sdxs-512",
      "enabled": true
    }
  ]
}
```

### 4.3 Inference API Contract (FastAPI)

```python
# POST /api/v1/generate
{
  "prompt": "A young woman in burgundy coat at rainy train station...",
  "negative_prompt": "blurry, distorted...",
  "model_id": "flux-1-dev",  # or "auto" for router
  "mode": "txt2img",  # txt2img | img2img | edit | controlnet | ip_adapter
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "guidance_scale": 3.5,
  "seed": 424242,
  "reference_images": ["base64..."],  # for img2img, ip_adapter
  "controlnet": {
    "type": "canny|depth|openpose",
    "image": "base64...",
    "conditioning_scale": 0.8
  },
  "ip_adapter": {
    "image": "base64...",
    "scale": 0.6
  },
  "lora": [
    {"path": "character_lora.safetensors", "scale": 0.8}
  ],
  "return_metadata": true
}

# Response
{
  "images": ["base64_png..."],
  "seed": 424242,
  "model": "flux-1-dev",
  "timings": {"load": 1200, "inference": 3400, "total": 4600},
  "vram_peak_gb": 18.2,
  "metadata": {
    "prompt_adherence": 0.92,
    "quality_score": 0.88
  }
}
```

### 4.4 Visual Intelligence Engine (VLM) API

```python
# POST /api/v1/vlm/analyze
{
  "prompt": "Create a cinematic image of a young African woman...",
  "reference_images": ["base64..."]  # optional
}

# Response: Structured Scene Representation
{
  "subjects": [
    {
      "type": "person",
      "attributes": {
        "age": "young",
        "gender": "woman",
        "ethnicity": "African",
        "clothing": "long burgundy coat",
        "accessories": ["transparent umbrella"]
      },
      "pose": "standing"
    }
  ],
  "environment": {
    "location": "old European railway station",
    "background_elements": ["vintage train approaching"],
    "weather": "heavy rain",
    "time_of_day": "blue hour"
  },
  "camera": {
    "angle": "ground level",
    "framing": "cinematic wide shot"
  },
  "lighting": "dramatic blue-hour, volumetric",
  "surface_details": ["wet pavement reflections", "rain droplets"],
  "composition": "rule of thirds, leading lines from platform",
  "style_keywords": ["cinematic", "photorealistic", "atmospheric"],
  "negative_constraints": ["no extra limbs", "no watermark", "no text artifacts"]
}
```

---

## 5. FILE STRUCTURE CHANGES (PHASE 1)

### 5.1 New Files to Create

```
├── python_inference/
│   ├── pyproject.toml / requirements.txt
│   ├── main.py                    # FastAPI entry point
│   ├── config/
│   │   ├── model_registry.json    # Model definitions
│   │   └── hardware_profiles.json # VRAM/quantization mappings
│   ├── core/
│   │   ├── registry.py            # ModelRegistry class
│   │   ├── hardware.py            # GPU detection, VRAM estimation
│   │   ├── queue.py               # JobQueue (asyncio + Redis optional)
│   │   └── adapters.py            # BaseGenerationAdapter abstract class
│   ├── adapters/
│   │   ├── flux_adapter.py        # FLUX.1 implementation
│   │   ├── qwen_image_adapter.py  # Qwen-Image implementation
│   │   ├── moss_adapter.py        # MOSS-Image implementation
│   │   └── sdxs_onnx_adapter.py   # Wrapper for existing ONNX engine
│   ├── pipelines/
│   │   ├── controlnet.py          # ControlNet integration
│   │   ├── ip_adapter.py          # IP-Adapter integration
│   │   ├── lora.py                # LoRA loading/management
│   │   └── vlm.py                 # Qwen-VL for prompt understanding + quality eval
│   ├── services/
│   │   ├── downloader.py          # Model download with progress
│   │   ├── upscaler.py            # Real-ESRGAN wrapper
│   │   └── segmenter.py           # SAM 2 wrapper
│   └── routes/
│       ├── generate.py            # /api/v1/generate
│       ├── models.py              # /api/v1/models (list, status, download)
│       ├── vlm.py                 # /api/v1/vlm/*
│       └── health.py              # /api/v1/health
│
├── src/
│   ├── services/
│   │   ├── inferenceClient.ts     # TypeScript client for Python service
│   │   ├── modelRegistry.ts       # Client-side model registry sync
│   │   └── hardwareDetector.ts    # WebGPU/GPU detection for UI hints
│   ├── engine/
│   │   ├── generationRouter.ts    # Intelligent model selection logic
│   │   └── promptInterpreter.ts   # Structured prompt → VLM → generation params
│   ├── components/
│   │   ├── ModelSelector.tsx      # Advanced model selection UI
│   │   ├── GenerationSettings.tsx # Unified settings panel
│   │   ├── ReferenceImageUpload.tsx
│   │   ├── ControlNetPanel.tsx
│   │   ├── IPAdapterPanel.tsx
│   │   └── QualityAssessment.tsx  # Post-generation quality review
│   └── types/
│       └── generation.ts          # New types for multi-model generation
```

### 5.2 Files to Modify

| File | Changes |
|------|---------|
| `server.ts` | Add proxy routes to Python service (`/api/v1/*`), deprecate direct ONNX calls |
| `src/components/GeneratePanel.tsx` | Replace engine selector with ModelSelector, add ReferenceImageUpload, ControlNetPanel, IPAdapterPanel |
| `src/types/studio.ts` | Add `GenerationMode`, `ModelConfig`, `ReferenceImage`, `ControlNetConfig`, `IPAdapterConfig`, `QualityMetrics` |
| `src/data/imageEngines.ts` | Deprecate `IMAGE_ENGINES` in favor of dynamic registry from Python service |
| `package.json` | Add `concurrently` for dev, Python service startup script |

---

## 6. HARDWARE REQUIREMENTS & COMPATIBILITY

### 6.1 Minimum Hardware Targets

| Tier | GPU VRAM | System RAM | Models Supported (Quantized) |
|------|----------|------------|------------------------------|
| **Entry** | 8 GB | 16 GB | SDXS-512 (ONNX CPU), Qwen-Image (INT8), FLUX-schnell (NF4) |
| **Standard** | 12-16 GB | 32 GB | FLUX.1-dev (INT8/NF4), Qwen-Image (FP16), MOSS (FP16) |
| **High-End** | 24+ GB | 64 GB | FLUX.1-dev (FP16/BF16), all models FP16, multi-model concurrent |

### 6.2 Quantization Strategy

| Model | FP16 VRAM | INT8 VRAM | NF4/4-bit VRAM | Quality Loss |
|-------|-----------|-----------|----------------|--------------|
| FLUX.1-dev | ~24 GB | ~14 GB | ~10 GB | Minimal (NF4) |
| FLUX.1-schnell | ~12 GB | ~8 GB | ~6 GB | Minimal |
| Qwen-Image | ~15 GB | ~9 GB | ~7 GB | Low |
| MOSS-Image | ~12 GB | ~7 GB | ~5 GB | Low |
| SDXS-512 (ONNX) | N/A (CPU) | 2 GB RAM | N/A | Baseline |

**Implementation:** Use `bitsandbytes` for 8-bit/4-bit quantization, `optimum-quanto` for NF4. Auto-select based on detected VRAM.

---

## 7. LICENSING AUDIT (CRITICAL)

| Model | License | Commercial Use | Derivatives | Attribution | Status |
|-------|---------|----------------|-------------|-------------|--------|
| FLUX.1-dev | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| FLUX.1-schnell | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| Qwen-Image | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| Qwen-Image-Edit | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| MOSS-Image | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| Qwen2.5-VL | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| ControlNet (SDXL) | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| IP-Adapter | MIT | ✅ | ✅ | Required | APPROVED |
| SAM 2 | Apache-2.0 | ✅ | ✅ | Required | APPROVED |
| Real-ESRGAN | BSD-3-Clause | ✅ | ✅ | Required | APPROVED |
| SDXS-512 | Apache-2.0 | ✅ | ✅ | Required | APPROVED (current) |

> **All candidate models have commercially permissive licenses.** No GPL, Creative Commons Non-Commercial, or restrictive licenses in the Phase 1 selection.

---

## 8. IMPLEMENTATION PHASES (DETAILED)

### Phase 1A: Foundation (Week 1-2)
- [ ] Set up Python FastAPI service skeleton with model registry
- [ ] Implement hardware detection (GPU, VRAM, CUDA version)
- [ ] Create base `GenerationAdapter` abstract class
- [ ] Wrap existing SDXS-512 ONNX engine as `SDXSONNXAdapter`
- [ ] Add `/api/v1/health`, `/api/v1/models` endpoints
- [ ] TypeScript `InferenceClient` with health check + model list sync

### Phase 1B: Core Model Integration (Week 2-4)
- [ ] Integrate **FLUX.1-schnell** (fastest to validate pipeline)
- [ ] Integrate **FLUX.1-dev** (quality baseline)
- [ ] Integrate **Qwen-Image** (text rendering, composition)
- [ ] Implement model download manager with progress UI
- [ ] Add quantization auto-selection based on VRAM
- [ ] Benchmark all 3 models on identical prompts

### Phase 1C: Visual Intelligence & Control (Week 3-5)
- [ ] Integrate **Qwen2.5-VL-7B** for prompt understanding
- [ ] Build structured scene representation parser
- [ ] Integrate **ControlNet** (Canny, Depth, OpenPose)
- [ ] Integrate **IP-Adapter** (Plus, FaceID variants)
- [ ] Integrate **LoRA** loading (character LoRAs)
- [ ] Add reference image upload + preprocessing in GeneratePanel

### Phase 1D: Quality & Polish (Week 4-6)
- [ ] Implement **Real-ESRGAN** upscaling (2x, 4x)
- [ ] Implement **SAM 2** for object masking
- [ ] Build **QualityAssessment** component (VLM-based comparison)
- [ ] Add generation history + comparison gallery
- [ ] Implement intelligent generation router (auto model selection)
- [ ] Regression test: existing SDXS-512, timeline, export, all panels

---

## 9. TESTING & VERIFICATION PLAN

### 9.1 Benchmark Suite (Automated)
```python
BENCHMARK_PROMPTS = [
    # Photorealistic portrait
    "Portrait of a young woman, burgundy coat, transparent umbrella, rainy European train station, blue hour, cinematic lighting, 85mm lens",
    # Complex environment
    "Aerial view of cyberpunk megacity at night, neon reflections on wet streets, flying vehicles, volumetric fog, ultra wide angle",
    # Multiple characters
    "Two anime characters facing each other on a cliff at sunset, detailed hair, expressive eyes, wind blowing, Studio Ghibli style",
    # Detailed clothing
    "Victorian noblewoman in embroidered silk gown, lace cuffs, pearl necklace, ornate brocade patterns, studio lighting",
    # Cinematic composition
    "Low angle hero shot, lone astronaut on Mars, dust storm, Olympus Mons in background, dramatic rim lighting, anamorphic lens flare",
    # Typography
    "Movie poster title 'AUTEUR' in bold cinematic lettering, metallic texture, volumetric light rays, dark atmosphere",
    # Character consistency (reference-based)
    "Same character as reference, now in a medieval castle library, reading a glowing book, warm candlelight",
    # Image editing
    "Change the character's coat color from burgundy to emerald green, keep everything else identical",
]
```

### 9.2 Metrics to Capture
| Metric | Target |
|--------|--------|
| Generation time (FLUX-schnell) | < 5s @ 1024² |
| Generation time (FLUX-dev) | < 15s @ 1024² |
| Generation time (Qwen-Image) | < 12s @ 1024² |
| VRAM peak (FLUX-dev NF4) | < 11 GB |
| Prompt adherence (VLM eval) | > 0.85 |
| Quality score (VLM eval) | > 0.80 |
| Character consistency (IP-Adapter) | > 0.75 face similarity |
| Text accuracy (Qwen-Image) | > 90% character correctness |

### 9.3 Regression Tests
- [ ] Existing SDXS-512 generation still works (txt2img, img2img, CFG)
- [ ] All 6 left-dock tabs functional
- [ ] Timeline: add/move/split/trim clips, transitions, Ken Burns
- [ ] Audio: voice, music, subtitle sync
- [ ] Export: WebM render matches preview
- [ ] Character cast: upload, traits, overlay insert
- [ ] Prompt enhancement (Gemini fallback still works)

---

## 10. RISK ASSESSMENT & MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FLUX models require too much VRAM for target users | High | High | Prioritize NF4/INT8 quantization; keep SDXS-512 as CPU fallback |
| Python service adds deployment complexity | Medium | Medium | Provide Docker Compose; single binary via PyInstaller for distribution |
| Model download sizes (20-50 GB total) | High | Medium | Lazy download on first use; progress UI; cleanup unused models |
| VLM quality evaluation adds latency | Medium | Low | Run async; cache results; optional in UI |
| Breaking changes in Diffusers/Transformers | Medium | Medium | Pin versions; test in CI; vendor critical patches |
| IP-Adapter/ControlNet compatibility across models | Medium | High | Test each adapter with each model; document supported combinations |

---

## 11. DEPENDENCY ADDITIONS

### 11.1 Python (python_inference/requirements.txt)
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.10.0
pydantic-settings==2.6.0
torch==2.5.0
torchvision==0.20.0
diffusers==0.31.0
transformers==4.46.0
accelerate==1.3.0
safetensors==0.5.0
huggingface-hub==0.26.0
bitsandbytes==0.45.0
optimum-quanto==0.2.0
controlnet-aux==0.0.8
segment-anything-2==1.0.0
realesrgan==0.3.0
pillow==11.0.0
numpy==2.1.0
redis==5.2.0  # optional, for distributed queue
python-multipart==0.0.12
```

### 11.2 Node.js (package.json additions)
```json
{
  "dependencies": {
    "concurrently": "^9.0.0"
  },
  "scripts": {
    "dev": "concurrently \"tsx server.ts\" \"cd python_inference && python -m uvicorn main:app --reload --port 8000\"",
    "inference:install": "cd python_inference && pip install -r requirements.txt",
    "inference:start": "cd python_inference && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
  }
}
```

---

## 12. APPROVAL CHECKLIST (PRE-IMPLEMENTATION)

Before Phase 1 implementation begins, confirm:

- [ ] **Architecture Review** — Python service + adapters pattern approved
- [ ] **Model Selection** — FLUX.1-schnell, FLUX.1-dev, Qwen-Image, MOSS-Image confirmed
- [ ] **Licensing** — All models Apache-2.0/MIT/BSD-3 approved for commercial use
- [ ] **Hardware Targets** — Minimum 8GB VRAM / 16GB RAM baseline accepted
- [ ] **Existing Preservation** — SDXS-512 ONNX engine remains functional throughout
- [ ] **API Contract** — FastAPI `/api/v1/*` interface approved
- [ ] **UI Changes** — GeneratePanel refactor scope approved (ModelSelector, ControlNet, IP-Adapter panels)
- [ ] **Timeline** — 6-week Phase 1 timeline accepted
- [ ] **Benchmark Criteria** — Success metrics defined and agreed

---

## 13. NEXT STEPS

**Upon approval of this plan:**

1. **Create `python_inference/` directory structure** with FastAPI skeleton
2. **Implement ModelRegistry + HardwareDetection** (Phase 1A)
3. **Wrap SDXS-512 as first adapter** — validates architecture with zero model changes
4. **Present working end-to-end demo** (Node.js → Python → SDXS-512 → back)
5. **Then integrate FLUX.1-schnell** as first PyTorch model

**Do not modify any existing TypeScript/React files until Phase 1A Python service is running and validated.**

---

*End of Phase 1 Technical Integration Plan*