# AUTEUR STUDIO — IMAGE GENERATION QUALITY UPGRADE
## Technical Implementation Plan

**Date:** 2026-11-16  
**Status:** PLANNING — PRE-IMPLEMENTATION  
**Environment Constraint:** CPU-only Codespace (7.8GB RAM). No GPU. All model training/inference experiments must run on Kaggle GPU (T4/P100 16GB VRAM).

---

## 0. EXECUTIVE SUMMARY

Replace SDXS-512 (ONNX CPU, 512px, weak prompt fidelity) as the primary image generation engine with a multi-model architecture based on Z-Image-Turbo and FLUX.2-klein-4B as primary production models, with Z-Image and Qwen-Image-2512 as high-quality experimental options. Preserve SDXS-512 as a lightweight CPU fallback.

**Core principle:** Do not replace the production engine until Kaggle benchmarks demonstrate measurable improvement in prompt fidelity and image quality.

---

## 1. MODEL ARCHITECTURE (TASK 1)

### 1.1 Model Tiers

| Tier | Model | HF Repo ID | License | Role | Steps | Resolution | VRAM (est.) |
|------|-------|------------|---------|------|-------|------------|-------------|
| Primary 1 | Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | Apache-2.0 | Default production | 8 NFE | 1024×1024 | ~12-16GB |
| Primary 2 | FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4B` | Apache-2.0 | Fast fallback | 4 | up to 4MP | ~8.4-13GB |
| Experimental | Z-Image | `Tongyi-MAI/Z-Image` | Apache-2.0 | High quality, 50 steps | 50 | 1024×1024 | ~16-24GB |
| Experimental | Qwen-Image-2512 | `Qwen/Qwen-Image` | Apache-2.0 | Best fidelity (24-48GB VRAM needed) | 1-2 | up to 1664×1664 | ~27.7GB+ |
| Optional | MOSS-Image-SFT | `OpenMOSS-Team/MOSS-Image-SFT` | Apache-2.0 | Text/layout | 50 | 1024×1024 | ~13GB |
| Fallback | SDXS-512 (existing) | `skillsafeai/sdxs-512-dreamshaper-onnx-webgpu` | Apache-2.0 | CPU fallback | 1 | 512×512 | N/A (CPU) |

### 1.2 Adapter Architecture (Python Inference Service)

Each model uses its **official, compatible inference pipeline** — no mixing components across architectures.

```
File structure:
python-service/
├── main.py              # FastAPI app
├── models.py            # Pydantic schemas
├── engine_loader.py     # Lazy loading, VRAM management, queue
├── adapters/
│   ├── base_adapter.py    # Abstract BaseGenerationAdapter
│   ├── zimage_turbo.py    # DiffusionPipeline (ZImagePipeline)
│   ├── flux_klein.py      # Flux2KleinPipeline
│   ├── zimage_base.py     # DiffusionPipeline (Z-Image base)
│   ├── qwen_2512.py       # QwenImagePipeline (official)
│   ├── moss_sft.py        # DiffusionPipeline + pipeline_moss_image
│   └── sdxs_onnx.py       # Existing ONNX wrapper (fallback)
├── quality/
│   ├── vlm_evaluator.py   # Qwen3-VL-4B optional visual evaluator
│   └── florence2.py       # Florence-2 object detection + OCR
├── prompt/
│   ├── parser.py          # Qwen3-VL-4B or lightweight instruction model
│   └── schemas.py         # Structured scene spec
├── upscale/
│   └── real_esrgan.py     # Real-ESRGAN 4x upscaling
├── requirements.txt
└── .env.example
```

### 1.3 Official Pipeline Details

**Z-Image-Turbo:**
- Pipeline: `DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo")` → `ZImagePipeline`
- Official config: `num_inference_steps=8` (9 scheduler steps → 8 DiT forwards), `guidance_scale=0.0`, `height=1024`, `width=1024`
- No CFG, no negative prompts
- Text encoder: Qwen3 4B (native, loaded by pipeline)

**FLUX.2-klein-4B:**
- Pipeline: `Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")` (requires `git+https://github.com/huggingface/diffusers.git`)
- Official config: `num_inference_steps=4`, `guidance_scale=1.0`, `height=1024`, `width=1024`
- Text encoder: Qwen3 4B (native)

**Z-Image (base):**
- Pipeline: `DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image")`
- Official config: `num_inference_steps=50`, CFG supported, negative prompts supported

**Qwen-Image-2512:**
- Pipeline: Official QwenImagePipeline (diffusers native)
- VRAM: 27.7GB min BF16 — does NOT fit Kaggle 16GB. Requires 24-48GB production GPU.
- Disabled by default in Kaggle benchmarks

**MOSS-Image-SFT:**
- Pipeline: `DiffusionPipeline` + `custom_pipeline="pipeline_moss_image"`, `trust_remote_code=True`
- Official config: `num_inference_steps=50`, `guidance_scale=3.0`, `uncond_skip_layers=[22]`, shift=5
- VAE: FP32 (model card requirement)

### 1.4 Multi-GPU Constraint

Each benchmark script loads ONE model on ONE GPU. Two T4 GPUs (T4x2) provide 16GB VRAM each, **not pooled**. Scripts use single-GPU placement (`enable_model_cpu_offload()` + `.to("cuda")`).

---

## 2. PROMPT UNDERSTANDING LAYER (TASK 2)

### 2.1 Model Selection

**Primary candidate:** Qwen3-VL-4B-Instruct (Apache-2.0, 4B parameters, ~4GB VRAM BF16, instruction following + vision)
**Fallback candidates:** Qwen2.5-0.5B-Instruct (~1GB, text-only), Phi-3.5-mini-Instruct (3.8B, MIT)

### 2.2 Structured Scene Specification (JSON Schema)

```json
{
  "original_prompt": "string (preserved verbatim)",
  "optimized_prompt": "string (rewritten, preserving all elements)",
  "mode": "faithful | creative",
  "subjects": [
    {
      "type": "person | animal | object",
      "count": 1,
      "physical_appearance": "string",
      "facial_characteristics": "string",
      "clothing": "string",
      "accessories": ["string"],
      "action": "string",
      "position": "foreground | middle-ground | background",
      "spatial_relationship": "string"
    }
  ],
  "object_counts": { "object_name": number },
  "environment": {
    "location": "string",
    "background_elements": ["string"],
    "weather": "string",
    "time_of_day": "string",
    "season": "string"
  },
  "lighting": {
    "type": "string",
    "color_temperature": "string",
    "direction": "string"
  },
  "camera": {
    "framing": "string",
    "angle": "string",
    "perspective": "string",
    "lens": "string",
    "depth_of_field": "string"
  },
  "color_palette": ["string"],
  "artistic_style": "string",
  "atmospheric_conditions": "string",
  "constraints": {
    "must_include": ["string"],
    "must_not_include": ["string"],
    "colors_required": ["string"]
  },
  "model_recommendation": "z-image-turbo | flux-klein-4b | z-image | qwen-2512 | moss-sft",
  "estimated_vram_gb": 16,
  "estimated_steps": 8
}
```

### 2.3 Modes

- **FAITHFUL MODE:** Preserve user description as closely as possible. Minimal rewording. Never discard subjects, counts, or positions.
- **CREATIVE MODE:** Optional enhancement (quality descriptors, technical terms) while retaining all essential requirements.

### 2.4 Implementation

File: `python-service/prompt/parser.py`
- Load Qwen3-VL-4B-Instruct (or Qwen2.5-0.5B-Instruct fallback)
- Endpoint: `POST /v1/prompt/parse`
- Returns structured scene spec + optimized prompt
- Never uses the old 77-token CLIP pipeline for new models — each model uses its native text encoder

---

## 3. VISUAL QUALITY VERIFICATION (TASK 3)

### 3.1 Models

- **Primary:** Qwen3-VL-4B-Instruct (visual understanding, prompt-image comparison)
- **Supporting:** Florence-2 (object detection, image captioning, OCR)

### 3.2 Verification Pipeline

After generation:
1. Load generated image + original prompt + structured scene spec
2. Qwen3-VL-4B evaluates: missing subjects, incorrect clothing, wrong colors, incorrect counts, incorrect positions, missing background, incorrect composition, unrequested additions
3. Florence-2 provides object detection counts + OCR verification
4. Output: verification report with pass/fail per criterion + confidence scores

### 3.3 Refinement Stage (optional)

If verification fails:
- Re-generate with adjusted prompt (emphasize missing elements)
- Optionally use inpainting to fix specific regions
- Return both original and refined images

File: `python-service/quality/vlm_evaluator.py`

---

## 4. GENERATION CONTROLS (TASK 4)

| Control | Model Support | Implementation | Priority |
|---------|--------------|----------------|----------|
| IP-Adapter | FLUX.2-klein, Z-Image | Diffusers `load_ip_adapter` | High (character consistency) |
| ControlNet | FLUX.2-klein, Z-Image | Diffusers ControlNet pipeline | High (composition) |
| Depth conditioning | All DiT models | ControlNet depth | Medium |
| Pose conditioning | All DiT models | ControlNet openpose | Medium |
| Reference-image conditioning | Z-Image img2img, FLUX.2-klein | Img2Img + IP-Adapter | High |
| LoRA | All DiT models | `load_lora_weights` | Medium |
| Inpainting | Z-Image, FLUX.2-klein | Inpainting pipeline | Medium |
| Outpainting | Z-Image, FLUX.2-klein | Outpainting pipeline | Low |
| SAM 2 | All (post-process) | `facebook/sam2-hiera-base-plus` | Low |
| Real-ESRGAN | All (post-process) | 4x upscaling | Medium |

**Character consistency:** IP-Adapter + reference image conditioning. Do not promise perfect identity preservation.

File: `python-service/adapters/` (each adapter implements controls it supports)

---

## 5. COMPUTE ENVIRONMENT (TASK 5)

### 5.1 Environment Strategy

| Environment | Use | Models |
|-------------|-----|--------|
| Codespace (CPU-only) | Development, TypeScript, SDXS-512 fallback | SDXS-512 (ONNX) |
| Kaggle GPU (T4/P100 16GB) | Benchmarks, controlled experiments | Z-Image-Turbo, FLUX.2-klein, MOSS-Image |
| Production GPU (24-48GB) | Qwen-Image-2512, multi-model concurrent | All models |

### 5.2 Benchmark Script Requirements

Each benchmark script must:
1. **Detect GPU architecture** (name, VRAM per device, compute capability)
2. **Select precision:** BF16 on Ampere+ (CC 8.0+), FP16 on T4 (Turing CC 7.5) / P100 (Pascal CC 6.0)
3. **Record all precision settings** in results JSON
4. **Measure per-GPU VRAM** (no combined VRAM assumption)
5. **Measure generation time** per image
6. **Record success/failure** per prompt
7. **Smoke test first** — abort full benchmark if smoke test fails
8. **Save images + results JSON + review records + contact sheet**

### 5.3 Official Configurations

- **Z-Image-Turbo:** 8 steps (9 scheduler → 8 DiT), guidance=0.0, 1024×1024
- **FLUX.2-klein-4B:** 4 steps, guidance=1.0, 1024×1024
- **Z-Image base:** 50 steps, CFG, negative prompts, 1024×1024
- **Qwen-Image-2512:** 1-2 steps (distilled), up to 1664×1664, 24-48GB VRAM required
- **MOSS-Image-SFT:** 50 steps, CFG=3, uncond_skip_layers=[22], 1024×1024

---

## 6. BENCHMARK FRAMEWORK (TASK 6)

### 6.1 Prompt Suite

33 prompts in `benchmark_framework.py` across 20+ categories, each with:
- Expected subjects, object counts, colors
- Camera angle, lighting, environment
- Used for structured prompt-fidelity review

### 6.2 Models Benchmarked

1. SDXS-512 (baseline, CPU)
2. Z-Image-Turbo (Kaggle)
3. FLUX.2-klein-4B (Kaggle)
4. Z-Image base (Kaggle, if VRAM permits)
5. Qwen-Image-2512 (production GPU only, not Kaggle)

### 6.3 Metrics

- Generation success rate
- Model loading time
- Actual generation time
- Peak VRAM per GPU
- Peak system RAM
- Image resolution
- Prompt fidelity (human 0-1 scores)
- Object-count correctness
- Composition correctness
- Photorealism
- Fine detail
- Human preference

### 6.4 Outputs

- Generated images (PNG)
- Contact sheets (side-by-side comparison)
- Results JSON (machine-readable)
- Review records JSON (structured human evaluation)
- Human evaluation CSV + HTML (anonymous labels Model A/B/C)
- Comparison report markdown

### 6.5 Rule: No Fabricated Results

All scores must come from actual benchmark runs or human evaluation. No theoretical numbers in final reports.

---

## 7. PYTHON INFERENCE SERVICE (TASK 7)

### 7.1 Architecture

```
Node.js/Express (port 3000) ← HTTP → Python FastAPI (port 8100)
```

### 7.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /v1/health` | GET | Service health, GPU status, loaded models |
| `GET /v1/models` | GET | List available models with metadata |
| `POST /v1/generate` | POST | Generate image with specified model |
| `POST /v1/prompt/parse` | POST | Parse prompt into structured scene spec |
| `POST /v1/quality/verify` | POST | Verify generated image against prompt |
| `POST /v1/upscale` | POST | Real-ESRGAN upscaling |
| `DELETE /v1/models/{id}` | DELETE | Unload model to free VRAM |

### 7.3 Features

- Model registry (JSON config)
- Lazy model loading/unloading
- Generation queue (asyncio)
- Progress tracking (SSE or polling)
- Cancellation support
- Error handling with fallback to SDXS-512
- Memory management (VRAM tracking)
- Model caching (disk)
- Stable HTTP API
- Image output as base64 or URL

### 7.4 Node.js Integration

Files to create:
- `src/engine/pythonDiffusionClient.ts` — HTTP client
- `src/engine/generationRouter.ts` — route to SDXS or Python service

Files to modify:
- `src/types/studio.ts` — add `python-*` engine IDs
- `src/data/imageEngines.ts` — add new engine specs
- `server.ts` — add Python client, modify `/api/media/image`, extend `/api/health`

### 7.5 Fallback Behavior

- Python service unreachable → fall back to `local-sdxs`
- Model load failure → fall back to `local-sdxs`
- VRAM OOM → return error, suggest smaller model
- Prompt parsing failure → use original prompt unchanged

---

## 8. IMPLEMENTATION ROADMAP

### Phase 1: Benchmark Infrastructure (Week 1) — COMPLETE
- [x] Corrected Kaggle benchmark scripts (Z-Image-Turbo, FLUX.2-klein-4B, MOSS-Image)
- [x] Benchmark framework with ReviewRecord, contact sheets, human eval
- [x] Aggregator with VRAM/RAM stats
- [x] Kaggle notebook instructions
- [x] Research reports corrected (theoretical markers)
- [ ] Add Qwen-Image-2512 and Z-Image-Base benchmark scripts
- [ ] Run benchmarks on Kaggle T4/P100
- [ ] Collect actual images and metrics

### Phase 2: Prompt Understanding (Week 2)
- [ ] Create `python-service/prompt/` directory
- [ ] Implement Qwen3-VL-4B or Qwen2.5-0.5B parser
- [ ] Structured scene spec JSON schema
- [ ] Faithful/creative modes
- [ ] `POST /v1/prompt/parse` endpoint

### Phase 3: Python Inference Service (Week 3-4)
- [ ] `python-service/main.py` FastAPI app
- [ ] `python-service/models.py` Pydantic schemas
- [ ] `python-service/engine_loader.py` lazy loading + VRAM management
- [ ] `python-service/adapters/base_adapter.py` abstract class
- [ ] `python-service/adapters/zimage_turbo.py`
- [ ] `python-service/adapters/flux_klein.py`
- [ ] `python-service/adapters/moss_sft.py`
- [ ] `python-service/adapters/sdxs_onnx.py` (fallback wrapper)
- [ ] `POST /v1/generate`, `GET /v1/health`, `GET /v1/models` endpoints
- [ ] Model queue, progress tracking, cancellation
- [ ] Model caching to disk

### Phase 4: Node.js Integration (Week 4-5)
- [ ] `src/engine/pythonDiffusionClient.ts`
- [ ] `src/engine/generationRouter.ts`
- [ ] Modify `server.ts` `/api/media/image` route
- [ ] Add new engine IDs to `src/types/studio.ts`
- [ ] Add engine specs to `src/data/imageEngines.ts`
- [ ] Extend `/api/health` with Python service status

### Phase 5: Frontend + Prompt Understanding UI (Week 5-6)
- [ ] Add "Prompt Understanding" toggle to GeneratePanel
- [ ] Faithful/creative mode selector
- [ ] Original vs optimized prompt display
- [ ] New engine options in dropdown
- [ ] Quality presets (already added)

### Phase 6: Quality Verification (Week 6-7)
- [ ] `python-service/quality/vlm_evaluator.py` (Qwen3-VL-4B)
- [ ] `python-service/quality/florence2.py` (object detection + OCR)
- [ ] `POST /v1/quality/verify` endpoint
- [ ] Optional targeted refinement stage

### Phase 7: Generation Controls (Week 7-8)
- [ ] IP-Adapter integration (character consistency)
- [ ] ControlNet integration (composition, depth, pose)
- [ ] LoRA loading
- [ ] Inpainting/outpainting
- [ ] Real-ESRGAN upscaling
- [ ] SAM 2 segmentation

### Phase 8: Regression Testing & Production Decision (Week 8)
- [ ] SDXS-512 regression test (txt2img, img2img, CFG)
- [ ] All 6 left-dock tabs functional
- [ ] Timeline, export, audio, subtitles all working
- [ ] Compare benchmark results
- [ ] Go/no-go decision: replace production engine or keep SDXS-512 primary

---

## 9. FILES TO CREATE

```
python-service/
├── main.py
├── models.py
├── engine_loader.py
├── requirements.txt
├── .env.example
├── adapters/
│   ├── base_adapter.py
│   ├── zimage_turbo.py
│   ├── flux_klein.py
│   ├── zimage_base.py
│   ├── qwen_2512.py
│   ├── moss_sft.py
│   └── sdxs_onnx.py
├── prompt/
│   ├── parser.py
│   └── schemas.py
├── quality/
│   ├── vlm_evaluator.py
│   └── florence2.py
├── upscale/
│   └── real_esrgan.py
└── routes/
    ├── generate.py
    ├── models.py
    ├── prompt.py
    ├── quality.py
    ├── upscale.py
    └── health.py

src/engine/
├── pythonDiffusionClient.ts
└── generationRouter.ts

kaggle_qwen_2512_benchmark.py
kaggle_zimage_base_benchmark.py
```

---

## 10. FILES TO MODIFY

| File | Change |
|------|--------|
| `src/types/studio.ts` | Add `python-zimage-turbo`, `python-flux-klein`, `python-zimage-base`, `python-qwen-2512`, `python-moss-sft` to `ImageEngineId` |
| `src/data/imageEngines.ts` | Add engine specs for new models with VRAM/metadata |
| `server.ts` | Add Python client, modify `/api/media/image`, extend `/api/health`, add `/v1/*` proxy routes |
| `src/components/GeneratePanel.tsx` | Add prompt understanding toggle, quality presets (done), new engine options |
| `benchmark_framework.py` | Add Qwen-Image-2512, Z-Image-Base to prompt suite if needed |
| `benchmark_aggregator.py` | Add new model result files |

---

## 11. IMMEDIATE NEXT STEPS

1. **Add Qwen-Image-2512 and Z-Image-Base benchmark scripts** (next action)
2. **Upload Kaggle notebooks** — create 5 notebooks (Z-Image-Turbo, FLUX.2-klein-4B, MOSS-Image, Z-Image-Base, SDXS-512 baseline)
3. **Run benchmarks** on Kaggle T4/P100 GPU
4. **Collect actual results** — images, metrics JSON, contact sheets
5. **Run aggregator** — `python benchmark_aggregator.py`
6. **Human evaluation** — score images using HTML review page
7. **Decide** — replace production engine only if results demonstrate measurable improvement

**Do not replace the production SDXS-512 engine until benchmark results are reviewed.**
