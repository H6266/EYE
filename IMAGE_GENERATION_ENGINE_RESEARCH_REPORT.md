# AUTEUR STUDIO — OPEN-SOURCE IMAGE GENERATION ENGINE RESEARCH REPORT

> Historical pre-benchmark research snapshot. Use [IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md](./IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md) for the current model shortlist, repository audit, and next steps.

**Status:** RESEARCH COMPLETE (Stage 1-8) — PRE-BENCHMARK  
**Date:** 2026-11-16  
**Hardware Environment:** GitHub Codespace (CPU-only, 7.8GB RAM, 15GB disk available) — all model experiments must run on Kaggle GPU (T4x2 16GB VRAM per GPU or P100 16GB)

**BENCHMARK STATUS:** Not yet executed. No GPU available in this environment. All model rankings and VRAM claims below are theoretical estimates from official model cards, NOT measured results. No model winner can be declared until Kaggle benchmarks actually run.

---

## 1. EXISTING ARCHITECTURE AUDIT

### 1.1 Current SDXS-512 Pipeline (`src/engine/localDiffusionPipeline.ts`)
```
CLIP BPE Tokenizer (vocab.json + merges.txt)
    ↓
CLIP ViT-L/14 Text Encoder (text_encoder.onnx) → [1, 77, 768] embeddings
    ↓
SDXS-512 UNet Brain (unet.onnx) → 1-step Euler denoising at σ=14.6
    ↓
TAESD VAE Decoder (vae_decoder.onnx) → 512×512 RGB
```
- **Runtime:** onnxruntime-node (CPU-only)
- **Weights:** ~500MB, auto-downloaded from `skillsafeai/sdxs-512-dreamshaper-onnx-webgpu`
- **Speed:** ~2.5s/image on CPU
- **Resolution:** Fixed 512×512 latent → cropped (512×288 16:9, 288×512 9:16, 512×512 1:1)
- **Modes:** `local-sdxs` (CFG=0), `local-sdxs-cfg` (CFG 1.05-1.35), `local-sdxs-img2img`
- **License:** Apache-2.0

### 1.2 API Endpoints (`server.ts`)
| Endpoint | Model | Purpose |
|----------|-------|---------|
| `POST /api/prompts/enhance` | Gemini 4 Hydrogen | Prompt enhancement + audio transcription |
| `POST /api/media/image` | SDXS-512 ONNX | Local text-to-image / img2img |
| `POST /api/media/video` | Gemini Omni Flash | Video generation |
| `POST /api/media/music` | Lyria 3 | Music generation |
| `POST /api/media/voice` | Gemini TTS | Voice acting |
| `POST /api/subtitles/generate` | Gemini 4 Hydrogen / 3.5 Transcribe | Subtitle cues |

### 1.3 Gap Analysis
| Capability | Current State | Gap |
|------------|---------------|-----|
| Multi-model support | SDXS-512 only | Need adapter pattern + Python inference service |
| FLUX/Qwen/MOSS/Z-Image | Not integrated | Need PyTorch/Diffusers inference service |
| Local VLM prompt understanding | Gemini-only (cloud) | Need local VLM for structured scene parsing |
| ControlNet / IP-Adapter / LoRA | Not supported | Need Diffusers pipeline integration |
| Character consistency | VAE Img2Img (weak) | Need IP-Adapter + LoRA identity preservation |
| Long-prompt adherence | CLIP 77-token truncation | Need models with 4K+ token context |
| Hardware-aware selection | Fixed SDXS only | Need GPU detection, VRAM estimation |

---

## 2. CANDIDATE MODEL RESEARCH

### 2.1 Evaluation Criteria
| Criterion | Weight |
|-----------|--------|
| Image quality / realism | 25% |
| Prompt adherence / accuracy | 25% |
| Generation speed | 15% |
| VRAM / memory consumption | 15% |
| Commercial licensing | 10% |
| Ease of integration | 10% |

### 2.2 Candidate Comparison Table (THEORETICAL — from model cards, not measured)

| Model | Vendor | Params | License | VRAM (BF16/FP16) | VRAM (Quant) | Steps | Resolution | Commercial |
|-------|--------|--------|---------|------------------|--------------|-------|------------|------------|
| **Z-Image-Turbo** | Alibaba Tongyi-MAI | 6B | Apache-2.0 | ~12-16GB | ~8GB (FP8), ~6GB (GGUF Q4) | 8 (official baseline) | 1024×1024 | ✅ |
| **FLUX.2 [klein] 4B** | Black Forest Labs | 4B | Apache-2.0 | ~8.4-13GB | ~5GB (FP8), ~3.8GB (NVFP4) | 4 (official baseline) | up to 4MP | ✅ |
| **Qwen-Image** | Alibaba Qwen | ~9B (est.) | Apache-2.0 | ~27.7GB min | ~22.8GB (FP8), ~11.4GB (INT4) | 50 | up to 1664×1664 | ✅ |
| **MOSS-Image-SFT** | OpenMOSS | 5B | Apache-2.0 | ~13GB | ~8GB (INT8), ~5GB (4-bit) | 50 | 1024×1024 | ✅ |
| **SDXS-512** (current) | skillsafeai | N/A (ONNX) | Apache-2.0 | N/A (CPU) | N/A | 1 | 512×512 | ✅ |

**Notes on VRAM:** Figures are manufacturer/model-card estimates. Actual measured VRAM will be captured by the benchmark scripts (`vram_peak_gb` per image). T4 (Turing CC 7.5) and P100 (Pascal CC 6.0) do NOT support BF16 — scripts auto-detect and use FP16 on those GPUs.

### 2.3 Model Details

#### Z-Image-Turbo (Tongyi-MAI/Alibaba)
- **HF:** https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- **GitHub:** https://github.com/Tongyi-MAI/Z-Image
- **License:** Apache-2.0 (weights + code)
- **Architecture:** 6B S3-DiT diffusion transformer, Qwen3 4B text encoder
- **Key strengths:** 8-step distilled (Decoupled-DMD), sub-second on H800, bilingual text rendering (EN+ZH), robust instruction adherence
- **VRAM:** Fits comfortably in 16GB consumer GPUs per model card
- **Official baseline:** `num_inference_steps=8`, `guidance_scale=0.0` (HF reference uses 9 steps → 8 DiT forwards; benchmark uses 8 as baseline)
- **Kaggle fit:** T4x2/P100 16GB VRAM — target range
- **Weaknesses:** No negative prompts, low variety between seeds, text-to-image only (no edit mode)
- **Pipeline:** `DiffusionPipeline` (ZImagePipeline in diffusers)

#### FLUX.2 [klein] 4B (Black Forest Labs)
- **HF:** https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- **GitHub:** https://github.com/black-forest-labs/flux2
- **License:** Apache-2.0 (4B)
- **Architecture:** 4B rectified flow transformer, Qwen3 4B text encoder
- **Key strengths:** 4-step distilled, sub-second generation, unified generation + editing + multi-reference, FP8/NVFP4 quantized versions (NVIDIA collaboration)
- **VRAM:** ~8.4GB distilled (bfl.ai), ~13GB (HF model card)
- **Official baseline:** `num_inference_steps=4`, `guidance_scale=1.0`
- **Pipeline:** `Flux2KleinPipeline` (requires `pip install git+https://github.com/huggingface/diffusers.git`)
- **Weaknesses:** No prompt upsampling (write detailed prompts), 9B variant is non-commercial

#### Qwen-Image (Alibaba Qwen)
- **HF:** https://huggingface.co/Qwen/Qwen-Image
- **GitHub:** https://github.com/QwenLM/Qwen-Image
- **License:** Apache-2.0
- **Architecture:** MMDiT, Qwen2.5-VL text encoder
- **Key strengths:** Strong open-source performance, text rendering, 7 aspect ratios
- **VRAM:** 27.7GB min / 41.8GB recommended BF16; FP8 ~22.8GB; INT4 ~11.4GB
- **Kaggle:** Does NOT fit on free T4/P100 (16GB) even with FP8
- **Status:** NOT in benchmark scope (excluded due to VRAM requirements)

#### MOSS-Image-SFT (OpenMOSS)
- **HF:** https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
- **GitHub:** https://github.com/OpenMOSS/MOSS-Image
- **License:** Apache-2.0
- **Architecture:** 5B single-stream DiT, Qwen3.5-9B text encoder, FLUX.2 VAE
- **Key strengths:** 96.3 Mean-6 on LLTB (text rendering benchmark), JSON layout control, 4K token prompts, English+Chinese typography
- **VRAM:** ~3.65/3.85 GiB peak at 1024×1024 50 steps (measured per model card); ~13GB BF16 resident
- **Official baseline:** `num_inference_steps=50`, `guidance_scale=3.0`, `uncond_skip_layers=[22]`, shift=5
- **Pipeline:** `DiffusionPipeline` with `custom_pipeline="pipeline_moss_image"`, `trust_remote_code=True`
- **Requires:** `diffusers>=0.38`, `transformers>=5.3`
- **Weaknesses:** T2I only (no editing, no img2img), slow (50 steps), no reference-image conditioning
- **Kaggle notebook:** Available via HF Kaggle link (T4)

---

## 3. HARDWARE REQUIREMENTS FOR PRODUCTION DEPLOYMENT

| Tier | GPU VRAM | System RAM | Recommended Models |
|------|----------|------------|-------------------|
| Entry | 8-12 GB | 16-32 GB | Z-Image-Turbo (INT8/GGUF), FLUX.2 klein 4B (FP8/NVFP4), MOSS-Image (INT8) |
| Standard | 16-24 GB | 32-64 GB | Z-Image-Turbo (BF16/FP16), FLUX.2 klein 4B (BF16/FP16), MOSS-Image (BF16/FP16) |
| High-End | 24-48 GB | 64+ GB | Qwen-Image (FP8/INT8), multiple models concurrent |

**Codespace constraint:** No GPU, 7.8GB RAM — cannot run any of these models locally. Must use Kaggle for experiments.

**Production recommendation:** Deploy via Python inference service with Docker + NVIDIA GPU. Keep SDXS-512 as CPU fallback for users without GPU.

**IMPORTANT:** Kaggle is a testing environment, not a permanent production inference server. Do not design Auteur Studio to depend on an expiring Kaggle notebook URL for normal image generation.

---

## 4. LICENSING REPORT

| Model | License | Commercial Use | Derivatives | Attribution | Approved |
|-------|---------|---------------|-------------|-------------|----------|
| Z-Image-Turbo | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| FLUX.2 [klein] 4B | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| Qwen-Image | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| MOSS-Image-Base/SFT | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| SDXS-512 (current) | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |

**All candidates are commercially permissive.** No GPL, Creative Commons NC, or restrictive licenses.

---

## 5. RECOMMENDATION (THEORETICAL — PRE-BENCHMARK)

**No benchmark has been run.** The following is a theoretical ranking based on model card specifications only. It MUST be validated against actual benchmark results before any production decision.

### Primary Candidate (theoretical): **Z-Image-Turbo** (Tongyi-MAI/Alibaba)

**Theoretical rationale:**
- Apache-2.0 license — fully commercial, no revenue cap
- 6B parameters — smallest of the strong candidates
- 8-step distilled — fastest generation among quality models
- Fits in 16GB VRAM — deployable on consumer hardware
- Strong photorealism and instruction adherence per model card
- Bilingual text rendering (English + Chinese)
- Kaggle-compatible (T4x2/P100 16GB VRAM)

### Secondary Candidate (theoretical): **FLUX.2 [klein] 4B** (Black Forest Labs)

**Theoretical rationale:**
- Apache-2.0 license — fully commercial
- 4B parameters — smallest parameter count
- 4-step distilled — fastest inference
- Unified generation + editing + multi-reference in one model
- FP8/NVFP4 quantized versions from NVIDIA collaboration
- ~8.4-13GB VRAM range — consumer-deployable

### Tertiary Candidate (theoretical): **MOSS-Image-SFT** (OpenMOSS)

**Theoretical rationale:**
- Apache-2.0 license — fully commercial
- 5B parameters
- Best text rendering/layout (96.3 Mean-6 on LLTB)
- JSON layout control, 4K token prompts
- **Drawback:** 50 steps is too slow for interactive use; T2I only

### Not recommended for this hardware tier:
- **Qwen-Image** — requires 24-48GB VRAM, does not fit in 16GB even quantized

---

## 6. NEXT STEPS (STAGE 5-10)

Before modifying production code:

1. **Upload Kaggle benchmark notebooks** — Create 3 Kaggle notebooks from the Python benchmark scripts, run on T4x2/P100 GPU
2. **Run benchmarks** — Execute `kaggle_zimage_turbo_benchmark.py`, `kaggle_flux_klein_benchmark.py`, `kaggle_moss_benchmark.py`
3. **Collect real results** — Save generated images, metrics JSON, review records, contact sheets from each model
4. **Run SDXS-512 baseline** — Generate same 33 prompts on existing SDXS-512 for comparison
5. **Run aggregator** — Combine results with `benchmark_aggregator.py`
6. **Human evaluation** — Score images on the 8 evaluation criteria using the HTML review page
7. **Decide** — Confirm or adjust model selection based on REAL measured data
8. **Implement** — Build Python service, Node.js client, frontend changes per integration plan (only after benchmark review)

**No production files will be modified until benchmark results are reviewed.**

**Do not proceed to production model integration until benchmark results are available for review.**
