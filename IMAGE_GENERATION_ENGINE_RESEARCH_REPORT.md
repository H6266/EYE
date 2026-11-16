# AUTEUR STUDIO — OPEN-SOURCE IMAGE GENERATION ENGINE RESEARCH REPORT

**Status:** RESEARCH COMPLETE (Stage 1-8)  
**Date:** 2026-11-16  
**Hardware Environment:** GitHub Codespace (CPU-only, 7.8GB RAM, 15GB disk available) — all model experiments must run on Kaggle GPU (T4x2 16GB VRAM per GPU or P100 16GB)

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

### 2.2 Candidate Comparison Table

| Model | Vendor | Params | License | VRAM (BF16) | VRAM (Quant) | Steps | Resolution | Commercial |
|-------|--------|--------|---------|-------------|--------------|-------|------------|------------|
| **Z-Image-Turbo** | Alibaba Tongyi-MAI | 6B | Apache-2.0 | ~16GB | ~8GB (FP8), ~6GB (GGUF Q4) | 8 | 1024×1024 | ✅ |
| **FLUX.2 [klein] 4B** | Black Forest Labs | 4B | Apache-2.0 | ~13GB | ~8GB (FP8/NVFP4) | 4 | up to 4MP | ✅ |
| **Qwen-Image-2512** | Alibaba Qwen | 20B | Apache-2.0 | ~57.7GB | ~22.8GB (FP8), ~11.4GB (INT4) | 50 | up to 1664×1664 | ✅ |
| **MOSS-Image-SFT** | OpenMOSS | 5B | Apache-2.0 | ~13GB | ~8GB (INT8), ~5GB (4-bit) | 50 | 1024×1024 | ✅ |
| **FLUX.1 [schnell]** | Black Forest Labs | 12B | Apache-2.0 | ~24GB | ~12GB (FP8), ~8GB (GGUF) | 1-4 | 1024×1024 | ✅ |
| **Lumina-Image 2.0** | Shanghai AI Lab | 2B | Apache-2.0 | ~9GB | ~5GB | ~10-20 | 1024×1024 | ✅ |
| **HiDream-O1-Image** | HiDream.ai | ~9B | MIT | ~16-34GB | TBD | TBD | TBD | ✅ |
| **ERNIE-Image-Turbo** | Baidu | 8B | Apache-2.0 | ~24GB | TBD | 8 | 1024×1024 | ✅ |
| **SDXS-512** (current) | skillsafeai | N/A (ONNX) | Apache-2.0 | N/A (CPU) | N/A | 1 | 512×512 | ✅ |

### 2.3 Model Details

#### Z-Image-Turbo (Tongyi-MAI/Alibaba)
- **HF:** https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- **GitHub:** https://github.com/Tongyi-MAI/Z-Image
- **License:** Apache-2.0 (weights + code)
- **Architecture:** 6B S3-DiT diffusion transformer, Qwen3 4B text encoder
- **Key strengths:** 8-step distilled, sub-second on H800, bilingual text rendering (EN+ZH), robust instruction adherence
- **VRAM:** Fits comfortably in 16GB consumer GPUs per model card
- **Kaggle notebooks:** Available (shihao520/z-image-turbo, pasgara/z-image-turbo) on T4x2/P100
- **Kaggle fit:** T4x2 16GB VRAM per GPU — exactly target range
- **Weaknesses:** No negative prompts, low variety between seeds, not for fine-tuning

#### FLUX.2 [klein] 4B (Black Forest Labs)
- **HF:** https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- **GitHub:** https://github.com/black-forest-labs/flux2
- **License:** Apache-2.0 (4B), FLUX NCL (9B)
- **Architecture:** 4B rectified flow transformer, Qwen3 4B text encoder
- **Key strengths:** Sub-second generation (4 steps), unified generation + editing + multi-reference in one model, FP8/NVFP4 quantized versions available
- **VRAM:** ~8-13GB depending on offload strategy
- **Kaggle:** Available via HF Kaggle link
- **Weaknesses:** No prompt upsampling (write detailed prompts), 9B variant is non-commercial

#### Qwen-Image-2512 (Alibaba Qwen)
- **HF:** https://huggingface.co/Qwen/Qwen-Image-2512
- **GitHub:** https://github.com/QwenLM/Qwen-Image
- **License:** Apache-2.0
- **Architecture:** 20B MMDiT, Qwen2.5-VL-7B text encoder
- **Key strengths:** Claims strongest open-source (10,000+ blind rounds on AI Arena), best human realism, best text rendering, 7 aspect ratios
- **VRAM:** 27.7GB min / 41.8GB recommended BF16; FP8 ~22.8GB; INT4 ~11.4GB
- **Kaggle:** Does NOT fit on free T4/P100 (16GB) even with FP8
- **Weaknesses:** Too heavy for consumer deployment, 50 steps slow, requires high-VRAM GPU

#### MOSS-Image-SFT (OpenMOSS)
- **HF:** https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
- **GitHub:** https://github.com/OpenMOSS/MOSS-Image
- **License:** Apache-2.0
- **Architecture:** 5B single-stream DiT, Qwen3.5-9B text encoder, FLUX.2 VAE
- **Key strengths:** 96.3 Mean-6 on LLTB (text rendering benchmark), JSON layout control, 4K token prompts, English+Chinese typography
- **VRAM:** ~3.65/3.85 GiB peak at 1024×1024 50 steps (measured); ~13GB BF16 resident
- **Kaggle notebook:** Available via HF Kaggle link (T4)
- **Weaknesses:** T2I only (no editing, no img2img), slow (50 steps), no reference-image conditioning

#### FLUX.1 [schnell] (Black Forest Labs)
- **HF:** https://huggingface.co/black-forest-labs/FLUX.1-schnell
- **License:** Apache-2.0
- **Key strengths:** 1-4 step distilled FLUX, proven track record, Apache-2.0
- **VRAM:** ~24GB fp16 / ~8GB quantized
- **Weaknesses:** Larger than Z-Image-Turbo, older architecture

---

## 3. HARDWARE REQUIREMENTS FOR PRODUCTION DEPLOYMENT

| Tier | GPU VRAM | System RAM | Recommended Models |
|------|----------|------------|-------------------|
| Entry | 8-12 GB | 16-32 GB | Z-Image-Turbo (INT8/GGUF), FLUX.2 klein 4B (FP8/NVFP4), MOSS-Image (INT8) |
| Standard | 16-24 GB | 32-64 GB | Z-Image-Turbo (BF16), FLUX.2 klein 4B (BF16), MOSS-Image (BF16) |
| High-End | 24-48 GB | 64+ GB | Qwen-Image-2512 (FP8/INT8), multiple models concurrent |

**Codespace constraint:** No GPU, 7.8GB RAM — cannot run any of these models locally. Must use Kaggle for experiments.

**Production recommendation:** Deploy via Python inference service with Docker + NVIDIA GPU. Keep SDXS-512 as CPU fallback for users without GPU.

---

## 4. LICENSING REPORT

| Model | License | Commercial Use | Derivatives | Attribution | Approved |
|-------|---------|---------------|-------------|-------------|----------|
| Z-Image-Turbo | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| FLUX.2 [klein] 4B | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| Qwen-Image-2512 | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| MOSS-Image-Base/SFT | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| FLUX.1 [schnell] | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| Lumina-Image 2.0 | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| HiDream-O1-Image | MIT | ✅ Yes | ✅ Yes | Required | ✅ |
| ERNIE-Image | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |
| SDXS-512 (current) | Apache-2.0 | ✅ Yes | ✅ Yes | Required | ✅ |

**All candidates are commercially permissive.** No GPL, Creative Commons NC, or restrictive licenses.

---

## 5. RECOMMENDATION

### Primary Recommendation: **Z-Image-Turbo** (Tongyi-MAI/Alibaba)

**Rationale:**
- Apache-2.0 license — fully commercial, no revenue cap
- 6B parameters — smallest of the strong candidates
- 8-step distilled — fastest generation among quality models
- Fits in 16GB VRAM — deployable on consumer hardware (RTX 3090/4070 era)
- Strong photorealism and instruction adherence per model card
- Bilingual text rendering (English + Chinese)
- Kaggle-compatible (T4x2/P100 16GB VRAM)
- Multiple quantized versions (FP8, GGUF Q4) for lower-VRAM deployment
- Active Kaggle notebooks available for benchmarking

### Secondary Recommendation: **FLUX.2 [klein] 4B** (Black Forest Labs)

**Rationale:**
- Apache-2.0 license — fully commercial
- 4B parameters — smallest parameter count
- Sub-second generation — fastest inference
- Unified generation + editing + multi-reference in one model
- FP8/NVFP4 quantized versions from NVIDIA collaboration
- ~8-13GB VRAM range — consumer-deployable
- Slightly larger weights than Z-Image-Turbo but faster distilled inference

### For Future Consideration (when higher-VRAM hardware available):
- **Qwen-Image-2512** — best quality but requires 24-48GB VRAM
- **MOSS-Image-SFT** — best text rendering/layout but T2I only, no editing

---

## 6. NEXT STEPS (STAGE 5-10)

Before modifying production code:

1. **Create Kaggle benchmark notebooks** for Z-Image-Turbo, FLUX.2 Klein 4B, MOSS-Image
2. **Create 30+ prompt accuracy benchmark** with evaluation criteria
3. **Run benchmarks** on Kaggle T4x2 (16GB VRAM)
4. **Compare results** against SDXS-512 baseline
5. **Present results** before any production integration

**No production files will be modified until benchmark results are reviewed.**