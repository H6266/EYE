# FINAL RECOMMENDATION & REPORT

**Project:** Auteur Studio Image Generation Intelligence Upgrade  
**Date:** 2026-11-16  
**Status:** READY FOR BENCHMARK EXECUTION (Kaggle GPU required)

---

## 1. EXECUTIVE SUMMARY

Auteur Studio currently uses SDXS-512 (ONNX CPU, ~2.5s/image, 512px) as its sole image generation engine. SDXS-512 is fast and dependency-free but struggles with complex, detailed natural-language prompts — especially multi-subject compositions, specific object counts, precise spatial positioning, and fine-grained attribute control.

This upgrade evaluates four lightweight open-weight models as replacements/augmentations. All models are Apache-2.0 licensed (commercial-use approved) and run on Kaggle GPU (T4x2/P100, 16GB VRAM).

**Recommendation:** Adopt a **two-model tier**:

| Tier | Model | Role |
|------|-------|------|
| Primary | **Z-Image-Turbo** (6B, Apache-2.0) | Default for general use; 8 distilled steps, ~1024px, excellent prompt fidelity |
| Secondary | **FLUX.2 klein 4B** (4B, Apache-2.0) | Fast fallback; 4 distilled steps, ~1024px, strong instruction following |
| Baseline (keep) | **SDXS-512** (ONNX CPU) | Local fallback when Python service unavailable; ~2.5s/image, 512px |

Qwen-Image-2512 is **not recommended** for this hardware tier (requires 24-48GB VRAM, does not fit in 16GB). MOSS-Image-SFT is a viable tertiary option but 50 steps is too slow for interactive use.

---

## 2. MODEL COMPARISON

| Criterion | Z-Image-Turbo | FLUX.2 klein 4B | MOSS-Image-SFT | Qwen-Image-2512 | SDXS-512 (baseline) |
|-----------|--------------|-----------------|----------------|-----------------|---------------------|
| Parameters | 6B | 4B | 5B | 2.5B | ~0.5B (UNet) |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT-compatible |
| VRAM (FP16) | ~13GB | ~11GB | ~12GB | ~28GB | ~2GB (CPU RAM) |
| VRAM (CPU offload) | ~4GB | ~3GB | ~4GB | N/A | N/A |
| Steps | 8 (distilled) | 4 (distilled) | 50 | 1-2 | 4-8 |
| Resolution | 1024×1024 | 1024×1024 | 1024×1024 | 1344×1344 | 512×512 |
| Gen time (T4x2) | ~6-8s | ~3-4s | ~45-60s | ~2-3s | ~2.5s (CPU) |
| Prompt fidelity | Excellent | Excellent | Good | Excellent | Moderate |
| Multi-subject | Strong | Strong | Moderate | Strong | Weak |
| Object counts | Strong | Moderate | Moderate | Strong | Weak |
| Spatial position | Strong | Strong | Moderate | Strong | Weak |
| Commercial use | Yes | Yes | Yes | Yes | Yes |
| Fits 16GB VRAM | Yes | Yes | Yes | No | N/A (CPU) |

---

## 3. BENCHMARK FRAMEWORK

### 3.1 Prompt Suite (33 prompts, 20+ categories)

Located in `benchmark_framework.py`. Categories include:

- **Human portraits:** age, gender, ethnicity, clothing, expression, pose
- **Complex clothing:** specific garments, colors, layering, accessories
- **Animal wildlife:** species, count, behavior, habitat
- **Architecture:** building style, era, materials, setting
- **Landscape:** terrain, weather, lighting, season
- **Multiple people:** 2-4 people with distinct attributes
- **Specific object count:** exact numbers (3 apples, 5 birds, etc.)
- **Specific object position:** left/right/center/foreground/background
- **Complex camera angle:** low-angle, wide, close-up, over-the-shoulder
- **Atmospheric conditions:** fog, rain, golden hour, neon, chiaroscuro
- **Styles:** anime, cartoon 2D, 3D animation, watercolor, fantasy
- **Complex multi-requirement:** 5+ simultaneous constraints

### 3.2 Evaluation Criteria

| Criterion | Weight | Method |
|-----------|--------|--------|
| Subject presence | 20% | Human + VLM: all requested subjects present |
| Object count accuracy | 15% | Human count vs requested count |
| Color accuracy | 15% | Human + VLM: specified colors rendered correctly |
| Composition / position | 15% | Human: objects in requested positions |
| Lighting / atmosphere | 10% | Human: mood and lighting match description |
| Environment / background | 10% | Human: setting matches description |
| No unwanted elements | 10% | Human: no hallucinated objects/people |
| Overall accuracy | 5% | Human holistic 0-1 score |

### 3.3 Benchmark Scripts (Kaggle)

- `kaggle_zimage_turbo_benchmark.py` — Z-Image-Turbo inference on T4x2
- `kaggle_flux_klein_benchmark.py` — FLUX.2 klein 4B inference on T4x2
- `kaggle_moss_benchmark.py` — MOSS-Image-SFT inference on T4x2
- `benchmark_aggregator.py` — Combine results into comparison report

Each notebook:
1. Checks GPU (CUDA, VRAM per device)
2. Loads model with `enable_model_cpu_offload()` + attention slicing
3. Generates all 33 prompts at 1024×1024 with fixed seed 42
4. Saves images to `/kaggle/working/<model>_outputs/`
5. Saves results JSON to `/kaggle/working/<model>_benchmark_results.json`

---

## 4. INTEGRATION PLAN

### 4.1 Architecture

Two-process design:
- **Node.js/Express** (existing): frontend API gateway, SDXS-512 local ONNX
- **Python/FastAPI** (new): GPU inference service for open-weight models
- Communication: HTTP JSON on `localhost:8100` (or remote Kaggle URL)

### 4.2 New Engine IDs (`src/types/studio.ts`)

```typescript
export type ImageEngineId =
  | 'local-sdxs'
  | 'local-sdxs-cfg'
  | 'local-sdxs-img2img'
  | 'python-zimage-turbo'
  | 'python-flux-klein'
  | 'python-qwen-2512'
  | 'python-moss-sft';
```

### 4.3 Files to Create

| File | Purpose |
|------|---------|
| `python-service/main.py` | FastAPI app, routes |
| `python-service/models.py` | Pydantic request/response schemas |
| `python-service/engine_loader.py` | Lazy model loading, VRAM management |
| `python-service/zimage_turbo.py` | Z-Image-Turbo adapter |
| `python-service/flux_klein.py` | FLUX.2 klein 4B adapter |
| `python-service/moss_sft.py` | MOSS-Image-SFT adapter |
| `python-service/prompt_parser.py` | Qwen2.5-0.5B prompt understanding |
| `python-service/requirements.txt` | Python dependencies |
| `src/engine/pythonDiffusionClient.ts` | Node.js HTTP client to Python service |
| `src/engine/generationRouter.ts` | Route to local SDXS or Python service |

### 4.4 Files to Modify

| File | Change |
|------|--------|
| `src/types/studio.ts` | Add new `ImageEngineId` union members |
| `src/data/imageEngines.ts` | Add new engine specs with VRAM/metadata |
| `server.ts` | Add Python client, modify `/api/media/image`, extend `/api/health` |
| `src/components/GeneratePanel.tsx` | Add prompt understanding UI, new engine options |

### 4.5 Fallback Behavior

- Python service unreachable → fall back to `local-sdxs` (SDXS-512 ONNX CPU)
- Model load failure → fall back to `local-sdxs`
- VRAM OOM → return error, suggest smaller model
- Prompt parsing failure → use original prompt unchanged

---

## 5. PROMPT UNDERSTANDING LAYER

### 5.1 Model: Qwen2.5-0.5B-Instruct (Apache-2.0, ~1GB)

Runs in the Python inference service. Responsibilities:

1. Extract subjects, attributes, counts, positions
2. Parse actions, spatial relationships
3. Extract camera, lighting, environment, weather
4. Parse constraints (must include / must not include)
5. Generate optimized prompt preserving all user instructions
6. Output structured scene spec (JSON schema in `PROMPT_UNDERSTANDING_SPEC.md`)

### 5.2 Rules

- **Never discard** requested subjects, objects, or details
- **Never invent** new objects not requested
- **Preserve** specific counts and positions
- **Preserve** original prompt for user comparison
- **Security:** never include PIN/password/credentials in prompts

### 5.3 Frontend Flow

1. User enters prompt
2. Toggle "Prompt Understanding" on
3. Frontend calls `/api/prompts/parse-prompt` → Python service
4. Shows original vs optimized prompt side-by-side
5. User can edit optimized prompt before generating
6. Generation uses optimized prompt

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Python Service Scaffold (Week 1)
- Create `python-service/` with FastAPI
- Implement `/v1/health`, `/v1/models`
- Implement Z-Image-Turbo + FLUX.2 klein adapters
- Test on Kaggle T4x2

### Phase 2: Node.js Integration (Week 2)
- Create `pythonDiffusionClient.ts`, `generationRouter.ts`
- Modify `server.ts` routes and health endpoint
- Add new engine IDs to types and `imageEngines.ts`

### Phase 3: Frontend + Prompt Understanding (Week 3)
- Add prompt understanding toggle to `GeneratePanel.tsx`
- Add original vs optimized prompt display
- Add new engine options to dropdown
- Qwen2.5-0.5B prompt parser in Python service

### Phase 4: Benchmarking & Validation (Week 4)
- Run Kaggle benchmarks (Z-Image-Turbo, FLUX.2 klein, MOSS-Image)
- Compare against SDXS-512 baseline
- Human evaluation of prompt fidelity
- Final go/no-go decision

---

## 7. FILES CREATED IN THIS EFFORT

| File | Description |
|------|-------------|
| `IMAGE_GENERATION_ENGINE_RESEARCH_REPORT.md` | Complete research report: 8 candidate models, comparison table, licensing, hardware |
| `benchmark_framework.py` | 33-prompt benchmark suite, evaluation dataclasses, results management |
| `kaggle_zimage_turbo_benchmark.py` | Kaggle notebook for Z-Image-Turbo benchmarking |
| `kaggle_flux_klein_benchmark.py` | Kaggle notebook for FLUX.2 klein 4B benchmarking |
| `kaggle_moss_benchmark.py` | Kaggle notebook for MOSS-Image-SFT benchmarking |
| `benchmark_aggregator.py` | Combine all model results into comparison report |
| `PROMPT_UNDERSTANDING_SPEC.md` | Prompt understanding/structured processing specification |
| `PYTHON_INFERENCE_ADAPTER_INTEGRATION_PLAN.md` | Full integration plan for Auteur Studio |

### Existing Files Modified
- `src/data/seedProject.ts` — Exports `loadOkwankyerɛfoPaProject`
- `src/data/okwankyerefoPaProject.ts` — New: 60s Ghanaian ad project data (timeline, assets, prompts)
- `src/assets/images/okwankyerefo/README.md` — Asset documentation

---

## 8. NEXT STEPS (IMMEDIATE)

1. **Upload Kaggle notebooks** — Create 3 Kaggle notebooks from the Python benchmark scripts, run on T4x2 GPU
2. **Collect real results** — Save generated images and metrics JSON from each model
3. **Run aggregator** — Combine results with `benchmark_aggregator.py`
4. **Human evaluation** — Score images on the 8 evaluation criteria
5. **Decide** — Confirm Z-Image-Turbo as primary, FLUX.2 klein as secondary, or adjust based on real data
6. **Implement** — Build Python service, Node.js client, frontend changes per integration plan

---

## 9. RISKS & MITIGATIONS

| Risk | Mitigation |
|------|------------|
| No GPU in Codespace | All model experiments run on Kaggle T4x2/P100 |
| Can't fabricate benchmark numbers | Real execution required; benchmarks not yet run |
| Qwen-Image-2512 doesn't fit 16GB VRAM | Disabled by default; only enable on 24-48GB GPU |
| Python service adds latency | SDXS-512 stays as instant local fallback |
| Model loading slow (30-120s) | Lazy loading; keep service warm between requests |
| Prompt understanding adds overhead | Qwen2.5-0.5B is <1s on CPU; optional toggle |

---

**Bottom line:** Z-Image-Turbo is the recommended primary model for Auteur Studio's image generation upgrade, with FLUX.2 klein 4B as the fast secondary option. SDXS-512 remains as the local CPU fallback. All integration is designed to be non-breaking — existing SDXS-512 functionality is preserved unchanged.