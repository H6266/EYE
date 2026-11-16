# FINAL RECOMMENDATION & REPORT

**Project:** Auteur Studio Image Generation Intelligence Upgrade  
**Date:** 2026-11-16  
**Status:** PRE-BENCHMARK — AWAITING KAGGLE GPU EXECUTION  
**WARNING:** No benchmarks have been run. No model winner is declared. All rankings below are theoretical estimates from official model cards.

---

## 1. EXECUTIVE SUMMARY

Auteur Studio currently uses SDXS-512 (ONNX CPU, ~2.5s/image, 512px) as its sole image generation engine. SDXS-512 is fast and dependency-free but struggles with complex, detailed natural-language prompts — especially multi-subject compositions, specific object counts, precise spatial positioning, and fine-grained attribute control.

This upgrade evaluates three lightweight open-weight models as potential replacements/augmentations. All models are Apache-2.0 licensed (commercial-use approved) and target Kaggle GPU (T4x2/P100, 16GB VRAM).

**No model is recommended as the winner yet.** Benchmarks must run on Kaggle GPU first. The table below shows theoretical candidates only.

| Tier | Model | Role (theoretical) |
|------|-------|-------------------|
| Candidate 1 | **Z-Image-Turbo** (6B, Apache-2.0) | Default for general use; 8 distilled steps, ~1024px |
| Candidate 2 | **FLUX.2 klein 4B** (4B, Apache-2.0) | Fast fallback; 4 distilled steps, ~1024px |
| Candidate 3 | **MOSS-Image-SFT** (5B, Apache-2.0) | Tertiary option; 50 steps, strong text/layout |
| Baseline (keep) | **SDXS-512** (ONNX CPU) | Local fallback when Python service unavailable; ~2.5s/image, 512px |
| Not in scope | **Qwen-Image** | Requires 24-48GB VRAM, does not fit 16GB Kaggle |

---

## 2. MODEL COMPARISON (THEORETICAL — FROM MODEL CARDS, NOT MEASURED)

| Criterion | Z-Image-Turbo | FLUX.2 klein 4B | MOSS-Image-SFT | SDXS-512 (baseline) |
|-----------|--------------|-----------------|----------------|---------------------|
| Parameters | 6B | 4B | 5B | N/A (ONNX) |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| VRAM (est.) | ~12-16GB | ~8.4-13GB | ~13GB | ~2GB (CPU RAM) |
| Steps (official) | 8 | 4 | 50 | 1 |
| Resolution | 1024×1024 | up to 4MP | 1024×1024 | 512×512 |
| Gen time (est., T4) | ~6-8s | ~3-4s | ~45-60s | ~2.5s (CPU) |
| Prompt fidelity | Strong (per card) | Strong (per card) | Strong (per card) | Moderate |
| Multi-subject | Strong | Strong | Moderate | Weak |
| Object counts | Strong | Moderate | Moderate | Weak |
| Spatial position | Strong | Strong | Moderate | Weak |
| Commercial use | Yes | Yes | Yes | Yes |
| Fits 16GB VRAM | Yes | Yes | Yes | N/A (CPU) |

**All VRAM, speed, and quality figures above are manufacturer/model-card estimates.** Actual measured values will come from the Kaggle benchmarks.

---

## 3. BENCHMARK FRAMEWORK

### 3.1 Prompt Suite (33 prompts, 20 categories)

Located in `benchmark_framework.py`. Categories include human portraits, complex clothing, animal wildlife, architecture, landscape, multiple people, specific object count, specific object position, complex camera angle, atmospheric conditions, styles (anime, cartoon, 3D, watercolor, fantasy), and complex multi-requirement.

Each prompt includes: expected subjects, expected object counts, expected colors, camera angle, lighting, environment — used for structured prompt-fidelity review.

### 3.2 Evaluation Criteria

| Criterion | Weight | Method |
|-----------|--------|--------|
| Subject presence | 20% | Human: all requested subjects present |
| Object count accuracy | 15% | Human count vs requested count |
| Color accuracy | 15% | Human: specified colors rendered correctly |
| Composition / position | 15% | Human: objects in requested positions |
| Lighting / atmosphere | 10% | Human: mood and lighting match |
| Environment / background | 10% | Human: setting matches description |
| No unwanted elements | 10% | Human: no hallucinated objects/people |
| Overall accuracy | 5% | Human holistic 0-1 score |

### 3.3 Benchmark Scripts (Kaggle)

- `kaggle_zimage_turbo_benchmark.py` — Z-Image-Turbo inference on T4/P100
- `kaggle_flux_klein_benchmark.py` — FLUX.2 klein 4B inference on T4/P100
- `kaggle_moss_benchmark.py` — MOSS-Image-SFT inference on T4/P100
- `benchmark_aggregator.py` — Combine results into comparison report
- `KAGGLE_BENCHMARK_NOTEBOOKS.md` — Step-by-step notebook instructions

Each notebook:
1. Detects GPU architecture and selects BF16 (Ampere+) or FP16 (T4/P100)
2. Reports per-GPU VRAM (no combined VRAM assumption)
3. Loads model with `enable_model_cpu_offload()` + attention slicing
4. Runs a smoke test (1 image) — aborts full benchmark if smoke test fails
5. Generates all 33 prompts at 1024×1024 with fixed seed 42
6. Tracks generation time, peak VRAM per GPU, peak system RAM
7. Saves images, results JSON, review records JSON, contact sheet PNG

### 3.4 Review Records

Each generated image gets a structured `ReviewRecord` containing:
- Original prompt, generated image path
- Expected subjects, object counts, colors, camera angle, lighting, environment
- Human evaluation fields (subject presence, object count, color, composition, lighting, environment, unwanted elements, overall accuracy) — filled by human evaluator, NOT auto-fabricated
- Missing details, incorrect details, unwanted additions — filled by human evaluator
- Optional VLM evaluation (clearly marked as NOT ground truth)

### 3.5 Human Evaluation Export

`benchmark_framework.py` generates:
- CSV: `benchmark_results/<model>_review_records.csv` — all fields for human scoring
- HTML: `benchmark_results/<model>_review_records.html` — visual review page with images
- Contact sheet: `benchmark_results/<model>_contact_sheet.png` — all images side by side
- Anonymous labels: Model A, Model B, Model C — reduces selection bias in human comparison

---

## 4. INTEGRATION PLAN (DEFERRED)

**No production files will be modified until benchmark results are reviewed.**

The Python inference service, Node.js client, and frontend changes described in `PYTHON_INFERENCE_ADAPTER_INTEGRATION_PLAN.md` are deferred pending benchmark outcomes. SDXS-512 remains the active production image engine throughout this task.

---

## 5. PROMPT UNDERSTANDING LAYER (DEFERRED)

The Qwen2.5-0.5B prompt understanding layer described in `PROMPT_UNDERSTANDING_SPEC.md` is deferred pending benchmark outcomes.

---

## 6. DEPLOYMENT CONSTRAINT

**Kaggle is a testing environment, not a permanent production inference server.**

After benchmarking, recommend a persistent GPU hosting architecture:
- Docker + NVIDIA GPU instance (AWS EC2 G4dn, G5, or equivalent)
- Python inference service (FastAPI) containerized
- Model weights cached in persistent volume
- Node.js Express app connects to Python service via HTTP
- SDXS-512 remains as local CPU fallback when Python service unavailable

Do not design Auteur Studio to depend on an expiring Kaggle notebook URL for normal image generation.

---

## 7. NEXT STEPS (IMMEDIATE)

1. **Upload Kaggle notebooks** — Create 3 Kaggle notebooks from the Python benchmark scripts
2. **Run benchmarks** — Execute all 3 model benchmarks on Kaggle T4x2/P100 GPU
3. **Collect real results** — Generated images, metrics JSON, review records, contact sheets
4. **Run SDXS-512 baseline** — Same 33 prompts for comparison
5. **Run aggregator** — `python benchmark_aggregator.py`
6. **Human evaluation** — Score images using HTML review page
7. **Decide** — Select model based on REAL measured evidence, not theory
8. **Implement** — Only after benchmark review and go/no-go decision

---

## 8. RISKS & MITIGATIONS

| Risk | Mitigation |
|------|------------|
| No GPU in Codespace | All model experiments run on Kaggle T4x2/P100 |
| Can't fabricate benchmark numbers | Real execution required; benchmarks not yet run |
| Qwen-Image doesn't fit 16GB VRAM | Excluded from benchmark scope |
| No model winner yet | Do not declare one until benchmarks run |
| Kaggle is ephemeral | Persistent GPU hosting recommended for production |
| Python service adds latency | SDXS-512 stays as instant local fallback |

---

## 9. BOTTOM LINE

**Z-Image-Turbo and FLUX.2 klein 4B are the leading theoretical candidates**, but neither can be declared the winner until Kaggle benchmarks actually run and produce measured data on prompt fidelity, generation speed, and VRAM consumption.

**Do not proceed to production model integration until benchmark results are available for review.**
