# KAGGLE BENCHMARK NOTEBOOK INSTRUCTIONS

> Historical notebook instructions. Preflight the scripts and dependencies against [IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md](./IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md) before treating their measurements as valid.

This document describes how to run the Auteur Studio image-generation benchmarks on Kaggle GPU notebooks.

## Overview

Three model-specific benchmark scripts are provided, plus a shared framework:

| Script | Model | HF Repo ID | Baseline Steps | Guidance |
|--------|-------|------------|----------------|----------|
| `kaggle_zimage_turbo_benchmark.py` | Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | 8 | 0.0 |
| `kaggle_flux_klein_benchmark.py` | FLUX.2 [klein] 4B | `black-forest-labs/FLUX.2-klein-4B` | 4 | 1.0 |
| `kaggle_moss_benchmark.py` | MOSS-Image-SFT | `OpenMOSS-Team/MOSS-Image-SFT` | 50 | 3.0 |

Shared framework: `benchmark_framework.py` (prompts, review records, contact sheets, human evaluation).

## Hardware Requirements

Run on Kaggle with:
- **Accelerator:** GPU T4 (1x, 16GB VRAM) or P100 (16GB VRAM)
- **Internet:** On (required to download model weights from Hugging Face)
- **Persistence:** On (recommended, so outputs survive session restart)

**Important:** Each script loads ONE model at a time. Do not run two benchmark scripts simultaneously — each model needs ~11-16GB VRAM and two models will OOM.

**Multi-GPU note:** Kaggle T4x2 provides 2x T4 GPUs with 16GB VRAM each. VRAM is NOT pooled — each GPU has independent 16GB. These scripts use single-GPU placement only (`enable_model_cpu_offload()` + `.to("cuda")`).

## Step 1: Create the Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Set **Accelerator:** GPU → T4 (or P100)
4. Set **Internet:** On
5. Set **Persistence:** On

## Step 2: Add Files

Upload these 4 files as notebook assets (or paste as cells):

1. `benchmark_framework.py`
2. `kaggle_zimage_turbo_benchmark.py`
3. `kaggle_flux_klein_benchmark.py`
4. `kaggle_moss_benchmark.py`

If pasting as cells, put `benchmark_framework.py` first (it defines shared types), then each model script.

## Step 3: Install Dependencies

Run this in the first cell (Kaggle has PyTorch pre-installed, but diffusers may need upgrading for `Flux2KleinPipeline`):

```python
# Cell 1: Install dependencies
!pip install -U diffusers transformers accelerate safetensors huggingface-hub pillow psutil
# FLUX.2-klein-4B requires diffusers from GitHub main for Flux2KleinPipeline
!pip install git+https://github.com/huggingface/diffusers.git
```

Verify versions after install:

```python
# Cell 2: Verify environment
import torch, diffusers, transformers
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"Diffusers: {diffusers.__version__}")
print(f"Transformers: {transformers.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB VRAM, CC {props.major}.{props.minor}")
```

## Step 4: Run Each Model Benchmark

Run each script as a separate cell block. Example for Z-Image-Turbo:

```python
# Cell 3: Z-Image-Turbo benchmark
from kaggle_zimage_turbo_benchmark import main
main()
```

Then FLUX.2-klein-4B:

```python
# Cell 4: FLUX.2-klein-4B benchmark
from kaggle_flux_klein_benchmark import main
main()
```

Then MOSS-Image-SFT:

```python
# Cell 5: MOSS-Image-SFT benchmark
from kaggle_moss_benchmark import main
main()
```

### What each benchmark does (in order)

1. **GPU detection** — Reports per-GPU name, VRAM, compute capability, BF16 support. Selects BF16 on Ampere+ (CC 8.0+), FP16 on T4/P100.
2. **Model loading** — Loads the model with hardware-appropriate precision and `enable_model_cpu_offload()` for 16GB VRAM safety.
3. **Smoke test** — Generates ONE test image (`"A red apple on a wooden table..."`) to verify the pipeline works.
4. **Abort on smoke test failure** — If the smoke test fails, the full benchmark is NOT run. This prevents wasting time on broken setups.
5. **Full benchmark** — Generates all 33 prompts at 1024×1024 with fixed seed 42, tracking:
   - Generation time per image
   - Peak VRAM per GPU (per-image)
   - Peak system RAM
   - Success/failure per prompt
6. **Outputs saved** — Images, results JSON, review records JSON, contact sheet PNG.

## Step 5: Download Outputs

After all benchmarks complete, download from the Kaggle notebook output panel:

```
zimage_outputs/          # Z-Image-Turbo generated images (33 PNGs + smoke_test.png)
flux_klein_outputs/      # FLUX.2-klein-4B generated images
moss_outputs/            # MOSS-Image-SFT generated images

zimage_benchmark_results.json          # Generation metrics (time, VRAM, errors)
flux_klein_benchmark_results.json
moss_benchmark_results.json

zimage_review_records.json             # Structured review records (human evaluation templates)
flux_klein_review_records.json
moss_review_records.json

zimage_contact_sheet.png               # Visual contact sheet
flux_klein_contact_sheet.png
moss_contact_sheet.png
```

Package them into a single ZIP for transfer:

```python
# Cell 6: Package outputs for download
import zipfile, os
with zipfile.ZipFile("/kaggle/working/benchmark_outputs.zip", "w") as z:
    for root, _, files in os.walk("/kaggle/working"):
        for f in files:
            fpath = os.path.join(root, f)
            if any(k in fpath for k in ["_outputs/", "_results.json", "_review_records.json", "_contact_sheet.png"]):
                z.write(fpath, os.path.relpath(fpath, "/kaggle/working"))
print("Created /kaggle/working/benchmark_outputs.zip")
```

## Step 6: Aggregate Results Locally

Download `benchmark_outputs.zip`, extract, and run:

```bash
# Place all JSON files in ./benchmark_results/
mkdir -p benchmark_results
# Copy extracted JSON files here

python benchmark_aggregator.py
```

This produces `benchmark_comparison_report.md` with:
- Per-model generation statistics (time, VRAM, success rate)
- Human evaluation summary (if review records are scored)
- Contact sheet references
- Hardware notes

## Expected Outputs Per Model

### Z-Image-Turbo
- 33 images at 1024×1024
- ~6-8s per image on T4
- VRAM peak: ~11-14GB (BF16) or ~13-16GB (FP16)
- Contact sheet: `zimage_contact_sheet.png`

### FLUX.2-klein-4B
- 33 images at 1024×1024
- ~3-4s per image on T4
- VRAM peak: ~9-11GB (BF16)
- Contact sheet: `flux_klein_contact_sheet.png`

### MOSS-Image-SFT
- 33 images at 1024×1024
- ~45-60s per image on T4 (50 steps)
- VRAM peak: ~10-13GB
- Contact sheet: `moss_contact_sheet.png`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Flux2KleinPipeline` not found | Run `!pip install git+https://github.com/huggingface/diffusers.git` first |
| CUDA OOM | The script uses `enable_model_cpu_offload()`. If still OOM, reduce resolution or close other notebooks. |
| Smoke test failed | Full benchmark is aborted. Check the error, fix dependencies, retry. |
| No GPU detected | Set Accelerator to GPU T4/P100 in notebook settings and restart. |
| BF16 not supported | Script auto-detects and falls back to FP16 on T4/P100. |

## Precaution

**Do not depend on Kaggle notebook URLs for production.** Kaggle notebooks are ephemeral testing environments. After benchmarking, deploy the selected model on persistent GPU hosting (see research report deployment section).
