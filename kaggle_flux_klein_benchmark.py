#!/usr/bin/env python3
"""
KAGGLE NOTEBOOK: FLUX.2 [klein] 4B Benchmark
============================================
Run on Kaggle with T4 (16GB VRAM), P100 (16GB VRAM), or T4x2 (16GB VRAM per GPU).

Steps to run:
1. Create new Kaggle Notebook
2. Set Accelerator: GPU T4 (1x) or P100
3. Enable Internet: On
4. Upload this script + benchmark_framework.py as notebook cells
5. Run all cells

Model: black-forest-labs/FLUX.2-klein-4B (4B, Apache-2.0, 4-step distilled)
Official baseline: num_inference_steps=4, guidance_scale=1.0, 1024x1024
Note: Flux2KleinPipeline requires diffusers from GitHub main:
      pip install git+github.com/huggingface/diffusers.git
"""

import os
import sys
import time
import json
import subprocess
import traceback
import torch
from datetime import datetime
from PIL import Image
from dataclasses import asdict

# Add benchmark framework to path (must be uploaded alongside this script)
sys.path.append("/kaggle/working")
from benchmark_framework import (
    BENCHMARK_PROMPTS,
    GenerationResult,
    ReviewRecord,
    save_results,
    save_review_records,
    make_contact_sheet,
    save_human_evaluation,
    assign_anonymous_labels,
)

OUTPUT_DIR = "/kaggle/working/flux_klein_outputs"
RESULTS_FILE = "/kaggle/working/flux_klein_benchmark_results.json"
REVIEW_FILE = "/kaggle/working/flux_klein_review_records.json"
CONTACT_SHEET = "/kaggle/working/flux_klein_contact_sheet.png"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# Official FLUX.2-klein-4B baseline per model card
NUM_STEPS = 4          # 4-step distilled
GUIDANCE = 1.0
RESOLUTION = (1024, 1024)
SMOKE_TEST_PROMPT = "A red apple on a wooden table, studio lighting, photorealistic"
SMOKE_TEST_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_gpu_architecture():
    """Detect GPU architecture and return precision recommendation."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}

    gpu_count = torch.cuda.device_count()
    gpus = []
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        cc = f"{props.major}.{props.minor}"
        supports_bf16 = props.major >= 8
        gpus.append({
            "index": i,
            "name": props.name,
            "vram_gb": round(vram_gb, 2),
            "compute_capability": cc,
            "supports_bf16": supports_bf16,
        })

    primary_dtype = torch.bfloat16 if any(g["supports_bf16"] for g in gpus) else torch.float16
    primary_dtype_name = "bfloat16" if primary_dtype == torch.bfloat16 else "float16"

    return {
        "available": True,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "primary_dtype": primary_dtype,
        "primary_dtype_name": primary_dtype_name,
    }


def check_gpu():
    """Check GPU availability and report per-GPU VRAM (no combined VRAM assumption)."""
    info = detect_gpu_architecture()
    if not info["available"]:
        print("WARNING: CUDA not available. This notebook requires GPU.")
        return False, None

    print(f"GPUs available: {info['gpu_count']}")
    for g in info["gpus"]:
        print(
            f"  GPU {g['index']}: {g['name']}, {g['vram_gb']:.1f} GB VRAM, "
            f"CC {g['compute_capability']}, BF16: {g['supports_bf16']}"
        )
    print(f"Primary precision: {info['primary_dtype_name']}")

    if info["gpu_count"] > 1:
        print(
            "NOTE: Multi-GPU does NOT pool VRAM. Each GPU has independent VRAM. "
            "Single-GPU placement is used unless explicit multi-GPU is configured."
        )

    return True, info


def get_vram_peak_gb(device_index=0):
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated(device_index) / (1024**3), 3)
    except Exception:
        return None


def reset_vram_peak(device_index=0):
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device_index)
        except Exception:
            pass


def get_system_ram_gb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024**3), 2)
        except Exception:
            return None


def load_model(gpu_info):
    """Load FLUX.2-klein-4B pipeline with hardware-appropriate precision."""
    from diffusers import Flux2KleinPipeline

    dtype = gpu_info["primary_dtype"]
    dtype_name = gpu_info["primary_dtype_name"]

    print(f"Loading {MODEL_ID} with dtype={dtype_name}...")
    t0 = time.time()

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=False,
    )

    pipe.enable_model_cpu_offload()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s (dtype={dtype_name})")

    return pipe, load_time


def smoke_test(pipe):
    """Generate one test image to verify the pipeline works before full benchmark."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: verifying pipeline generates an image...")
    print("=" * 60)
    try:
        generator = torch.Generator(device="cpu").manual_seed(SMOKE_TEST_SEED)
        image = pipe(
            prompt=SMOKE_TEST_PROMPT,
            height=RESOLUTION[1],
            width=RESOLUTION[0],
            guidance_scale=GUIDANCE,
            num_inference_steps=NUM_STEPS,
            generator=generator,
        ).images[0]
        smoke_path = os.path.join(OUTPUT_DIR, "smoke_test.png")
        image.save(smoke_path)
        print(f"SMOKE TEST PASSED: {smoke_path}")
        return True, smoke_path
    except Exception as e:
        print(f"SMOKE TEST FAILED: {e}")
        traceback.print_exc()
        return False, None


def generate_image(pipe, prompt, seed=42, width=1024, height=1024, num_steps=NUM_STEPS):
    """Generate a single image with FLUX.2-klein-4B, tracking VRAM/RAM peaks."""
    reset_vram_peak(0)
    ram_before = get_system_ram_gb()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.time()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=GUIDANCE,
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]
    gen_time = time.time() - t0

    vram_peak = get_vram_peak_gb(0)
    ram_after = get_system_ram_gb()

    return image, gen_time, vram_peak, ram_after


def run_benchmark(pipe, prompts, output_dir, results_file, review_file, contact_sheet):
    """Run full benchmark across all prompts with memory tracking."""
    results = []
    review_records = []

    for i, prompt_entry in enumerate(prompts):
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]

        print(f"\n[{i+1}/{len(prompts)}] {prompt_id}: {prompt_text[:80]}...")

        image_path = os.path.join(output_dir, f"{prompt_id}.png")

        try:
            image, gen_time, vram_peak, ram_peak = generate_image(
                pipe,
                prompt_text,
                seed=42,
                width=RESOLUTION[0],
                height=RESOLUTION[1],
                num_steps=NUM_STEPS,
            )
            image.save(image_path)

            result = GenerationResult(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                category=prompt_entry.get("category", ""),
                image_path=image_path,
                generation_time_s=gen_time,
                vram_peak_gb=vram_peak,
                ram_peak_gb=ram_peak,
                metadata={
                    "steps": NUM_STEPS,
                    "guidance": GUIDANCE,
                    "resolution": f"{RESOLUTION[0]}x{RESOLUTION[1]}",
                    "seed": 42,
                    "model": MODEL_ID,
                },
            )
            print(f"  Generated in {gen_time:.1f}s, VRAM peak: {vram_peak} GB -> {image_path}")

            review_records.append(ReviewRecord(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                prompt=prompt_text,
                image_path=image_path,
                expected_subjects=prompt_entry.get("expected_subjects", []),
                expected_objects_count=prompt_entry.get("expected_objects_count", {}),
                expected_colors=prompt_entry.get("expected_colors", []),
                camera_angle=prompt_entry.get("camera_angle", ""),
                lighting=prompt_entry.get("lighting", ""),
                environment=prompt_entry.get("environment", ""),
                generation_time_s=gen_time,
                vram_peak_gb=vram_peak,
            ))

        except Exception as e:
            print(f"  ERROR: {e}")
            result = GenerationResult(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                category=prompt_entry.get("category", ""),
                image_path="",
                generation_time_s=0.0,
                error=str(e),
                metadata={"steps": NUM_STEPS, "guidance": GUIDANCE, "resolution": f"{RESOLUTION[0]}x{RESOLUTION[1]}"},
            )
            review_records.append(ReviewRecord(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                prompt=prompt_text,
                image_path="",
                expected_subjects=prompt_entry.get("expected_subjects", []),
                expected_objects_count=prompt_entry.get("expected_objects_count", {}),
                expected_colors=prompt_entry.get("expected_colors", []),
                camera_angle=prompt_entry.get("camera_angle", ""),
                lighting=prompt_entry.get("lighting", ""),
                environment=prompt_entry.get("environment", ""),
                generation_time_s=0.0,
                vram_peak_gb=None,
                error=str(e),
            ))

        results.append(asdict(result))

    save_results(results, results_file)
    save_review_records(review_records, review_file)

    successful_images = [(r["image_path"], r["prompt_id"]) for r in results if r.get("image_path")]
    if successful_images:
        make_contact_sheet(successful_images, contact_sheet, cols=4)
        print(f"Contact sheet saved: {contact_sheet}")

    print(f"\nBenchmark complete. Results saved to {results_file}")
    return results, review_records


def main():
    print("=" * 60)
    print("FLUX.2 [klein] 4B Benchmark on Kaggle")
    print("=" * 60)

    gpu_ok, gpu_info = check_gpu()
    if not gpu_ok:
        print("GPU required. Exiting.")
        return

    pipe, load_time = load_model(gpu_info)

    smoke_passed, smoke_path = smoke_test(pipe)
    if not smoke_passed:
        print("Aborting benchmark due to smoke test failure.")
        return

    results, review_records = run_benchmark(
        pipe,
        BENCHMARK_PROMPTS,
        OUTPUT_DIR,
        RESULTS_FILE,
        REVIEW_FILE,
        CONTACT_SHEET,
    )

    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(successful)}/{len(results)} successful")
    if successful:
        avg_time = sum(r["generation_time_s"] for r in successful) / len(successful)
        vram_peaks = [r["vram_peak_gb"] for r in successful if r.get("vram_peak_gb")]
        print(f"Average generation time: {avg_time:.1f}s")
        if vram_peaks:
            print(f"Peak VRAM range: {min(vram_peaks):.2f} - {max(vram_peaks):.2f} GB")
    if failed:
        print(f"Failed: {[r['prompt_id'] for r in failed]}")
    print(f"{'=' * 60}")
    print(f"\nOutputs: {OUTPUT_DIR}/")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Review records: {REVIEW_FILE}")
    print(f"  Contact sheet: {CONTACT_SHEET}")


if __name__ == "__main__":
    main()
