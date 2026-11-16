#!/usr/bin/env python3
"""
KAGGLE NOTEBOOK: Z-Image-Turbo Benchmark
========================================
Run on Kaggle with T4x2 GPU (16GB VRAM per GPU) or P100 (16GB VRAM).

Steps to run:
1. Create new Kaggle Notebook
2. Set Accelerator: GPU T4 (1x or 2x) or P100
3. Enable Internet: On
4. Upload this script as a cell or create notebook from it
5. Run all cells

Model: Tongyi-MAI/Z-Image-Turbo (6B, Apache-2.0, 8 steps distilled)
"""

import os
import sys
import time
import json
import torch
from datetime import datetime
from PIL import Image

# Add benchmark framework to path
sys.path.append("/kaggle/working")
from benchmark_framework import BENCHMARK_PROMPTS, GenerationResult, evaluate_generation, save_results

OUTPUT_DIR = "/kaggle/working/zimage_outputs"
RESULTS_FILE = "/kaggle/working/zimage_benchmark_results.json"
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_gpu():
    """Check GPU availability and VRAM."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. This notebook requires GPU.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"GPUs available: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}, {vram_gb:.1f} GB VRAM, CC {props.major}.{props.minor}")
    
    return True


def load_model():
    """Load Z-Image-Turbo pipeline."""
    from diffusers import DiffusionPipeline
    
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    
    # Move to GPU with CPU offload for VRAM safety on 16GB cards
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_slicing()
    
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    
    return pipe


def generate_image(pipe, prompt, seed=42, width=1024, height=1024, num_steps=9):
    """Generate a single image with Z-Image-Turbo."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    t0 = time.time()
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=0.0,  # Turbo is distilled without CFG
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]
    
    gen_time = time.time() - t0
    
    return image, gen_time


def run_benchmark(pipe, prompts, output_dir, results_file):
    """Run full benchmark across all prompts."""
    results = []
    
    for i, prompt_entry in enumerate(prompts):
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]
        
        print(f"\n[{i+1}/{len(prompts)}] {prompt_id}: {prompt_text[:80]}...")
        
        image_path = os.path.join(output_dir, f"{prompt_id}.png")
        
        try:
            image, gen_time = generate_image(
                pipe, prompt_text, seed=42, width=1024, height=1024, num_steps=9
            )
            image.save(image_path)
            
            result = GenerationResult(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                image_path=image_path,
                generation_time_s=gen_time,
                vram_peak_gb=None,  # Will be filled by nvidia-smi parsing
                metadata={"steps": 9, "guidance": 0.0, "resolution": "1024x1024"},
            )
            print(f"  Generated in {gen_time:.1f}s -> {image_path}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            result = GenerationResult(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                image_path="",
                generation_time_s=0.0,
                error=str(e),
            )
        
        results.append(asdict(result))
    
    save_results(results, results_file)
    print(f"\nBenchmark complete. Results saved to {results_file}")
    return results


def main():
    print("=" * 60)
    print("Z-Image-Turbo Benchmark on Kaggle")
    print("=" * 60)
    
    if not check_gpu():
        print("GPU required. Exiting.")
        return
    
    pipe = load_model()
    
    results = run_benchmark(pipe, BENCHMARK_PROMPTS, OUTPUT_DIR, RESULTS_FILE)
    
    # Summary
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(successful)}/{len(results)} successful")
    if successful:
        avg_time = sum(r["generation_time_s"] for r in successful) / len(successful)
        print(f"Average generation time: {avg_time:.1f}s")
    if failed:
        print(f"Failed: {[r['prompt_id'] for r in failed]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()