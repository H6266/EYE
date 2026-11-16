#!/usr/bin/env python3
"""
KAGGLE NOTEBOOK: FLUX.2 [klein] 4B Benchmark
============================================
Run on Kaggle with T4x2 GPU (16GB VRAM per GPU) or P100 (16GB VRAM).

Model: black-forest-labs/FLUX.2-klein-4B (4B, Apache-2.0, 4 steps distilled)
"""

import os
import sys
import time
import json
import torch
from datetime import datetime
from PIL import Image
from dataclasses import asdict

sys.path.append("/kaggle/working")
from benchmark_framework import BENCHMARK_PROMPTS, GenerationResult, save_results

OUTPUT_DIR = "/kaggle/working/flux_klein_outputs"
RESULTS_FILE = "/kaggle/working/flux_klein_benchmark_results.json"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
        return False
    gpu_count = torch.cuda.device_count()
    print(f"GPUs available: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}, {vram_gb:.1f} GB VRAM")
    return True


def load_model():
    from diffusers import Flux2KleinPipeline
    
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    
    pipe.enable_model_cpu_offload()
    
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    
    return pipe


def generate_image(pipe, prompt, seed=42, width=1024, height=1024, num_steps=4):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    t0 = time.time()
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=1.0,
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]
    
    gen_time = time.time() - t0
    
    return image, gen_time


def run_benchmark(pipe, prompts, output_dir, results_file):
    results = []
    
    for i, prompt_entry in enumerate(prompts):
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]
        
        print(f"\n[{i+1}/{len(prompts)}] {prompt_id}: {prompt_text[:80]}...")
        
        image_path = os.path.join(output_dir, f"{prompt_id}.png")
        
        try:
            image, gen_time = generate_image(
                pipe, prompt_text, seed=42, width=1024, height=1024, num_steps=4
            )
            image.save(image_path)
            
            result = GenerationResult(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                image_path=image_path,
                generation_time_s=gen_time,
                metadata={"steps": 4, "guidance": 1.0, "resolution": "1024x1024"},
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
    print("FLUX.2 [klein] 4B Benchmark on Kaggle")
    print("=" * 60)
    
    if not check_gpu():
        return
    
    pipe = load_model()
    results = run_benchmark(pipe, BENCHMARK_PROMPTS, OUTPUT_DIR, RESULTS_FILE)
    
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