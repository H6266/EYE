#!/usr/bin/env python3
"""
KAGGLE NOTEBOOK: MOSS-Image Benchmark
=====================================
Run on Kaggle with T4x2 GPU (16GB VRAM per GPU) or P100 (16GB VRAM).

Model: OpenMOSS-Team/MOSS-Image-SFT (5B, Apache-2.0, 50 steps)
Also test: OpenMOSS-Team/MOSS-Image-Base
"""

import os
import sys
import time
import json
import torch
from PIL import Image
from dataclasses import asdict

sys.path.append("/kaggle/working")
from benchmark_framework import BENCHMARK_PROMPTS, GenerationResult, save_results

OUTPUT_DIR = "/kaggle/working/moss_outputs"
RESULTS_FILE = "/kaggle/working/moss_benchmark_results.json"
MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"

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
    from diffusers import DiffusionPipeline
    
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_slicing()
    
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    
    return pipe


def generate_image(pipe, prompt, seed=42, width=1024, height=1024, num_steps=50):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    t0 = time.time()
    
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=3.0,
        uncond_skip_layers=[22],
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
                pipe, prompt_text, seed=42, width=1024, height=1024, num_steps=50
            )
            image.save(image_path)
            
            result = GenerationResult(
                model_id=MODEL_ID,
                prompt_id=prompt_id,
                image_path=image_path,
                generation_time_s=gen_time,
                metadata={"steps": 50, "guidance": 3.0, "resolution": "1024x1024"},
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
    print("MOSS-Image Benchmark on Kaggle")
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