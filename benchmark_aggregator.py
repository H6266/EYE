#!/usr/bin/env python3
"""
BENCHMARK RESULTS AGGREGATOR
============================
Combine results from all model benchmarks and generate comparison report.

Run locally after downloading results from Kaggle notebooks.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_framework import BENCHMARK_PROMPTS, generate_comparison_table

RESULTS_DIR = "./benchmark_results"
REPORT_FILE = "./benchmark_comparison_report.md"

MODEL_RESULT_FILES = {
    "Z-Image-Turbo": "zimage_benchmark_results.json",
    "FLUX.2-klein-4B": "flux_klein_benchmark_results.json",
    "MOSS-Image-SFT": "moss_benchmark_results.json",
    "SDXS-512 (baseline)": "sdxs_baseline_results.json",  # To be filled from local runs
}


def load_all_results() -> Dict[str, List[dict]]:
    """Load all benchmark result files."""
    all_results = {}
    for model_id, filename in MODEL_RESULT_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            all_results[model_id] = data.get("results", data)
            print(f"Loaded {model_id}: {len(all_results[model_id])} results")
        else:
            print(f"WARNING: {filepath} not found")
    return all_results


def compute_model_stats(results: List[dict]) -> dict:
    """Compute summary statistics for a model's results."""
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    
    if not successful:
        return {
            "total": len(results),
            "successful": 0,
            "failed": len(failed),
            "avg_generation_time_s": 0,
            "success_rate": 0,
        }
    
    avg_time = sum(r["generation_time_s"] for r in successful) / len(successful)
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "avg_generation_time_s": round(avg_time, 2),
        "success_rate": round(len(successful) / len(results) * 100, 1),
    }


def generate_report(all_results: Dict[str, List[dict]]) -> str:
    """Generate markdown comparison report."""
    lines = []
    
    lines.append("# Image Generation Engine Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTotal prompts tested: {len(BENCHMARK_PROMPTS)}")
    lines.append(f"\nPrompt categories: human_portrait, complex_clothing, animal_wildlife, architecture, landscape, multiple_people, specific_object_count, specific_object_position, complex_camera_angle, atmospheric_conditions, anime, cartoon_2d, 3d_animation, watercolor, fantasy, complex_multi_requirement")
    
    lines.append("\n## Model Summary Statistics")
    lines.append("\n| Model | Total | Successful | Failed | Avg Time (s) | Success Rate |")
    lines.append("|-------|-------|------------|--------|--------------|--------------|")
    
    for model_id, results in all_results.items():
        stats = compute_model_stats(results)
        lines.append(
            f"| {model_id} | {stats['total']} | {stats['successful']} | {stats['failed']} | "
            f"{stats['avg_generation_time_s']} | {stats['success_rate']}% |"
        )
    
    lines.append("\n## Detailed Per-Model Results")
    
    for model_id, results in all_results.items():
        lines.append(f"\n### {model_id}")
        lines.append("\n| Prompt ID | Category | Time (s) | Status | Error |")
        lines.append("|-----------|----------|----------|--------|-------|")
        
        for r in results:
            status = "FAILED" if r.get("error") else "OK"
            error = r.get("error", "")[:60] if r.get("error") else ""
            lines.append(
                f"| {r['prompt_id']} | {r.get('category', 'N/A')} | "
                f"{r.get('generation_time_s', 0):.1f} | {status} | {error} |"
            )
    
    lines.append("\n## Hardware Notes")
    lines.append("- Benchmarks run on Kaggle free tier: T4x2 (16GB VRAM per GPU) or P100 (16GB VRAM)")
    lines.append("- Z-Image-Turbo and FLUX.2 klein 4B fit in 16GB VRAM")
    lines.append("- Qwen-Image-2512 does NOT fit in 16GB VRAM even quantized (requires 24-48GB)")
    lines.append("- MOSS-Image fits in 16GB VRAM with CPU offload")
    
    lines.append("\n## Next Steps")
    lines.append("1. Human evaluators score each generated image on prompt accuracy (0-1 scale)")
    lines.append("2. Fill in human_notes for each result")
    lines.append("3. Run VLM-based automated evaluation for subject presence, object count, color accuracy")
    lines.append("4. Compare results against SDXS-512 baseline")
    lines.append("5. Select winning model for Auteur Studio integration")
    
    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = load_all_results()
    
    if not all_results:
        print("No result files found. Place benchmark JSON files in ./benchmark_results/")
        return
    
    report = generate_report(all_results)
    
    os.makedirs(os.path.dirname(REPORT_FILE) or ".", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to {REPORT_FILE}")
    print("\n--- Report Preview ---")
    print(report[:2000])


if __name__ == "__main__":
    main()