#!/usr/bin/env python3
"""
BENCHMARK RESULTS AGGREGATOR
============================
Combine results from all model benchmarks and generate comparison report.

Run locally after downloading results from Kaggle notebooks.

Reads:
- ./benchmark_results/<model>_benchmark_results.json  (generation metrics)
- ./benchmark_results/<model>_review_records.json     (human evaluation)
- ./benchmark_results/<model>_contact_sheet.png       (visual comparison)

Writes:
- ./benchmark_comparison_report.md
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

# Models in scope for this benchmark (Z-Image-Turbo, FLUX.2-klein-4B, MOSS-Image-SFT, SDXS-512 baseline)
MODEL_RESULT_FILES = {
    "Z-Image-Turbo": "zimage_benchmark_results.json",
    "FLUX.2-klein-4B": "flux_klein_benchmark_results.json",
    "MOSS-Image-SFT": "moss_benchmark_results.json",
    "SDXS-512 (baseline)": "sdxs_baseline_results.json",
}

MODEL_REVIEW_FILES = {
    "Z-Image-Turbo": "zimage_review_records.json",
    "FLUX.2-klein-4B": "flux_klein_review_records.json",
    "MOSS-Image-SFT": "moss_review_records.json",
}

MODEL_CONTACT_SHEETS = {
    "Z-Image-Turbo": "zimage_contact_sheet.png",
    "FLUX.2-klein-4B": "flux_klein_contact_sheet.png",
    "MOSS-Image-SFT": "moss_contact_sheet.png",
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


def load_all_reviews() -> Dict[str, List[dict]]:
    """Load all human review records."""
    all_reviews = {}
    for model_id, filename in MODEL_REVIEW_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            all_reviews[model_id] = data.get("records", data)
            print(f"Loaded {model_id} reviews: {len(all_reviews[model_id])} records")
        else:
            print(f"WARNING: {filepath} not found")
    return all_reviews


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
            "avg_vram_peak_gb": None,
            "max_vram_peak_gb": None,
            "avg_ram_peak_gb": None,
        }

    avg_time = sum(r["generation_time_s"] for r in successful) / len(successful)
    vram_peaks = [r["vram_peak_gb"] for r in successful if r.get("vram_peak_gb") is not None]
    ram_peaks = [r["ram_peak_gb"] for r in successful if r.get("ram_peak_gb") is not None]

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "avg_generation_time_s": round(avg_time, 2),
        "success_rate": round(len(successful) / len(results) * 100, 1),
        "avg_vram_peak_gb": round(sum(vram_peaks) / len(vram_peaks), 3) if vram_peaks else None,
        "max_vram_peak_gb": round(max(vram_peaks), 3) if vram_peaks else None,
        "avg_ram_peak_gb": round(sum(ram_peaks) / len(ram_peaks), 2) if ram_peaks else None,
    }


def compute_human_eval_stats(reviews: List[dict]) -> dict:
    """Compute average human evaluation scores from review records."""
    scored = [r for r in reviews if r.get("overall_accuracy") is not None]
    if not scored:
        return {
            "avg_overall": None,
            "avg_subject_presence": None,
            "avg_object_count": None,
            "avg_color_accuracy": None,
            "avg_composition": None,
            "avg_lighting": None,
            "avg_environment": None,
            "unwanted_elements_count": 0,
            "total_scored": 0,
        }
    return {
        "avg_overall": round(sum(r["overall_accuracy"] for r in scored) / len(scored), 3),
        "avg_subject_presence": round(sum(r["subject_presence_score"] for r in scored if r.get("subject_presence_score") is not None) / max(1, sum(1 for r in scored if r.get("subject_presence_score") is not None)), 3),
        "avg_object_count": round(sum(r["object_count_accuracy"] for r in scored if r.get("object_count_accuracy") is not None) / max(1, sum(1 for r in scored if r.get("object_count_accuracy") is not None)), 3),
        "avg_color_accuracy": round(sum(r["color_accuracy"] for r in scored if r.get("color_accuracy") is not None) / max(1, sum(1 for r in scored if r.get("color_accuracy") is not None)), 3),
        "avg_composition": round(sum(r["composition_score"] for r in scored if r.get("composition_score") is not None) / max(1, sum(1 for r in scored if r.get("composition_score") is not None)), 3),
        "avg_lighting": round(sum(r["lighting_accuracy"] for r in scored if r.get("lighting_accuracy") is not None) / max(1, sum(1 for r in scored if r.get("lighting_accuracy") is not None)), 3),
        "avg_environment": round(sum(r["environment_match"] for r in scored if r.get("environment_match") is not None) / max(1, sum(1 for r in scored if r.get("environment_match") is not None)), 3),
        "unwanted_elements_count": sum(1 for r in scored if r.get("unwanted_elements")),
        "total_scored": len(scored),
    }


def generate_report(all_results: Dict[str, List[dict]], all_reviews: Dict[str, List[dict]]) -> str:
    """Generate markdown comparison report."""
    lines = []

    lines.append("# Image Generation Engine Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTotal prompts tested: {len(BENCHMARK_PROMPTS)}")
    lines.append(f"\nPrompt categories: human_portrait, complex_clothing, animal_wildlife, architecture, landscape, multiple_people, specific_object_count, specific_object_position, complex_camera_angle, atmospheric_conditions, anime, cartoon_2d, 3d_animation, watercolor, fantasy, complex_multi_requirement")
    lines.append(f"\nModels in scope: Z-Image-Turbo, FLUX.2-klein-4B, MOSS-Image-SFT, SDXS-512 (baseline)")
    lines.append(f"\nNote: All scores below are from ACTUAL benchmark runs. No theoretical or fabricated numbers are included.")

    lines.append("\n## Model Summary Statistics")
    lines.append("\n| Model | Total | Successful | Failed | Avg Time (s) | Success Rate | Avg VRAM Peak (GB) | Max VRAM Peak (GB) | Avg RAM Peak (GB) |")
    lines.append("|-------|-------|------------|--------|--------------|--------------|--------------------|--------------------|-------------------|")

    for model_id, results in all_results.items():
        stats = compute_model_stats(results)
        lines.append(
            f"| {model_id} | {stats['total']} | {stats['successful']} | {stats['failed']} | "
            f"{stats['avg_generation_time_s']} | {stats['success_rate']}% | "
            f"{stats['avg_vram_peak_gb']} | {stats['max_vram_peak_gb']} | {stats['avg_ram_peak_gb']} |"
        )

    lines.append("\n## Human Prompt-Fidelity Evaluation")
    lines.append("\nScores are 0-1 from human evaluators. 'N/A' means not yet scored.")
    lines.append("\n| Model | Total Scored | Overall (0-1) | Subject Presence | Object Count | Color Accuracy | Composition | Lighting | Environment | Unwanted Elements |")
    lines.append("|-------|--------------|---------------|------------------|--------------|----------------|-------------|----------|-------------|-------------------|")

    for model_id, reviews in all_reviews.items():
        stats = compute_human_eval_stats(reviews)
        ue = stats["unwanted_elements_count"]
        lines.append(
            f"| {model_id} | {stats['total_scored']} | {stats['avg_overall']} | "
            f"{stats['avg_subject_presence']} | {stats['avg_object_count']} | "
            f"{stats['avg_color_accuracy']} | {stats['avg_composition']} | "
            f"{stats['avg_lighting']} | {stats['avg_environment']} | {ue} |"
        )

    lines.append("\n## Detailed Per-Model Results")

    for model_id, results in all_results.items():
        lines.append(f"\n### {model_id}")
        lines.append("\n| Prompt ID | Category | Time (s) | VRAM Peak (GB) | Status | Error |")
        lines.append("|-----------|----------|----------|----------------|--------|-------|")

        for r in results:
            status = "FAILED" if r.get("error") else "OK"
            error = r.get("error", "")[:60] if r.get("error") else ""
            lines.append(
                f"| {r['prompt_id']} | {r.get('category', 'N/A')} | "
                f"{r.get('generation_time_s', 0):.1f} | {r.get('vram_peak_gb', 'N/A')} | {status} | {error} |"
            )

    lines.append("\n## Visual Comparison (Contact Sheets)")
    lines.append("\nContact sheets show all generated images side by side for visual comparison:")
    for model_id, sheet_file in MODEL_CONTACT_SHEETS.items():
        sheet_path = os.path.join(RESULTS_DIR, sheet_file)
        if os.path.exists(sheet_path):
            lines.append(f"- **{model_id}:** `{sheet_path}`")
        else:
            lines.append(f"- **{model_id}:** not available")

    lines.append("\n## Hardware Notes")
    lines.append("- Benchmarks run on Kaggle free tier: T4 (16GB VRAM), P100 (16GB VRAM), or T4x2 (16GB VRAM per GPU)")
    lines.append("- VRAM is reported PER GPU. Multi-GPU does NOT pool VRAM (each GPU has independent 16GB).")
    lines.append("- Precision: BF16 used on Ampere+ GPUs, FP16 on T4 (Turing) and P100 (Pascal).")
    lines.append("- Z-Image-Turbo: official 8-step baseline, guidance=0.0")
    lines.append("- FLUX.2-klein-4B: official 4-step baseline, guidance=1.0")
    lines.append("- MOSS-Image-SFT: official 50-step baseline, CFG=3, uncond_skip_layers=[22]")

    lines.append("\n## Next Steps")
    lines.append("1. Human evaluators complete scoring in the HTML review page (benchmark_comparison_report references CSVs)")
    lines.append("2. Fill in human_notes, missing_details, incorrect_details, unwanted_additions for each record")
    lines.append("3. Run VLM-based automated evaluation as a cross-check (NOT ground truth)")
    lines.append("4. Compare results against SDXS-512 baseline")
    lines.append("5. Select winning model based on measured evidence — not theoretical estimates")

    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = load_all_results()
    all_reviews = load_all_reviews()

    if not all_results:
        print("No result files found. Place benchmark JSON files in ./benchmark_results/")
        print("Expected files: zimage_benchmark_results.json, flux_klein_benchmark_results.json, moss_benchmark_results.json, sdxs_baseline_results.json")
        return

    report = generate_report(all_results, all_reviews)

    with open(REPORT_FILE, "w") as f:
        f.write(report)

    print(f"\nReport saved to {REPORT_FILE}")
    print("\n--- Report Preview ---")
    print(report[:2500])


if __name__ == "__main__":
    main()
