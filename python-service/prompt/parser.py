"""Prompt parser: converts natural language prompts into structured scene specs.

Primary model: Qwen3-VL-4B-Instruct (Apache-2.0, 4B, vision + instruction following).
Fallback: Qwen2.5-0.5B-Instruct (Apache-2.0, 0.5B, text-only).
If no model is available, uses a lightweight rule-based fallback.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt.schemas import SCENE_SPEC_SCHEMA, empty_scene_spec

logger = logging.getLogger("prompt_parser")

# Model candidates in order of preference
PRIMARY_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_RECOMMENDATION_RULES = {
    "z-image-turbo": {"min_steps": 1, "max_steps": 10, "vram_gb": 16},
    "flux-klein-4b": {"min_steps": 1, "max_steps": 10, "vram_gb": 13},
    "z-image-base": {"min_steps": 11, "max_steps": 60, "vram_gb": 16},
    "moss-sft": {"min_steps": 11, "max_steps": 60, "vram_gb": 13},
    "qwen-2512": {"min_steps": 1, "max_steps": 5, "vram_gb": 48},
}


class PromptParser:
    """Parses prompts into structured scene specifications."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or PRIMARY_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_attempted = False

    def _try_load(self) -> bool:
        """Attempt to load the primary model, falling back to Qwen2.5-0.5B."""
        if self._load_attempted:
            return self._loaded
        self._load_attempted = True

        for candidate in [self.model_id, FALLBACK_MODEL]:
            try:
                logger.info(f"Loading prompt parser model: {candidate}")
                self.tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    candidate,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device == "cuda" and not hasattr(self.model, "hf_device_map"):
                    self.model.to(self.device)
                self.model_id = candidate
                self._loaded = True
                logger.info(f"Prompt parser loaded: {candidate} on {self.device}")
                return True
            except Exception as exc:
                logger.warning(f"Failed to load {candidate}: {exc}")
                continue

        logger.warning("No prompt parser model available; using rule-based fallback")
        self._loaded = False
        return False

    def parse(self, prompt: str, mode: str = "faithful", target_model: str | None = None) -> dict[str, Any]:
        """Parse a prompt into a structured scene spec.

        Returns a dict matching the StructuredSceneSpec schema.
        Never raises — falls back to rule-based parsing on any error.
        """
        if not prompt or not prompt.strip():
            return empty_scene_spec("", mode)

        # Try VLM/LLM-based parsing
        if not self._load_attempted:
            self._try_load()

        if self._loaded and self.model is not None and self.tokenizer is not None:
            try:
                return self._llm_parse(prompt, mode, target_model)
            except Exception as exc:
                logger.warning(f"LLM parse failed, falling back to rules: {exc}")

        # Rule-based fallback
        return self._rule_based_parse(prompt, mode, target_model)

    def _llm_parse(self, prompt: str, mode: str, target_model: str | None) -> dict[str, Any]:
        """Use the loaded LLM to parse the prompt into structured JSON."""
        system_prompt = (
            "You are a prompt understanding assistant for image generation. "
            "Parse the user's image description into a structured JSON scene spec. "
            "Preserve ALL subjects, objects, colors, positions, camera angles, "
            "lighting, and environment details. Never invent elements not requested. "
            "Output valid JSON only."
        )

        user_prompt = (
            f"Mode: {mode}\n"
            f"Target model: {target_model or 'auto'}\n"
            f"User prompt: {prompt}\n\n"
            f"Output JSON with these fields: original_prompt, optimized_prompt, "
            f"mode, subjects (type, count, attributes, position, spatial_relationship, action), "
            f"environment, lighting, camera, color_palette, artistic_style, "
            f"constraints (must_include, must_not_include, colors_required), "
            f"model_recommendation, estimated_vram_gb, estimated_steps."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        generated = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract JSON from the response
        json_str = self._extract_json(generated)
        if json_str:
            import json as _json
            parsed = _json.loads(json_str)
            parsed["original_prompt"] = prompt  # always preserve verbatim
            parsed["mode"] = mode
            # Fill defaults
            base = empty_scene_spec(prompt, mode)
            base.update(parsed)
            return base

        return self._rule_based_parse(prompt, mode, target_model)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract a JSON object from LLM output text."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _rule_based_parse(prompt: str, mode: str, target_model: str | None) -> dict[str, Any]:
        """Lightweight rule-based prompt parsing (no model required)."""
        spec = empty_scene_spec(prompt, mode)

        # Simple keyword extraction for common elements
        lower = prompt.lower()

        # Detect subjects (very rough)
        subject_keywords = ["person", "man", "woman", "boy", "girl", "child", "cat", "dog",
                            "bird", "horse", "car", "house", "tree", "flower", "airship",
                            "lantern", "rocket", "robot", "castle", "mountain", "ocean"]
        found_subjects = [kw for kw in subject_keywords if kw in lower]
        for subj in found_subjects[:3]:
            spec["subjects"].append({
                "type": "object",
                "count": 1,
                "attributes": {},
                "position": "foreground",
                "action": "",
            })

        # Detect colors
        color_keywords = ["red", "blue", "green", "yellow", "purple", "orange", "pink",
                          "black", "white", "brown", "gray", "grey", "gold", "silver"]
        found_colors = [c for c in color_keywords if c in lower]
        spec["color_palette"] = found_colors
        spec["constraints"]["colors_required"] = found_colors

        # Detect lighting
        lighting_keywords = ["dusk", "dawn", "night", "daylight", "sunset", "sunrise",
                             "studio lighting", "candlelight", "moonlight", "neon"]
        for lk in lighting_keywords:
            if lk in lower:
                spec["lighting"]["type"] = lk
                break

        # Detect time of day
        for tod in ["dusk", "dawn", "night", "day", "sunset", "sunrise"]:
            if tod in lower:
                spec["environment"]["time_of_day"] = tod
                break

        # Detect camera framing
        framing_keywords = ["close-up", "close up", "wide shot", "wide-shot", "medium shot",
                            "aerial", "bird's eye", "low angle", "high angle", "portrait", "landscape"]
        for fk in framing_keywords:
            if fk in lower:
                spec["camera"]["framing"] = fk
                break

        # Model recommendation based on prompt complexity
        if target_model:
            spec["model_recommendation"] = target_model
        elif len(prompt) > 100 or len(found_subjects) > 2:
            spec["model_recommendation"] = "z-image-base"
            spec["estimated_vram_gb"] = 16.0
            spec["estimated_steps"] = 50
        else:
            spec["model_recommendation"] = "z-image-turbo"
            spec["estimated_vram_gb"] = 16.0
            spec["estimated_steps"] = 8

        spec["optimized_prompt"] = prompt  # rule-based: keep unchanged
        return spec
