"""Structured scene specification schemas for prompt understanding.

Defines the JSON schema used by the prompt parser to represent a decomposed
image generation request.
"""

from __future__ import annotations

from typing import Any


SCENE_SPEC_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "StructuredSceneSpec",
    "type": "object",
    "required": ["original_prompt", "optimized_prompt", "mode"],
    "properties": {
        "original_prompt": {"type": "string", "description": "User prompt preserved verbatim"},
        "optimized_prompt": {"type": "string", "description": "Rewritten prompt preserving all elements"},
        "mode": {"type": "string", "enum": ["faithful", "creative"]},
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["person", "animal", "object"]},
                    "count": {"type": "integer", "minimum": 1},
                    "attributes": {"type": "object"},
                    "position": {"type": "string", "enum": ["foreground", "middle-ground", "background", "specific"]},
                    "spatial_relationship": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["type", "count"],
            },
        },
        "object_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        "environment": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "background_elements": {"type": "array", "items": {"type": "string"}},
                "weather": {"type": "string"},
                "time_of_day": {"type": "string"},
                "season": {"type": "string"},
            },
        },
        "lighting": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "color_temperature": {"type": "string"},
                "direction": {"type": "string"},
            },
        },
        "camera": {
            "type": "object",
            "properties": {
                "framing": {"type": "string"},
                "angle": {"type": "string"},
                "perspective": {"type": "string"},
                "lens": {"type": "string"},
                "depth_of_field": {"type": "string"},
            },
        },
        "color_palette": {"type": "array", "items": {"type": "string"}},
        "artistic_style": {"type": "string"},
        "atmospheric_conditions": {"type": "string"},
        "constraints": {
            "type": "object",
            "properties": {
                "must_include": {"type": "array", "items": {"type": "string"}},
                "must_not_include": {"type": "array", "items": {"type": "string"}},
                "colors_required": {"type": "array", "items": {"type": "string"}},
            },
        },
        "security": {
            "type": "object",
            "properties": {
                "no_pin_visible": {"type": "boolean"},
                "no_password_spoken": {"type": "boolean"},
                "user_retains_control": {"type": "boolean"},
            },
        },
        "model_recommendation": {"type": "string"},
        "estimated_vram_gb": {"type": "number"},
        "estimated_steps": {"type": "integer"},
    },
}


def empty_scene_spec(original_prompt: str, mode: str = "faithful") -> dict[str, Any]:
    """Return an empty scene spec with required fields filled."""
    return {
        "original_prompt": original_prompt,
        "optimized_prompt": original_prompt,
        "mode": mode,
        "subjects": [],
        "object_counts": {},
        "environment": {
            "location": "",
            "background_elements": [],
            "weather": "",
            "time_of_day": "",
            "season": "",
        },
        "lighting": {"type": "", "color_temperature": "", "direction": ""},
        "camera": {
            "framing": "",
            "angle": "",
            "perspective": "",
            "lens": "",
            "depth_of_field": "",
        },
        "color_palette": [],
        "artistic_style": "",
        "atmospheric_conditions": "",
        "constraints": {
            "must_include": [],
            "must_not_include": [],
            "colors_required": [],
        },
        "security": {
            "no_pin_visible": True,
            "no_password_spoken": True,
            "user_retains_control": True,
        },
        "model_recommendation": None,
        "estimated_vram_gb": None,
        "estimated_steps": None,
    }
