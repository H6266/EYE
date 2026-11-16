"""Pydantic request/response models for the Python inference service API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str = "ok"
    gpu_available: bool = False
    gpu_count: int = 0
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    loaded_models: list[str] = Field(default_factory=list)
    primary_dtype: str = "float32"


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_id: str
    display_name: str
    description: str
    license: str
    min_vram_gb: float
    recommended_vram_gb: float
    default_steps: int
    default_resolution: list[int]
    supports_cfg: bool
    supports_negative_prompt: bool
    supports_img2img: bool
    supports_ip_adapter: bool
    supports_controlnet: bool
    cpu_fallback: bool = False


class ModelsListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[ModelInfoResponse]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model ID (e.g. z-image-turbo, flux-klein-4b)")
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int = 42
    return_base64: bool = True


class Timings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_load_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_base64: str | None = None
    image_url: str | None = None
    seed: int
    model: str
    timings: Timings
    vram_peak_gb: float | None = None
    ram_peak_gb: float | None = None
    prompt_used: str
    optimized_prompt: str | None = None
    error: str | None = None


class ParsePromptRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(..., description="Natural language prompt")
    return_json: bool = True
    mode: str = Field("faithful", description="faithful or creative")
    target_model: str | None = None


class ParsePromptResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original_prompt: str
    optimized_prompt: str
    mode: str
    model_recommendation: str | None = None
    estimated_vram_gb: float | None = None
    estimated_steps: int | None = None
    scene_spec: dict[str, Any] | None = None
    error: str | None = None


class QualityVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_base64: str = Field(..., description="Base64 data URI of the generated image")
    prompt: str = Field(..., description="Original text prompt")
    scene_spec: dict[str, Any] | None = None


class QualityVerifyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    overall_pass: bool
    confidence: float
    criteria: list[dict[str, Any]]
    missing_subjects: list[str] = Field(default_factory=list)
    incorrect_attributes: list[str] = Field(default_factory=list)
    unrequested_additions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: str
    detail: str | None = None
