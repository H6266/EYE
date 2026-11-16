"""SDXS-512 ONNX fallback adapter (stub — SDXS-512 runs in the Node.js process).

This adapter exists for interface completeness. The actual SDXS-512 inference
happens in-process in server.ts via ONNX Runtime. This Python adapter raises
NotImplementedError on generate() to signal that the Node.js local engine
should handle SDXS-512 requests directly.
"""

from __future__ import annotations

from adapters.base_adapter import (
    BaseGenerationAdapter,
    GenerationInput,
    GenerationResult,
    ModelInfo,
)


class SdxsOnnxAdapter(BaseGenerationAdapter):
    """Stub adapter for SDXS-512 ONNX CPU (runs in Node.js, not Python)."""

    def __init__(self, model_id: str = "skillsafeai/sdxs-512-dreamshaper-onnx-webgpu", device: str = "cpu", dtype=None):
        super().__init__(model_id, device, dtype)
        self.loaded = True  # Always "loaded" — handled by Node.js

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = True

    def generate(self, input: GenerationInput) -> GenerationResult:
        return GenerationResult(
            model=self.model_id,
            prompt_used=input.prompt,
            seed=input.seed,
            inference_time_s=0.0,
            error=(
                "SDXS-512 ONNX runs in the Node.js process, not the Python service. "
                "Route to local-sdxs engine instead."
            ),
            metadata={"model": self.model_id, "cpu_fallback": True},
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            display_name="SDXS-512 (ONNX CPU)",
            description="Existing local ONNX CPU fallback, 512x512, Apache-2.0",
            license="Apache-2.0",
            min_vram_gb=0.0,
            recommended_vram_gb=0.0,
            default_steps=4,
            default_resolution=(512, 512),
            supports_cfg=True,
            supports_negative_prompt=True,
            supports_img2img=True,
            supports_ip_adapter=False,
            supports_controlnet=False,
            cpu_fallback=True,
        )

    def health_check(self) -> bool:
        return True
