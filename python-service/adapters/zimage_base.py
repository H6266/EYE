"""Z-Image base adapter using ZImagePipeline."""

from __future__ import annotations

import time
import torch
from PIL import Image

from adapters.base_adapter import (
    BaseGenerationAdapter,
    GenerationInput,
    GenerationResult,
    ModelInfo,
    get_torch_dtype,
    get_system_ram_gb,
    get_vram_peak_gb,
    reset_vram_peak,
    image_to_base64,
    image_to_bytes,
)


class ZImageBaseAdapter(BaseGenerationAdapter):
    """Adapter for Tongyi-MAI/Z-Image (base model, 50 steps, CFG, 1024x1024)."""

    def __init__(self, model_id: str = "Tongyi-MAI/Z-Image", device: str = "cuda", dtype=None):
        super().__init__(model_id, device, dtype)
        self.dtype = dtype or get_torch_dtype(prefer_bf16=True)

    def load(self) -> None:
        if self.is_loaded():
            return
        from diffusers import ZImagePipeline

        t0 = time.time()
        self.pipe = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.pipe.enable_model_cpu_offload()
        self.load_time_s = time.time() - t0
        self.loaded = True

    def unload(self) -> None:
        if self.pipe is not None:
            self.pipe = None
        self.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, input: GenerationInput) -> GenerationResult:
        if not self.is_loaded():
            self.load()

        reset_vram_peak(0)
        generator = torch.Generator(device="cpu").manual_seed(input.seed)

        t0 = time.time()
        image: Image.Image = self.pipe(
            prompt=input.prompt,
            negative_prompt=input.negative_prompt,
            height=input.height,
            width=input.width,
            guidance_scale=input.guidance_scale if input.guidance_scale is not None else 5.0,
            num_inference_steps=input.num_inference_steps or 50,
            cfg_normalization=False,
            generator=generator,
        ).images[0]
        inference_s = time.time() - t0

        vram_peak = get_vram_peak_gb(0)
        ram_peak = get_system_ram_gb()

        result = GenerationResult(
            model=self.model_id,
            prompt_used=input.prompt,
            seed=input.seed,
            inference_time_s=inference_s,
            vram_peak_gb=vram_peak,
            ram_peak_gb=ram_peak,
            metadata={
                "steps": input.num_inference_steps or 50,
                "guidance_scale": input.guidance_scale if input.guidance_scale is not None else 5.0,
                "cfg_normalization": False,
                "resolution": f"{input.width}x{input.height}",
                "model": self.model_id,
                "dtype": str(self.dtype),
            },
        )

        if input.return_base64:
            result.image_base64 = image_to_base64(image)
        else:
            result.image_bytes = image_to_bytes(image)

        return result

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            display_name="Z-Image (base)",
            description="Base Z-Image model, 50-step generation, CFG supported, 1024x1024, Apache-2.0",
            license="Apache-2.0",
            min_vram_gb=8.0,
            recommended_vram_gb=16.0,
            default_steps=50,
            default_resolution=(1024, 1024),
            supports_cfg=True,
            supports_negative_prompt=True,
            supports_img2img=False,
            supports_ip_adapter=False,
            supports_controlnet=False,
            cpu_fallback=False,
        )

    def health_check(self) -> bool:
        return self.is_loaded()
