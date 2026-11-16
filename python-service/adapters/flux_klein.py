"""FLUX.2-klein-4B adapter using Flux2KleinPipeline."""

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


class FluxKleinAdapter(BaseGenerationAdapter):
    """Adapter for black-forest-labs/FLUX.2-klein-4B (4B, 4 steps, guidance 1.0, up to 4MP)."""

    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-klein-4B", device: str = "cuda", dtype=None):
        super().__init__(model_id, device, dtype)
        self.dtype = dtype or get_torch_dtype(prefer_bf16=False)  # T4/Turing prefers FP16

    def load(self) -> None:
        if self.is_loaded():
            return
        # Flux2KleinPipeline requires diffusers from git main
        from diffusers import Flux2KleinPipeline

        t0 = time.time()
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.to(self.device)
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
            guidance_scale=input.guidance_scale if input.guidance_scale is not None else 1.0,
            num_inference_steps=input.num_inference_steps or 4,
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
                "steps": input.num_inference_steps or 4,
                "guidance_scale": input.guidance_scale if input.guidance_scale is not None else 1.0,
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
            display_name="FLUX.2-klein-4B",
            description="4B flux model, 4-step generation, up to 4MP, Apache-2.0",
            license="Apache-2.0",
            min_vram_gb=8.0,
            recommended_vram_gb=13.0,
            default_steps=4,
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
