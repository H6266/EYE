"""FastAPI application for the Python inference service."""

from __future__ import annotations

import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engine_loader import EngineLoader, MODEL_REGISTRY
from models import (
    HealthResponse,
    ModelsListResponse,
    GenerateRequest,
    GenerateResponse,
    Timings,
    ParsePromptRequest,
    ParsePromptResponse,
    QualityVerifyRequest,
    QualityVerifyResponse,
    ErrorResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

loader: EngineLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader
    loader = EngineLoader()
    logger.info(f"Python inference service starting on device={loader.device}")
    yield
    if loader:
        loader.unload_all()
    logger.info("Python inference service shutting down")


app = FastAPI(
    title="Auteur Studio Python Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_loader() -> EngineLoader:
    if loader is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return loader


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    ldr = _get_loader()
    health_info = ldr.health_check()
    return HealthResponse(
        status="ok",
        gpu_available=health_info["gpu_available"],
        gpu_count=health_info["gpu_count"],
        gpus=health_info["gpus"],
        loaded_models=health_info["loaded_models"],
        primary_dtype="bfloat16" if health_info["gpu_available"] else "float32",
    )


@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    ldr = _get_loader()
    models = ldr.list_models()
    return ModelsListResponse(models=[ModelsListResponse.model_construct(**{k: v for k, v in m.items() if k != "engine_id"}) for m in models])


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    ldr = _get_loader()
    adapter = ldr.get_adapter(req.model)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' not available or not enabled. "
                   f"Enabled models: {[m for m, e in MODEL_REGISTRY.items() if e['enabled']]}",
        )

    from adapters.base_adapter import GenerationInput

    t_total = time.time()
    try:
        gen_input = GenerationInput(
            model=req.model,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            return_base64=req.return_base64,
        )
        result = adapter.generate(gen_input)
        total_ms = (time.time() - t_total) * 1000

        if result.error:
            return GenerateResponse(
                image_base64=None,
                image_url=None,
                seed=result.seed,
                model=result.model,
                timings=Timings(inference_ms=result.inference_time_s * 1000, total_ms=total_ms),
                vram_peak_gb=result.vram_peak_gb,
                ram_peak_gb=result.ram_peak_gb,
                prompt_used=result.prompt_used,
                optimized_prompt=None,
                error=result.error,
            )

        return GenerateResponse(
            image_base64=result.image_base64,
            image_url=None,
            seed=result.seed,
            model=result.model,
            timings=Timings(inference_ms=result.inference_time_s * 1000, total_ms=total_ms),
            vram_peak_gb=result.vram_peak_gb,
            ram_peak_gb=result.ram_peak_gb,
            prompt_used=result.prompt_used,
            optimized_prompt=result.metadata.get("optimized_prompt"),
        )
    except Exception as exc:
        logger.exception(f"Generation failed for model {req.model}")
        total_ms = (time.time() - t_total) * 1000
        return GenerateResponse(
            image_base64=None,
            image_url=None,
            seed=req.seed,
            model=req.model,
            timings=Timings(inference_ms=0.0, total_ms=total_ms),
            prompt_used=req.prompt,
            error=str(exc),
        )


@app.post("/v1/prompt/parse", response_model=ParsePromptResponse)
async def parse_prompt(req: ParsePromptRequest):
    """Parse a natural language prompt into a structured scene spec."""
    from prompt.parser import PromptParser

    parser = PromptParser()
    try:
        result = parser.parse(req.prompt, mode=req.mode, target_model=req.target_model)
        return ParsePromptResponse(**result)
    except Exception as exc:
        logger.exception("Prompt parsing failed")
        return ParsePromptResponse(
            original_prompt=req.prompt,
            optimized_prompt=req.prompt,
            mode=req.mode,
            error=str(exc),
        )


@app.post("/v1/quality/verify", response_model=QualityVerifyResponse)
async def verify_quality(req: QualityVerifyRequest):
    """Verify a generated image against its prompt using VLM evaluation."""
    from quality.vlm_evaluator import VLMQualityEvaluator

    evaluator = VLMQualityEvaluator()
    try:
        result = evaluator.verify(req.image_base64, req.prompt, req.scene_spec)
        return QualityVerifyResponse(**result)
    except Exception as exc:
        logger.exception("Quality verification failed")
        return QualityVerifyResponse(
            overall_pass=False,
            confidence=0.0,
            criteria=[],
            error=str(exc),
        )


@app.post("/v1/upscale")
async def upscale():
    """Real-ESRGAN upscaling (placeholder)."""
    raise HTTPException(status_code=501, detail="Upscaling not yet implemented")


@app.delete("/v1/models/{model_id}")
async def unload_model(model_id: str):
    ldr = _get_loader()
    ok = ldr.unload_model(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    return {"status": "unloaded", "model_id": model_id}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
