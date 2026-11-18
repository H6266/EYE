"""Model registry and lazy loader for the Python inference service.

Manages model lifecycle: lazy loading, unloading, VRAM tracking, health checks.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import torch

from adapters.base_adapter import BaseGenerationAdapter, ModelInfo
from adapters.flux_klein import FluxKleinAdapter
from adapters.moss_sft import MossSftAdapter
from adapters.qwen_2512 import Qwen2512Adapter
from adapters.sdxs_onnx import SdxsOnnxAdapter
from adapters.zimage_base import ZImageBaseAdapter
from adapters.zimage_turbo import ZImageTurboAdapter

logger = logging.getLogger("engine_loader")

# Model registry: model_id -> adapter class + enabled flag
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "z-image-turbo": {
        "cls": ZImageTurboAdapter,
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "enabled": os.getenv("ENABLE_ZIMAGE_TURBO", "true").lower() == "true",
    },
    "flux-klein-4b": {
        "cls": FluxKleinAdapter,
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "enabled": os.getenv("ENABLE_FLUX_KLEIN", "true").lower() == "true",
    },
    "z-image-base": {
        "cls": ZImageBaseAdapter,
        "model_id": "Tongyi-MAI/Z-Image",
        "enabled": os.getenv("ENABLE_ZIMAGE_BASE", "true").lower() == "true",
    },
    "qwen-2512": {
        "cls": Qwen2512Adapter,
        "model_id": "Qwen/Qwen-Image-2512",
        "enabled": os.getenv("ENABLE_QWEN_2512", "false").lower() == "true",
    },
    "moss-sft": {
        "cls": MossSftAdapter,
        "model_id": "OpenMOSS-Team/MOSS-Image-SFT",
        "enabled": os.getenv("ENABLE_MOSS_SFT", "true").lower() == "true",
    },
    "sdxs-onnx": {
        "cls": SdxsOnnxAdapter,
        "model_id": "skillsafeai/sdxs-512-dreamshaper-onnx-webgpu",
        "enabled": True,
        "cpu_fallback": True,
    },
}

MODEL_ALIASES = {
    "python-zimage-turbo": "z-image-turbo",
    "python-flux-klein": "flux-klein-4b",
    "python-zimage-base": "z-image-base",
    "python-qwen-2512": "qwen-2512",
    "python-moss-sft": "moss-sft",
}

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))


class EngineLoader:
    """Lazy model loader with thread-safe access and VRAM management."""

    def __init__(self, device: str = DEVICE, cache_dir: str = MODEL_CACHE_DIR):
        self.device = device
        self.cache_dir = cache_dir
        self._adapters: dict[str, BaseGenerationAdapter] = {}
        self._lock = threading.Lock()
        self._generation_lock = threading.RLock()
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    def list_models(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        """Return model metadata for all registered models."""
        result = []
        for model_id, entry in MODEL_REGISTRY.items():
            if not entry["enabled"] and not include_disabled:
                continue
            # Instantiate temporarily just to get info (no model load)
            adapter = entry["cls"](model_id=entry["model_id"], device=self.device)
            info = adapter.get_info()
            result.append({
                "model_id": model_id,
                "engine_id": model_id,
                "display_name": info.display_name,
                "description": info.description,
                "license": info.license,
                "min_vram_gb": info.min_vram_gb,
                "recommended_vram_gb": info.recommended_vram_gb,
                "default_steps": info.default_steps,
                "default_resolution": list(info.default_resolution),
                "supports_cfg": info.supports_cfg,
                "supports_negative_prompt": info.supports_negative_prompt,
                "supports_img2img": info.supports_img2img,
                "supports_ip_adapter": info.supports_ip_adapter,
                "supports_controlnet": info.supports_controlnet,
                "cpu_fallback": info.cpu_fallback,
                "enabled": entry["enabled"],
                "loaded": model_id in self._adapters and self._adapters[model_id].is_loaded(),
            })
        return result

    def is_model_available(self, model_id: str) -> bool:
        model_id = MODEL_ALIASES.get(model_id, model_id)
        entry = MODEL_REGISTRY.get(model_id)
        return bool(entry and entry["enabled"])

    def get_adapter(self, model_id: str) -> BaseGenerationAdapter | None:
        """Get or lazily load an adapter for the given model_id."""
        model_id = MODEL_ALIASES.get(model_id, model_id)
        if model_id not in MODEL_REGISTRY:
            return None
        entry = MODEL_REGISTRY[model_id]
        if not entry["enabled"]:
            return None
        if not entry.get("cpu_fallback", False) and (
            self.device == "cpu" or not torch.cuda.is_available()
        ):
            raise RuntimeError(
                f"Model '{model_id}' requires a CUDA GPU. "
                "Choose the local-sdxs engine or connect a GPU inference service."
            )

        with self._lock:
            if model_id not in self._adapters:
                adapter = entry["cls"](model_id=entry["model_id"], device=self.device)
                try:
                    adapter.load()
                except Exception as exc:
                    logger.exception("Failed to load model %s", model_id)
                    raise RuntimeError(f"Failed to load model '{model_id}': {exc}") from exc
                self._adapters[model_id] = adapter
            return self._adapters[model_id]

    def generate(self, model_id: str, generation_input) -> Any:
        """Serialize generations and keep at most one heavyweight adapter loaded."""
        model_id = MODEL_ALIASES.get(model_id, model_id)
        if model_id not in MODEL_REGISTRY:
            return None

        with self._generation_lock:
            for loaded_id in list(self._adapters):
                if loaded_id != model_id:
                    self.unload_model(loaded_id)

            was_loaded = model_id in self._adapters and self._adapters[model_id].is_loaded()
            adapter = self.get_adapter(model_id)
            if adapter is None:
                return None
            result = adapter.generate(generation_input)
            result.metadata["model_load_time_s"] = 0.0 if was_loaded else adapter.load_time_s
            return result

    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model to free VRAM."""
        with self._generation_lock:
            with self._lock:
                adapter = self._adapters.get(model_id)
                if adapter is not None:
                    try:
                        adapter.unload()
                    except Exception as exc:
                        logger.exception("Error unloading %s", model_id)
                        raise RuntimeError(
                            f"Failed to unload model '{model_id}': {exc}"
                        ) from exc
                    del self._adapters[model_id]
                    return True
                return False

    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            for model_id, adapter in list(self._adapters.items()):
                try:
                    adapter.unload()
                except Exception as exc:
                    logger.error(f"Error unloading {model_id}: {exc}")
            self._adapters.clear()

    def health_check(self) -> dict[str, Any]:
        """Return health status for all models."""
        import torch as _torch

        gpu_available = _torch.cuda.is_available()
        gpu_count = _torch.cuda.device_count() if gpu_available else 0
        gpus = []
        if gpu_available:
            for i in range(gpu_count):
                props = _torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        models_status = {}
        for model_id, entry in MODEL_REGISTRY.items():
            adapter = self._adapters.get(model_id)
            models_status[model_id] = {
                "enabled": entry["enabled"],
                "loaded": adapter.is_loaded() if adapter else False,
                "healthy": adapter.health_check() if adapter else False,
            }

        return {
            "status": "ok",
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpus": gpus,
            "device": self.device,
            "loaded_models": list(self._adapters.keys()),
            "models": models_status,
        }
