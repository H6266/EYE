"""Abstract base class for all image generation adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationInput:
    """Input parameters for image generation."""

    model: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 42
    return_base64: bool = True
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of image generation."""

    model: str
    prompt_used: str
    image_bytes: bytes | None = None
    image_base64: str | None = None
    seed: int = 42
    inference_time_s: float = 0.0
    vram_peak_gb: float | None = None
    ram_peak_gb: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Metadata about a registered model."""

    model_id: str
    display_name: str
    description: str
    license: str
    min_vram_gb: float
    recommended_vram_gb: float
    default_steps: int
    default_resolution: tuple[int, int]
    supports_cfg: bool
    supports_negative_prompt: bool
    supports_img2img: bool
    supports_ip_adapter: bool
    supports_controlnet: bool
    cpu_fallback: bool = False


class BaseGenerationAdapter(ABC):
    """Abstract base class for image generation adapters."""

    def __init__(self, model_id: str, device: str = "cuda", dtype=None):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.loaded = False
        self.load_time_s: float = 0.0

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free VRAM."""

    @abstractmethod
    def generate(self, input: GenerationInput) -> GenerationResult:
        """Generate an image from the given input."""

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return metadata about this model."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the adapter is healthy and ready."""

    def is_loaded(self) -> bool:
        return self.loaded and self.pipe is not None


def get_torch_dtype(prefer_bf16: bool = True):
    """Return the best available torch dtype for the current GPU."""
    import torch

    if torch.cuda.is_available():
        major = torch.cuda.get_device_properties(0).major
        if prefer_bf16 and major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_system_ram_gb() -> float | None:
    """Return current process RSS in GB, or None if unavailable."""
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)
    except Exception:
        try:
            import psutil

            return round(psutil.Process().memory_info().rss / (1024**3), 2)
        except Exception:
            return None


def get_vram_peak_gb(device_index: int = 0) -> float | None:
    """Return peak VRAM allocated on the given CUDA device in GB."""
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated(device_index) / (1024**3), 3)
    except Exception:
        return None


def reset_vram_peak(device_index: int = 0) -> None:
    """Reset peak VRAM stats for the given CUDA device."""
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device_index)
        except Exception:
            pass


def image_to_bytes(image, format: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    import io

    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def image_to_base64(image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URI."""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{b64}"
