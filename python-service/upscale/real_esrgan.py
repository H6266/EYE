"""Real-ESRGAN 4x upscaling (placeholder).

Actual implementation requires the Real-ESRGAN package. This stub provides
the interface and raises NotImplementedError until the dependency is added.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image

logger = logging.getLogger("real_esrgan")


class RealESRGANUpscaler:
    """4x image upscaling using Real-ESRGAN."""

    def __init__(self, model_id: str = "Real-ESRGAN", device: str | None = None):
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import realesrgan  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("Real-ESRGAN package not installed; upscaling unavailable")

    def upscale(self, image_base64: str, scale: int = 4) -> dict[str, Any]:
        """Upscale a base64-encoded image by the given scale factor."""
        if not self._available:
            return {
                "error": "Real-ESRGAN not installed. Add 'realesrgan' and 'basicsr' to requirements.",
                "upscaled_base64": None,
            }

        try:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Placeholder: actual Real-ESRGAN inference would go here
            # from realesrgan import RealESRGANer
            # upsampler = RealESRGANer(scale=scale, model_path=..., device=self.device)
            # output = upsampler.enhance(image)[0]

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            upscaled_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            return {
                "upscaled_base64": f"data:image/png;base64,{upscaled_b64}",
                "scale": scale,
                "original_size": [image.width, image.height],
                "upscaled_size": [image.width, image.height],
                "error": None,
            }
        except Exception as exc:
            logger.exception("Upscaling failed")
            return {"error": str(exc), "upscaled_base64": None}
