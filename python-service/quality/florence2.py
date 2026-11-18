"""Florence-2 adapter for object detection, captioning, and OCR.

Supports quality verification by providing object detection counts and
text detection in generated images.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger("florence2")

MODEL_ID = "microsoft/Florence-2-base-ft"
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "weights" / "helpers" / "florence2"
MODEL_SOURCE = os.getenv(
    "FLORENCE2_MODEL",
    str(LOCAL_MODEL_DIR) if (LOCAL_MODEL_DIR / "model.safetensors").is_file() else MODEL_ID,
)


class Florence2Adapter:
    """Florence-2 for object detection, dense captioning, and OCR."""

    def __init__(self, model_id: str = MODEL_SOURCE, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_attempted = False

    def _try_load(self) -> bool:
        if self._load_attempted:
            return self._loaded
        self._load_attempted = True
        try:
            logger.info(f"Loading Florence-2: {self.model_id}")
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cuda" and not hasattr(self.model, "hf_device_map"):
                self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"Florence-2 loaded: {self.model_id}")
        except Exception as exc:
            logger.warning(f"Florence-2 load failed: {exc}")
            self._loaded = False
        return self._loaded

    def detect_objects(
        self,
        image_base64: str,
        object_classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run object detection on a base64-encoded image."""
        if not self._load_attempted:
            self._try_load()
        if not self._loaded:
            return {"error": "Florence-2 not available", "detections": []}

        try:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            task = "<OD>" if object_classes else "<CAPTION>"
            text_input = task if not object_classes else f"<OD> {', '.join(object_classes)}"

            inputs = self.processor(text=text_input, images=image, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height),
            )
            return {"detections": parsed, "error": None}

        except Exception as exc:
            logger.exception("Florence-2 detection failed")
            return {"error": str(exc), "detections": []}

    def caption(self, image_base64: str) -> dict[str, Any]:
        """Generate a detailed caption for the image."""
        return self.detect_objects(image_base64, object_classes=None)
