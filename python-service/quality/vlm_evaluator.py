"""VLM-based quality verification for generated images.

Uses Qwen3-VL-4B-Instruct (primary) to compare a generated image against
its original prompt and structured scene spec. Returns pass/fail per criterion
with confidence scores and recommendations.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger("vlm_evaluator")

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


class VLMQualityEvaluator:
    """Verifies generated image quality against prompt using a VLM."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
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
            logger.info(f"Loading VLM evaluator: {self.model_id}")
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cuda" and not hasattr(self.model, "hf_device_map"):
                self.model.to(self.device)
            self._loaded = True
            logger.info(f"VLM evaluator loaded: {self.model_id}")
        except Exception as exc:
            logger.warning(f"VLM evaluator load failed: {exc}")
            self._loaded = False
        return self._loaded

    def verify(
        self,
        image_base64: str,
        prompt: str,
        scene_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify a generated image against its prompt.

        Returns a dict matching the QualityVerifyResponse schema.
        Never raises — returns a safe default on any error.
        """
        if not self._load_attempted:
            self._try_load()

        if not self._loaded or self.model is None or self.processor is None:
            return self._default_response("VLM model not available")

        try:
            # Decode base64 image
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Build verification prompt
            subjects = ""
            if scene_spec and scene_spec.get("subjects"):
                subjects = "\n".join(
                    f"- {s.get('type', 'object')} (count: {s.get('count', 1)})"
                    for s in scene_spec["subjects"]
                )

            verification_prompt = (
                "You are a strict image quality verifier. Compare the generated image "
                "against the original prompt and scene spec. Check for:\n"
                "1. All requested subjects present with correct counts\n"
                "2. Correct colors, clothing, attributes\n"
                "3. Correct camera angle, framing, lighting\n"
                "4. Correct environment and background\n"
                "5. No unrequested additions or hallucinated elements\n\n"
                f"Original prompt: {prompt}\n"
                f"Expected subjects:\n{subjects}\n\n"
                "Output JSON: {\"overall_pass\": bool, \"confidence\": 0-1, "
                "\"criteria\": [{\"name\": str, \"pass\": bool, \"confidence\": 0-1}], "
                "\"missing_subjects\": [str], \"incorrect_attributes\": [str], "
                "\"unrequested_additions\": [str], \"recommendations\": [str]}"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": verification_prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
            )
            generated = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            # Parse JSON from response
            import json as _json
            json_str = self._extract_json(generated)
            if json_str:
                parsed = _json.loads(json_str)
                return {
                    "overall_pass": bool(parsed.get("overall_pass", False)),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "criteria": parsed.get("criteria", []),
                    "missing_subjects": parsed.get("missing_subjects", []),
                    "incorrect_attributes": parsed.get("incorrect_attributes", []),
                    "unrequested_additions": parsed.get("unrequested_additions", []),
                    "recommendations": parsed.get("recommendations", []),
                }

            return self._default_response("Failed to parse VLM output")

        except Exception as exc:
            logger.exception("VLM verification failed")
            return self._default_response(str(exc))

    @staticmethod
    def _extract_json(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _default_response(reason: str) -> dict[str, Any]:
        return {
            "overall_pass": False,
            "confidence": 0.0,
            "criteria": [],
            "missing_subjects": [],
            "incorrect_attributes": [],
            "unrequested_additions": [],
            "recommendations": [f"Verification skipped: {reason}"],
            "error": reason,
        }
