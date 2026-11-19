import base64
import io
import os
import re
import urllib.request
from functools import lru_cache
from urllib.parse import urlparse

import runpod
import torch
from PIL import Image


MODEL_REPOSITORIES = {
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
    "flux-2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux-2-klein-4b-edit": "black-forest-labs/FLUX.2-klein-4B",
    "qwen-image-2512": "Qwen/Qwen-Image-2512",
    "qwen-image-2512-lightning": "Qwen/Qwen-Image-2512",
}


def _image_from_input(reference):
    if not isinstance(reference, dict):
        raise ValueError("reference_image must be an object")

    data = reference.get("data")
    if isinstance(data, str):
        data = re.sub(r"^data:image/[^;]+;base64,", "", data)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    url = reference.get("url")
    if not isinstance(url, str) or urlparse(url).scheme != "https":
        raise ValueError("reference_image requires image data or an HTTPS URL")
    allowed_hosts = {
        host.strip().lower()
        for host in os.environ.get("ALLOWED_IMAGE_HOSTS", "").split(",")
        if host.strip()
    }
    host = (urlparse(url).hostname or "").lower()
    if not allowed_hosts or host not in allowed_hosts:
        raise ValueError("reference image URL host is not in ALLOWED_IMAGE_HOSTS")
    request = urllib.request.Request(url, headers={"Accept": "image/*"})
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=20) as response:
        image_bytes = response.read(12 * 1024 * 1024 + 1)
    if len(image_bytes) > 12 * 1024 * 1024:
        raise ValueError("Reference image exceeds 12 MB")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@lru_cache(maxsize=1)
def _load_pipeline(model_id):
    if model_id not in MODEL_REPOSITORIES:
        raise ValueError(f"Unsupported worker model: {model_id}")

    from diffusers import Flux2KleinPipeline, QwenImagePipeline, ZImagePipeline

    repository = MODEL_REPOSITORIES[model_id]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if model_id == "z-image-turbo":
        pipeline = ZImagePipeline.from_pretrained(repository, torch_dtype=dtype)
    elif model_id.startswith("flux-2-klein"):
        pipeline = Flux2KleinPipeline.from_pretrained(repository, torch_dtype=dtype)
    else:
        pipeline = QwenImagePipeline.from_pretrained(repository, torch_dtype=dtype)
        if model_id == "qwen-image-2512-lightning":
            pipeline.load_lora_weights("lightx2v/Qwen-Image-2512-Lightning")
    if torch.cuda.is_available():
        pipeline.to("cuda")
    return pipeline


def _generate(payload):
    model_id = payload["model_id"]
    task = payload["task"]
    if task not in ("image.generate", "image.edit"):
        raise ValueError(f"Unsupported image task: {task}")
    if task == "image.edit" and not model_id.startswith("flux-2-klein"):
        raise ValueError("This worker only supports FLUX.2 Klein image editing")

    parameters = payload.get("parameters") or {}
    arguments = {
        "prompt": payload.get("prompt", ""),
        "width": int(parameters.get("width", 1024)),
        "height": int(parameters.get("height", 1024)),
        "num_inference_steps": int(parameters.get("steps", 8)),
        "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(
            int(parameters.get("seed", 42))
        ),
    }
    if task == "image.edit":
        arguments["image"] = _image_from_input(payload.get("reference_image"))
    if "guidanceScale" in parameters:
        arguments["guidance_scale"] = float(parameters["guidanceScale"])

    result = _load_pipeline(model_id)(**arguments).images[0]
    output = io.BytesIO()
    result.save(output, format="PNG")
    return {
        "image_base64": base64.b64encode(output.getvalue()).decode("ascii"),
        "content_type": "image/png",
        "model_id": model_id,
    }


def _analyze(payload):
    image = _image_from_input(payload.get("reference_image"))
    task_prompt = payload.get("parameters", {}).get("taskPrompt", "<MORE_DETAILED_CAPTION>")
    model, processor = _load_florence()
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
    generated = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
    )
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    return {"analysis": text, "task_prompt": task_prompt}


def _segment(payload):
    image = _image_from_input(payload.get("reference_image"))
    segmenter = _load_segmenter()
    masks = segmenter(image)
    encoded_masks = []
    for item in masks:
        mask = item["mask"]
        output = io.BytesIO()
        mask.save(output, format="PNG")
        encoded_masks.append(base64.b64encode(output.getvalue()).decode("ascii"))
    return {"masks": encoded_masks}


@lru_cache(maxsize=1)
def _load_florence():
    from transformers import AutoModelForCausalLM, AutoProcessor

    repository = "microsoft/Florence-2-base-ft"
    model = AutoModelForCausalLM.from_pretrained(
        repository, trust_remote_code=True, torch_dtype=torch.float16
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, AutoProcessor.from_pretrained(repository, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_segmenter():
    from transformers import pipeline

    return pipeline(
        "mask-generation",
        model="facebook/sam2-hiera-base-plus",
        device=0 if torch.cuda.is_available() else -1,
    )


def handle_input(payload):
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    task = payload.get("task")
    if task in ("image.generate", "image.edit"):
        expected = MODEL_REPOSITORIES.get(payload.get("model_id"))
        if not expected:
            raise ValueError("Unsupported model_id")
        if payload.get("repository") and payload["repository"] != expected:
            raise ValueError("Model repository does not match the worker registry")
        return _generate(payload)
    if task == "image.analyze" and payload.get("model_id") == "florence-2-base-ft":
        return _analyze(payload)
    if task == "image.segment" and payload.get("model_id") == "sam2-hiera-base-plus":
        return _segment(payload)
    raise ValueError(f"Unsupported task/model combination: {task}/{payload.get('model_id')}")


def handler(job):
    return handle_input(job.get("input") or {})


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
