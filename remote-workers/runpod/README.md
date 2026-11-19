# RunPod worker template

This worker is a remote deployment artifact. It is not imported by the Auteur Studio app and its model dependencies must only be installed in the remote GPU image. No model weights or inference packages are downloaded into Codespaces.

## Deploy

1. Create a GPU RunPod Serverless endpoint with Python, CUDA-compatible PyTorch, and the dependencies required by the chosen model.
2. Copy `handler.py` into the worker image and configure it as the RunPod handler. `runpod.serverless.start` starts the worker.
3. Use these worker-side packages as a starting point, pin tested CUDA/model versions in your deployment image:

   ```text
   runpod
   torch
   diffusers
   transformers
   accelerate
   pillow
   ```

4. Configure model cache storage on the worker, GPU memory/precision, health checks, timeouts, concurrency, scaling-to-zero, and any gated-model credentials there.
5. Add `ALLOWED_IMAGE_HOSTS` as a comma-separated allowlist only if the worker should fetch HTTPS reference-image URLs. Data-URI inputs do not need that setting.
6. Configure the app's backend `RUNPOD_API_KEY` and corresponding generation, Florence, or SAM2 endpoint ID. Do not put these in browser environment variables.

The handler uses lazy imports and `lru_cache(maxsize=1)` for image pipelines so generation models are loaded only in the worker. Validate the exact Diffusers/Transformers version against each model revision before deployment. Model licenses and GPU memory requirements are model-specific.

## Worker contract

The app posts RunPod's standard `{ "input": ... }` body. The normalized input includes:

- `task`: `image.generate`, `image.edit`, `image.analyze`, or `image.segment`
- `model_id`, `repository`, `prompt`, `instructions`, `reference_image`, and `parameters`

Image workers return `{ "image_base64": "...", "content_type": "image/png" }`; analyze/segment workers return JSON. The app extracts image data into a short-lived result URL instead of storing the base64 in its job record. For production payloads, replace inline outputs with RunPod's configured object-storage integration and return a durable HTTPS `image_url`.

The example's model implementations are deployment starting points, not a claim that the worker has been built or tested against a live GPU endpoint. In particular:

- Qwen Lightning loads `Qwen/Qwen-Image-2512` and applies `lightx2v/Qwen-Image-2512-Lightning`.
- Florence-2 uses `trust_remote_code=True`; review and pin the upstream code before running it.
- SAM2's automatic `mask-generation` pipeline provides masks but does not yet interpret point/box prompts. Extend `_segment` for the worker's desired interactive mask API.
- The FLUX edit worker depends on the selected Diffusers version exposing image conditioning for `Flux2KleinPipeline`.

RunPod job status is polled by the backend; the frontend never calls RunPod directly.
