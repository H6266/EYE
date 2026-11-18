# EYE — Revised Image Generation Intelligence Upgrade Plan

**Audit date:** 2026-11-18  
**Status:** Core API wiring and lightweight inspection endpoint implemented; no model winner selected; GPU benchmarks remain outstanding.  
**Deployment assumptions from user:** Kaggle for initial experiments; hybrid local/cloud production; free initial budget; access to a 48 GB+ GPU. Confirm that the 48 GB+ device is available for benchmarking and/or production before relying on it.

This document supersedes the earlier model rankings and implementation sequencing in the dated research and integration plans. It does not claim that any candidate has been benchmarked or that the current Python adapters are production-ready.

## 1. Executive conclusion

Keep SDXS-512 as the working, lightweight fallback. Benchmark Z-Image-Turbo and FLUX.2 [klein] 4B first on an available 16 GB Kaggle GPU; test Qwen-Image-2512 and Qwen-Image-Edit-2511 on the available 48 GB+ device. Treat Boogu-Image-0.1 as an experimental candidate: its official repository calls it a research project and says it is not an official model release. Do not select or route to a “best” model until the same prompt suite has been generated and evaluated on the intended hardware.

The strongest near-term architecture is a model-agnostic inference service behind the existing Node.js application, with:

1. A capability- and hardware-aware router, not a prompt-length heuristic.
2. The original prompt preserved verbatim; structured interpretation is optional and user-visible.
3. Optional vision-based discrepancy reports, clearly labeled as estimates rather than ground truth.
4. Editing as a separate task path, using a model with documented reference-image support.
5. Explicit status, progress, cancellation, errors, model unloading, and opt-in fallback behavior.

The current Codespace is not a GPU experiment environment: it has 2 CPU cores and 7.8 GiB RAM, no visible NVIDIA tooling/GPU, and no PyTorch installation. No Kaggle GPU runtime was available to this session. Thus real generation experiments cannot be reported from this environment.

## 2. Repository and hardware audit

### Verified repository state

- The Node/React application has a functioning SDXS-512 ONNX path in `server.ts` and `src/engine/localDiffusionPipeline.ts`.
- Python adapters, a FastAPI application, a model registry, a prompt parser, and quality-verification modules already exist under `python-service/`. These are more than the earlier plans' “not yet implemented” descriptions imply.
- The TypeScript engine IDs and router now connect `/api/media/image` to the Python service when a Python engine is selected; SDXS remains the default. The Python client maps UI IDs to service IDs, handles snake_case responses, and reports unavailable engines instead of silently substituting SDXS.
- The Python service serializes generation and loads at most one heavyweight generator at a time. GPU-only adapters return a clear service error on CPU-only hosts. Z-Image, FLUX, and Qwen adapters have been updated to use CPU offload without subsequently moving the entire pipeline to CUDA.
- Florence-2 base-ft weights and a local inference bundle are present under the ignored `models/weights/helpers/florence2/` directory. A lazy `/api/quality/objects` endpoint proxies object/caption inspection when the Python service is available. This is a helper signal, not a complete prompt-fidelity verifier.
- Official repositories, model cards, and Hugging Face metadata are stored under the ignored `models/sources/` and `models/metadata/` directories. Large text-to-image and edit checkpoints were not downloaded due to workspace capacity.
- No `benchmark_results/` files were found. The repository has benchmark scripts and review scaffolding, but no generated results, scored human reviews, contact sheets, or completed benchmark comparison report.
- The repository's earlier candidate speed, fidelity, and VRAM tables are estimates or model-card claims, not EYE measurements.
- The benchmark prompt source says 33 prompts; its descriptions and the documents' reported category counts are not fully consistent. Normalize and validate the suite before collecting scores.

### Remaining blockers before a meaningful benchmark or production integration

1. **Dependency versions are stale for listed pipelines:** `python-service/requirements.txt` pins Diffusers 0.33.0 and Transformers 4.48.0, while the referenced official setup instructions require newer/current Diffusers for FLUX.2 klein, Qwen Image, and Z-Image. Resolve compatible, reproducible versions before attempting to load these models.
2. **Prompt/VLM loading is not production-ready:** the parser loads a large Qwen3-VL model through `AutoModelForCausalLM`; the evaluator uses `Qwen2VLForConditionalGeneration`; both need validation against Qwen3-VL's official classes and processor examples. Avoid loading these models per request or concurrently with a generator.
3. **Qwen edit and Boogu are research-only in this app:** their source and metadata are available, but no generation/edit adapters are registered. Implement each only after the exact pipeline API and supported reference-image contract are tested.
4. **GPU inference remains unverified:** adapter load, CPU offload, peak memory, image output, and speed have not been tested on CUDA. Python dependencies are not installed in this Codespace, so the FastAPI service was syntax-checked only.
5. **Benchmark harness needs a preflight:** verify per-device precision, smoke tests, output schemas, and peak-memory collection. Fix the Z-Image script's unconditional-true GPU precision predicate and test heterogeneous multi-GPU hosts before using its results.

These are integration/preflight issues, not evidence that the underlying candidate models perform poorly.

The TypeScript client contract tests pass and the web build succeeds. The existing TypeScript check still reports two errors in `src/data/okwankyerefoPaProject.ts` about `TimelineClip.transcript`; these are unrelated to the engine integration.

## 3. Model shortlist and verified source facts

| Candidate | Proposed role | Verified facts from official sources | Current decision |
|---|---|---|---|
| [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Fast text-to-image candidate | 6B family; Turbo is distilled to 8 NFEs; official card claims it fits 16 GB consumer devices; Apache-2.0 metadata. The current card describes the wider Z-Image family as including generation/edit variants, but verify that the exact checkpoint and pipeline are publicly downloadable before adding them to scope. | First-wave benchmark; not yet a winner. |
| [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Fast generation and reference editing | 4B rectified-flow model; supports text-to-image and multi-reference editing; card says about 13 GB VRAM; Apache-2.0. Official Diffusers usage specifies 4 steps and guidance 1.0. | First-wave benchmark and leading editing candidate. |
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | Higher-quality T2I candidate | Official card describes the December 2025 update and its improvements to realism, detail, and text rendering; Apache-2.0 metadata; official example uses 50 steps and 7 listed aspect ratios. Model files are substantially larger than the lightweight candidates. | Benchmark on the 48 GB+ GPU; not a Kaggle-16-GB assumption. |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Image editing and consistency | `QwenImageEditPlusPipeline`; the official example accepts multiple input images and uses 40 steps; Apache-2.0 metadata. The card reports character-consistency improvements. | Separate editing benchmark; do not treat as a T2I replacement. |
| [Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) | Accelerated Qwen T2I test | This is an adapter for the Qwen-Image-2512 base, not a standalone substitute. The Lightning project documents a matching 4-step 2512 adapter and BF16/FP32 variants; it separately documents an Edit-2511 Lightning adapter. | Test only with the matching base, exact adapter revision, supported pipeline, and documented settings. Compare against the unadapted base. |
| [Boogu-Image-0.1](https://github.com/Boogu-Project/Boogu-Image) | Research benchmark candidate | The official repository lists Base, Turbo, Edit, and Edit-Turbo families and Apache-2.0, but explicitly calls the work a research project and says it is not an official model release. Its Edit page notes one reference image for the initial edit model; newer edit-turbo revisions have their own caveats. | Optional experimental smoke test after the first-wave scripts are stable; no production commitment. |

For interpretation and verification, evaluate [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) as a visual-language evaluator, [Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft) for object/caption/OCR assistance, and [SAM 2](https://github.com/facebookresearch/sam2) only where segmentation/masks enable an actual edit. These tools are complementary; none guarantees correct image generation or reliable object counting by itself. Keep machine-generated assessments distinct from human ground truth.

All license statements above describe the cited model-card/repository metadata, not a legal review of every dependency, adapter, or user-supplied LoRA. Recheck the exact downloaded revision and all derivative/adaptor licenses before commercial deployment.

## 4. Quantization and compute strategy

- **16 GB Kaggle class (T4/P100):** start with each model's officially supported FP16 path. T4/P100 do not support BF16; two 16 GB cards do not make a single 32 GB memory pool. CPU offload can make a model load, but increases RAM use and can materially change speed; record the actual configuration.
- **48 GB+ target:** test native BF16 on supported hardware and FP16 where required. Record GPU model, driver, runtime, precision, peak allocated and reserved memory, and actual generation time.
- **FP8:** test only on hardware/kernel combinations that support the required FP8 format and the specific architecture. Do not assume it works on T4/P100 or that generic FP8 weights are interchangeable.
- **4-bit:** prioritize a publisher-supported, architecture-matched checkpoint or documented backend (for example, test a Qwen-compatible Nunchaku build only with its stated Qwen/Lightning setup). Do not apply generic bitsandbytes, TorchAO, GGUF, or ComfyUI quantization to a model unless that exact pipeline supports it. Compare quality and memory to its unquantized base.
- **Lightning:** it reduces inference steps through model-specific adapters; it is not weight-only quantization and must match its base model and configuration.
- **Experiment discipline:** do not load multiple heavyweight generation or VLM models together on a single GPU unless the measured memory headroom supports it. Initially unload the generator before optional vision verification, or use a separately provisioned verifier.

## 5. Target architecture

```text
React studio
  -> Express API (Node/TypeScript)
      -> generation router (capabilities + user choice + hardware/model availability)
          -> existing SDXS-512 ONNX fallback
          -> Python inference API (one active heavyweight model initially)
              -> model adapters: Z-Image-Turbo / FLUX.2 klein / Qwen-Image
              -> editing adapters added only after separate validation
      -> optional prompt interpreter (returns structured scene + original prompt)
      -> optional visual verification (discrepancy report, never a hidden pass/fail gate)
      -> optional targeted edit and re-verification
```

### Routing and fallback

- Return model capabilities, hardware requirement bands, and availability from the backend; the frontend must not advertise an unavailable model as ready.
- Start with explicit user selection and simple capability checks. A premium or editing request may not silently degrade to SDXS. Make fallback visible, preserve the selected model in history, and report which model actually generated each result.
- Load models lazily; allow one heavyweight pipeline at a time until concurrency tests prove the memory budget. Provide unload, load status, progress, cancellation, and actionable OOM errors.
- Keep inference requests asynchronous/queued and return job IDs for long-running work. Persist outputs/history according to the existing application design, not as transient notebook URLs.
- In production, use a persistent GPU endpoint for the cloud path. Kaggle is for experiments, not an always-on API. A user-owned 48 GB+ GPU can be the first premium target if confirmed; local SDXS remains available without GPU.

### Prompt interpretation

- Preserve the original prompt byte-for-byte; never replace it in history.
- Return a validated structured scene: entities/counts, attributes, actions, spatial relationships, foreground/background, camera, environment, lighting, style, must-include and must-not-include constraints.
- Default to faithful extraction, not prompt expansion. Keep any rewritten prompt editable and show it to the user before generation.
- Use a text instruction model for parsing only if benchmarks demonstrate value. Reserve Qwen3-VL for image-conditioned comparison; do not load a 4B VLM for simple text extraction without a measured benefit.
- Validate the structured result, preserve uncertain details verbatim, and fall back to the untouched prompt on parser errors.

### Verification and editing

- Produce a discrepancy report organized by requested constraint (present/missing/uncertain, evidence, confidence). VLM scores are estimates, not factual guarantees.
- Combine visual-language review with targeted object/OCR tools only for supported categories; have human review for benchmark scores.
- Do not automatically regenerate or edit on a low-confidence report. Let users approve targeted correction and preserve the original image as a separate take.
- Begin editing with FLUX.2 klein's documented multi-reference support and Qwen-Image-Edit-2511 as a separately benchmarked premium option. Add SAM 2 masks, ControlNet, pose/depth, IP-Adapter, LoRA, inpainting, or outpainting only when an exact compatible base/model workflow is proven.

## 6. Benchmark and release gates

### Benchmark protocol

1. Repair and smoke-test each harness before the full suite. Record model repo/revision, pipeline/library versions, hardware, precision, dimensions, steps, guidance, seed, warm-up, load time, generation time, failures, peak VRAM, and process/system RAM.
2. Use the same curated prompts and comparable aspect ratios across models. Retain each model's documented settings; do not force a shared step count that invalidates a distilled model. Use multiple seeds for a small subset to expose variance.
3. Report SDXS at its native 512 output separately from high-resolution model results. If comparing appearance at a common size, publish both native and consistently downscaled comparisons.
4. Blind human reviewers to model identity. Score instruction fidelity first (subjects, counts, attributes/colors, positions, composition, background, prohibited additions), then visual quality. Keep unscored records unscored.
5. Run Qwen-Image(-2512), Qwen-Image-Edit, and any quantized variants on the 48 GB+ device where possible. Maintain a distinct editing suite with source image, requested edit, and preservation criteria.
6. Save generated images, metadata JSON, review forms, and comparison reports outside Git; never commit weights, tokens, or credentials.

### Go/no-go gates

- **Gate A — Harness:** every candidate's smoke test succeeds and emits validated images and complete metadata; no benchmark score is produced before this gate.
- **Gate B — Comparative evidence:** complete a blinded human comparison against SDXS and at least one lightweight candidate. Publish actual, measured fidelity and resource results; do not select a winner from vendor claims.
- **Gate C — Integration:** contract tests cover model ID mapping, response serialization, no-GPU health, timeout/cancellation, unload, OOM, and explicit fallback behavior.
- **Gate D — Production:** test on the actual deployment GPU, measure cold/warm latency, memory, failure recovery, and cost. Verify all exact model/adaptor licenses and revision pins.

## 7. Incremental implementation sequence

1. **Preflight and audit existing code:** fix/validate benchmark placement and metadata; reconcile the model ID and snake_case/camelCase API contracts; connect the Express image route to the router; pin a compatible tested dependency set. Add focused smoke/contract tests.
2. **Run first-wave benchmarks:** Z-Image-Turbo and FLUX.2 klein 4B on Kaggle, with SDXS as a distinctly labeled baseline. Do not download all model families in advance.
3. **Run premium and quantization experiments:** use the 48 GB+ GPU for Qwen-Image-2512 and matching Lightning; compare native precision to only compatible quantized variants. Run Qwen-Image-Edit and Boogu in separate editing/experimental suites.
4. **Integrate the proven generator:** implement the shared adapter request/result contract, lazy load/unload, progress/cancellation, model availability, explicit fallback, and frontend engine status. Preserve SDXS behavior.
5. **Add faithful prompt interpretation:** structured parse, editable preview, original-prompt preservation; ship only if tests and human comparisons show it helps.
6. **Add optional visual verification:** discrepancy report with calibrated uncertainty, separate resource lifecycle, user-controlled targeted edits.
7. **Deploy:** maintain local fallback and connect to a persistent GPU server for premium generation. Do not make Kaggle notebooks part of the production request path.

## 8. Source references

- [Z-Image official repository](https://github.com/Tongyi-MAI/Z-Image) and [Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [FLUX.2 official repository](https://github.com/black-forest-labs/flux2) and [Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Qwen-Image official repository](https://github.com/QwenLM/Qwen-Image), [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512), and [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image Lightning documentation](https://github.com/ModelTC/Qwen-Image-Lightning) and [2512 Lightning model card](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)
- [Boogu-Image official repository](https://github.com/Boogu-Project/Boogu-Image)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft), and [SAM 2](https://github.com/facebookresearch/sam2)
