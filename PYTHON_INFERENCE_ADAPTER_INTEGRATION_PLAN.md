# AUTEUR STUDIO — PYTHON INFERENCE ADAPTER INTEGRATION PLAN

**Status:** SPECIFICATION (not yet implemented)  
**Date:** 2026-11-16  
**Part of:** Image Generation Intelligence Upgrade (Stage 9)

---

## 1. ARCHITECTURE OVERVIEW

Auteur Studio's existing stack is Node.js + Express + ONNX Runtime (SDXS-512 local CPU). The new open-weight models (Z-Image-Turbo, FLUX.2 klein 4B, Qwen-Image-2512, MOSS-Image) require PyTorch + Diffusers and run on GPU (Kaggle T4x2/P100 or local GPU). They cannot run in the Node.js process.

**Architecture:** Two-process design

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTEUR STUDIO (Node.js + Express)             │
│  server.ts  •  Express API  •  Frontend React (Vite)            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP JSON (localhost / remote)
                      │  port 8100 (Python service)
┌─────────────────────▼───────────────────────────────────────────┐
│              PYTHON INFERENCE SERVICE (FastAPI + Diffusers)      │
│  main.py  •  /v1/models  •  /v1/generate  •  /v1/parse-prompt   │
│  torch  •  diffusers  •  transformers  •  GPU (CUDA)             │
└─────────────────────────────────────────────────────────────────┘
```

**Design principles:**
- SDXS-512 ONNX stays in-process as local fallback (zero-config, works without GPU)
- Python service is optional: generation falls back to SDXS-512 if service is unreachable
- All new models accessed via `engineId` values: `python-zimage-turbo`, `python-flux-klein`, `python-qwen-2512`, `python-moss-sft`
- Prompt understanding (Qwen2.5-0.5B) runs in the same Python service

---

## 2. NEW IMAGE ENGINE IDS

Add to `ImageEngineId` union in `src/types/studio.ts`:

```typescript
export type ImageEngineId =
  | 'local-sdxs'
  | 'local-sdxs-cfg'
  | 'local-sdxs-img2img'
  | 'python-zimage-turbo'
  | 'python-flux-klein'
  | 'python-qwen-2512'
  | 'python-moss-sft';
```

Each new engine spec is registered in `src/data/imageEngines.ts` with metadata (VRAM requirement, steps, resolution, etc.).

---

## 3. PYTHON INFERENCE SERVICE

### 3.1 File Structure

```
python-service/
├── main.py              # FastAPI app, route definitions
├── models.py            # Pydantic request/response models
├── engine_loader.py     # Lazy model loading, VRAM management
├── zimage_turbo.py      # Z-Image-Turbo adapter
├── flux_klein.py        # FLUX.2 klein 4B adapter
├── qwen_2512.py         # Qwen-Image-2512 adapter
├── moss_sft.py          # MOSS-Image-SFT adapter
├── prompt_parser.py     # Qwen2.5-0.5B prompt understanding
├── requirements.txt     # fastapi, uvicorn, torch, diffusers, transformers, PIL
└── .env                 # MODEL_CACHE_DIR, PORT, ALLOWED_ORIGINS
```

### 3.2 Dependencies (`requirements.txt`)

```
fastapi==0.115.0
uvicorn==0.32.1
torch==2.7.0
torchvision==0.22.0
diffusers==0.33.0
transformers==4.48.0
Pillow==11.0.0
pydantic==2.9.2
python-multipart==0.0.9
```

### 3.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /v1/health` | GET | Service health, loaded models, GPU status |
| `GET /v1/models` | GET | List available generation models with metadata |
| `POST /v1/generate` | POST | Generate image with specified model |
| `POST /v1/parse-prompt` | POST | Parse prompt into structured scene spec (Qwen2.5-0.5B) |
| `POST /v1/optimize-prompt` | POST | Optimize prompt for target model |

### 3.4 Request/Response Schemas

**`POST /v1/generate`**

Request:
```json
{
  "model": "z-image-turbo",
  "prompt": "A glowing lantern airship drifting above misty pine valleys at dusk",
  "negative_prompt": "blurry, distorted anatomy, extra limbs, watermark, low quality",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 8,
  "guidance_scale": 0.0,
  "seed": 42,
  "return_base64": true
}
```

Response:
```json
{
  "image_base64": "iVBORw0KGgo...",
  "image_url": null,
  "seed": 42,
  "model": "z-image-turbo",
  "timings_ms": {
    "model_load": 0,
    "inference": 3200,
    "total": 3200
  },
  "vram_peak_gb": 14.2,
  "prompt_used": "A glowing lantern airship drifting above misty pine valleys at dusk",
  "optimized_prompt": null
}
```

**`POST /v1/parse-prompt`**

Request:
```json
{
  "prompt": "A glowing lantern airship drifting above misty pine valleys at dusk",
  "return_json": true
}
```

Response: Structured scene spec (see `PROMPT_UNDERSTANDING_SPEC.md` section 3)

---

## 4. NODE.JS SERVER INTEGRATION

### 4.1 Python Service Client (`src/engine/pythonDiffusionClient.ts`)

New file that wraps HTTP calls to the Python inference service. Mirrors the `LocalDiffusionEngine` interface.

```typescript
export interface PythonGenerationInput {
  model: string;
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  numInferenceSteps?: number;
  guidanceScale?: number;
  seed?: number;
}

export interface PythonGenerationResult {
  imageDataUrl: string;
  seed: number;
  model: string;
  timingsMs: {
    inference: number;
    total: number;
  };
  vramPeakGb?: number;
  promptUsed: string;
  optimizedPrompt?: string;
}

export class PythonDiffusionClient {
  private baseUrl: string;
  private available: boolean = false;
  private models: string[] = [];

  constructor(baseUrl: string = 'http://localhost:8100') {}
  
  async checkHealth(): Promise<boolean>;
  async listModels(): Promise<string[]>;
  async generate(input: PythonGenerationInput): Promise<PythonGenerationResult>;
  async parsePrompt(prompt: string): Promise<any>;
  async optimizePrompt(prompt: string, targetModel: string): Promise<string>;
  getStatus(): { available: boolean; models: string[]; baseUrl: string };
}
```

### 4.2 Generation Router (`src/engine/generationRouter.ts`)

New file that routes generation requests to either the local SDXS-512 engine or the Python service based on `engineId`.

```typescript
export async function generateImage(
  input: LocalDiffusionInput | PythonGenerationInput,
  engineId: ImageEngineId
): Promise<LocalDiffusionResult | PythonGenerationResult> {
  const pythonEngines: ImageEngineId[] = [
    'python-zimage-turbo',
    'python-flux-klein',
    'python-qwen-2512',
    'python-moss-sft',
  ];

  if (pythonEngines.includes(engineId)) {
    const client = getPythonClient();
    const available = await client.checkHealth();
    if (!available) {
      // Fall back to local SDXS-512
      return localDiffusionEngine.generate({ ...input, engineId: 'local-sdxs' });
    }
    return client.generate({ ...input, model: engineId });
  }

  return localDiffusionEngine.generate(input);
}
```

### 4.3 Express Route Changes (`server.ts`)

Modify `POST /api/media/image` to accept the new fields and route appropriately:

```typescript
app.post('/api/media/image', async (req: Request, res: Response) => {
  const {
    prompt,
    negativePrompt = 'blurry, distorted anatomy, extra limbs, watermark, low quality',
    engineId = 'local-sdxs',
    aspectRatio = '16:9',
    guidance = 0,
    seed = 0,
    randomizeSeed = true,
    batchCount = 1,
    referenceImagePngBase64,
    img2imgStrength = 0.78,
    usePromptUnderstanding = false,        // NEW
    optimizedPrompt,                        // NEW
  } = req.body ?? {};

  // ... validation ...

  const pythonEngineIds: ImageEngineId[] = [
    'python-zimage-turbo',
    'python-flux-klein',
    'python-qwen-2512',
    'python-moss-sft',
  ];

  const takes = [];
  for (let i = 0; i < safeCount; i++) {
    const takeSeed = baseSeed + i * 137;

    let finalPrompt = prompt.trim();
    if (usePromptUnderstanding && optimizedPrompt) {
      finalPrompt = optimizedPrompt;
    }

    const result = pythonEngineIds.includes(engineId)
      ? await pythonClient.generate({
          model: engineId,
          prompt: finalPrompt,
          negativePrompt,
          width: aspectRatio === '9:16' ? 512 : 1024,
          height: aspectRatio === '16:9' ? 576 : 1024,
          numInferenceSteps: engineId === 'python-zimage-turbo' ? 8 : 
                           engineId === 'python-flux-klein' ? 4 : 50,
          guidanceScale: engineId === 'python-zimage-turbo' ? 0 : 1.0,
          seed: takeSeed,
        })
      : await localDiffusionEngine.generate({
          prompt: finalPrompt,
          negativePrompt,
          guidanceScale: effectiveGuidance,
          seed: takeSeed,
          aspectRatio,
          referenceImagePngBase64: useReferenceLatent ? String(referenceImagePngBase64) : undefined,
          img2imgStrength,
        });

    takes.push({ ...result, model: engineId, ... });
  }

  res.json({ ... });
});
```

### 4.4 Health Endpoint Update

Extend `GET /api/health` to include Python service status:

```typescript
app.get('/api/health', async (_req, res) => {
  const pythonStatus = await pythonClient.getStatus();
  res.json({
    ok: true,
    hasApiKey: hasKey,
    localDiffusion: localDiffusionEngine.getStatus(),
    pythonService: pythonStatus,
  });
});
```

---

## 5. FRONTEND INTEGRATION (GeneratePanel.tsx)

### 5.1 New State & UI

Add to `GeneratePanel.tsx`:
- `usePromptUnderstanding` toggle (boolean state)
- `showOptimizedPrompt` (boolean state) — show original vs optimized side-by-side
- `pythonEngineAvailable` (from `/api/health`)
- `optimizedPrompt` (string state)

### 5.2 Engine Selector Updates

Add new options to the engine dropdown:
- `python-zimage-turbo` — "Z-Image-Turbo (Python GPU)" — 6B, ~8 steps, 1024px, Apache-2.0
- `python-flux-klein` — "FLUX.2 klein 4B (Python GPU)" — 4B, ~4 steps, 1024px, Apache-2.0
- `python-qwen-2512` — "Qwen-Image-2512 (Python GPU)" — 2.5B, 1-2 steps, 1344px, Apache-2.0
- `python-moss-sft` — "MOSS-Image-SFT (Python GPU)" — 5B, ~50 steps, 1024px, Apache-2.0

### 5.3 Prompt Understanding Flow

1. User enters prompt
2. User toggles "Prompt Understanding" on
3. Frontend calls `POST /api/prompts/parse-prompt` (Node proxy → Python service)
4. Python service returns structured scene spec + optimized prompt
5. Frontend shows original vs optimized prompt side-by-side
6. User can edit optimized prompt before generating
7. Generation uses optimized prompt

---

## 6. FALLBACK & ERROR HANDLING

| Scenario | Behavior |
|----------|----------|
| Python service unreachable | Fall back to `local-sdxs` (SDXS-512 ONNX CPU) |
| Python service returns error | Return 502 with error message; suggest retry |
| Model not loaded in Python service | Auto-load on first request (may take 30-120s) |
| VRAM OOM on Python service | Return error; suggest smaller model or lower resolution |
| Prompt parsing fails | Use original prompt unchanged; log error |

---

## 7. CONFIGURATION

### 7.1 Environment Variables (`server.ts`)

```
PYTHON_SERVICE_URL=http://localhost:8100   # Python inference service endpoint
PYTHON_SERVICE_ENABLED=true                 # Enable/disable Python routing
PYTHON_SERVICE_TIMEOUT_MS=300000           # Request timeout (5 min for model load)
```

### 7.2 Python Service Environment (`.env`)

```
PORT=8100
MODEL_CACHE_DIR=/root/.cache/huggingface
CUDA_VISIBLE_DEVICES=0
ENABLE_QWEN_2512=false                      # Disabled by default (needs 24-48GB VRAM)
ENABLE_ZIMAGE_TURBO=true
ENABLE_FLUX_KLEIN=true
ENABLE_MOSS_SFT=true
```

---

## 8. IMPLEMENTATION PHASES

### Phase 1: Python Service Scaffold (Week 1)
- Create `python-service/` directory with FastAPI app
- Implement `/v1/health` and `/v1/models` endpoints
- Implement Z-Image-Turbo adapter with Diffusers
- Implement FLUX.2 klein 4B adapter
- Test on Kaggle T4x2

### Phase 2: Node.js Integration (Week 2)
- Create `src/engine/pythonDiffusionClient.ts`
- Create `src/engine/generationRouter.ts`
- Modify `server.ts` `/api/media/image` route
- Add Python service health check
- Add new engine IDs to types and `imageEngines.ts`

### Phase 3: Frontend & Prompt Understanding (Week 3)
- Add prompt understanding toggle to `GeneratePanel.tsx`
- Add original vs optimized prompt display
- Add new engine options to dropdown
- Implement `/api/prompts/parse-prompt` proxy route
- Qwen2.5-0.5B prompt parser in Python service

### Phase 4: Testing & Benchmarking (Week 4)
- Run Kaggle benchmarks (Z-Image-Turbo, FLUX.2 klein, MOSS-Image)
- Compare against SDXS-512 baseline using `benchmark_framework.py`
- Human evaluation of prompt fidelity
- Final recommendation

---

## 9. DEPLOYMENT OPTIONS

### Option A: Local Development (CPU-only Codespace)
- Python service runs on local CPU with `enable_model_cpu_offload()`
- Slow (~30-60s/image) but functional for development
- SDXS-512 remains default for fast local iteration

### Option B: Kaggle GPU (T4x2 / P100)
- Python service runs on Kaggle notebook
- Node.js server connects remotely via `PYTHON_SERVICE_URL`
- Fast generation (~4-8s/image for turbo models)

### Option C: Production (Docker)
- Python service containerized with CUDA base image
- Deployed alongside Node.js on GPU-enabled instance
- Model weights cached in volume

---

## 10. FILES TO CREATE / MODIFY

### Create:
1. `python-service/main.py`
2. `python-service/models.py`
3. `python-service/engine_loader.py`
4. `python-service/zimage_turbo.py`
5. `python-service/flux_klein.py`
6. `python-service/qwen_2512.py`
7. `python-service/moss_sft.py`
8. `python-service/prompt_parser.py`
9. `python-service/requirements.txt`
10. `python-service/.env.example`
11. `src/engine/pythonDiffusionClient.ts`
12. `src/engine/generationRouter.ts`

### Modify:
1. `src/types/studio.ts` — Add new `ImageEngineId` union members
2. `src/data/imageEngines.ts` — Add new engine specs
3. `server.ts` — Add Python service client, modify `/api/media/image`, extend `/api/health`, add `/api/prompts/parse-prompt` proxy
4. `src/components/GeneratePanel.tsx` — Add prompt understanding UI, new engine options
5. `package.json` — No new deps needed (Python service is separate)

---

## 11. VRAM REQUIREMENTS (Kaggle T4x2 / P100: 16GB)

| Model | VRAM (FP16/BF16) | VRAM (CPU offload) | Steps | Resolution |
|-------|-----------------|-------------------|-------|------------|
| Z-Image-Turbo | ~13GB | ~4GB | 8 | 1024×1024 |
| FLUX.2 klein 4B | ~11GB | ~3GB | 4 | 1024×1024 |
| MOSS-Image-SFT | ~12GB | ~4GB | 50 | 1024×1024 |
| Qwen-Image-2512 | ~28GB | N/A | 1-2 | 1344×1344 |
| SDXS-512 (baseline) | ~2GB (ONNX CPU) | N/A | 4-8 | 512×512 |

Qwen-Image-2512 does not fit in 16GB VRAM even quantized. Disabled by default.