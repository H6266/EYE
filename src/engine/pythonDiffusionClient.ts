import { ImageEngineId, PYTHON_ENGINE_IDS } from '../types/studio';

const SERVICE_MODEL_IDS: Partial<Record<ImageEngineId, string>> = {
  'python-zimage-turbo': 'z-image-turbo',
  'python-flux-klein': 'flux-klein-4b',
  'python-zimage-base': 'z-image-base',
  'python-qwen-2512': 'qwen-2512',
  'python-moss-sft': 'moss-sft',
};

interface ApiTimings {
  model_load_ms?: number;
  inference_ms?: number;
  total_ms?: number;
}

interface ApiGenerationResult {
  image_base64?: string | null;
  image_url?: string | null;
  seed?: number;
  model?: string;
  timings?: ApiTimings;
  vram_peak_gb?: number | null;
  ram_peak_gb?: number | null;
  prompt_used?: string;
  optimized_prompt?: string | null;
  error?: string | null;
}

export interface PythonGenerationInput {
  model: string;
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  numInferenceSteps?: number;
  guidanceScale?: number;
  seed?: number;
  returnBase64?: boolean;
}

export interface TimingsMs {
  modelLoadMs: number;
  inferenceMs: number;
  totalMs: number;
}

export interface PythonGenerationResult {
  imageBase64: string | null;
  imageUrl: string | null;
  seed: number;
  model: string;
  timingsMs: TimingsMs;
  vramPeakGb: number | null;
  ramPeakGb: number | null;
  promptUsed: string;
  optimizedPrompt?: string;
  error?: string;
}

export interface PythonHealthStatus {
  available: boolean;
  models: string[];
  baseUrl: string;
  gpuAvailable: boolean;
  gpuCount: number;
  gpus: Array<{ index: number; name: string; vram_gb: number }>;
  loadedModels: string[];
}

export interface PythonObjectDetectionResult {
  detections: Record<string, unknown>;
  error?: string;
}

interface ApiHealth {
  gpu_available?: boolean;
  gpu_count?: number;
  gpus?: PythonHealthStatus['gpus'];
  loaded_models?: string[];
}

interface ApiModelList {
  models?: Array<{ model_id: string; enabled?: boolean }>;
}

export class PythonDiffusionClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private lastHealthCheckAt = 0;
  private status: PythonHealthStatus;
  private readonly healthCacheMs = 5000;

  constructor(baseUrl: string = 'http://localhost:8100', timeoutMs: number = 300000) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
    this.status = {
      available: false,
      models: [],
      baseUrl: this.baseUrl,
      gpuAvailable: false,
      gpuCount: 0,
      gpus: [],
      loadedModels: [],
    };
  }

  async checkHealth(force = false): Promise<boolean> {
    if (!force && Date.now() - this.lastHealthCheckAt < this.healthCacheMs) {
      return this.status.available;
    }

    this.lastHealthCheckAt = Date.now();
    try {
      const [healthResponse, modelsResponse] = await Promise.all([
        fetch(`${this.baseUrl}/v1/health`, { signal: AbortSignal.timeout(5000) }),
        fetch(`${this.baseUrl}/v1/models`, { signal: AbortSignal.timeout(5000) }),
      ]);
      if (!healthResponse.ok || !modelsResponse.ok) {
        this.markUnavailable();
        return false;
      }

      const [health, modelList] = (await Promise.all([
        healthResponse.json(),
        modelsResponse.json(),
      ])) as [ApiHealth, ApiModelList];
      this.status = {
        available: true,
        models: (modelList.models ?? [])
          .filter((model) => model.enabled !== false)
          .map((model) => model.model_id),
        baseUrl: this.baseUrl,
        gpuAvailable: health.gpu_available ?? false,
        gpuCount: health.gpu_count ?? 0,
        gpus: health.gpus ?? [],
        loadedModels: health.loaded_models ?? [],
      };
      return true;
    } catch {
      this.markUnavailable();
      return false;
    }
  }

  async listModels(): Promise<string[]> {
    if (!(await this.checkHealth())) return [];
    return [...this.status.models];
  }

  async generate(input: PythonGenerationInput): Promise<PythonGenerationResult> {
    const selectedEngine = input.model as ImageEngineId;
    const serviceModel = SERVICE_MODEL_IDS[selectedEngine];
    if (!PYTHON_ENGINE_IDS.includes(selectedEngine) || !serviceModel) {
      return this.errorResult(input, `Unsupported Python image engine '${input.model}'`);
    }

    if (!(await this.checkHealth())) {
      return this.errorResult(
        input,
        'Python inference service is unavailable. Start the GPU inference service or choose Local SDXS.'
      );
    }
    if (!this.status.models.includes(serviceModel)) {
      return this.errorResult(
        input,
        `Model '${serviceModel}' is not enabled in the Python inference service.`
      );
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}/v1/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: serviceModel,
          prompt: input.prompt,
          negative_prompt: input.negativePrompt ?? '',
          width: input.width ?? 1024,
          height: input.height ?? 1024,
          num_inference_steps: input.numInferenceSteps,
          guidance_scale: input.guidanceScale,
          seed: input.seed ?? 42,
          return_base64: input.returnBase64 ?? true,
        }),
        signal: controller.signal,
      });
      const data = (await response.json().catch(() => ({}))) as ApiGenerationResult & {
        detail?: string;
      };
      if (!response.ok) {
        return this.errorResult(
          input,
          `Python service error (${response.status}): ${data.detail ?? data.error ?? 'unknown error'}`
        );
      }
      if (data.error) return this.errorResult(input, data.error);

      const imageBase64 = data.image_base64 ?? null;
      const imageUrl = data.image_url ?? null;
      if (!imageBase64 && !imageUrl) {
        return this.errorResult(input, 'Python inference service returned no image.');
      }
      return {
        imageBase64,
        imageUrl,
        seed: data.seed ?? input.seed ?? 42,
        model: data.model ?? serviceModel,
        timingsMs: {
          modelLoadMs: data.timings?.model_load_ms ?? 0,
          inferenceMs: data.timings?.inference_ms ?? 0,
          totalMs: data.timings?.total_ms ?? 0,
        },
        vramPeakGb: data.vram_peak_gb ?? null,
        ramPeakGb: data.ram_peak_gb ?? null,
        promptUsed: data.prompt_used ?? input.prompt,
        optimizedPrompt: data.optimized_prompt ?? undefined,
      };
    } catch (error) {
      const message =
        error instanceof Error && error.name === 'AbortError'
          ? `Python inference request timed out after ${this.timeoutMs} ms.`
          : `Python service request failed: ${error instanceof Error ? error.message : String(error)}`;
      return this.errorResult(input, message);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async parsePrompt(prompt: string): Promise<Record<string, unknown>> {
    if (!(await this.checkHealth())) {
      return {
        error: 'Python inference service unavailable',
        original_prompt: prompt,
        optimized_prompt: prompt,
      };
    }

    try {
      const response = await fetch(`${this.baseUrl}/v1/prompt/parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, return_json: true, mode: 'faithful' }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      const data = (await response.json().catch(() => ({}))) as Record<string, unknown>;
      if (!response.ok) {
        return {
          error: `Prompt parsing failed (${response.status}): ${String(data.detail ?? 'unknown error')}`,
          original_prompt: prompt,
          optimized_prompt: prompt,
        };
      }
      return data;
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : String(error),
        original_prompt: prompt,
        optimized_prompt: prompt,
      };
    }
  }

  async optimizePrompt(prompt: string, _targetModel: string): Promise<string> {
    const result = await this.parsePrompt(prompt);
    return typeof result.optimized_prompt === 'string' ? result.optimized_prompt : prompt;
  }

  async detectObjects(imageBase64: string): Promise<PythonObjectDetectionResult> {
    if (!(await this.checkHealth())) {
      return { detections: {}, error: 'Python inference service is unavailable.' };
    }

    try {
      const response = await fetch(`${this.baseUrl}/v1/quality/objects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      const result = (await response.json().catch(() => ({}))) as {
        detections?: Record<string, unknown>;
        detail?: string;
        error?: string;
      };
      if (!response.ok) {
        return {
          detections: {},
          error: result.detail ?? result.error ?? `Image inspection failed (${response.status}).`,
        };
      }
      return { detections: result.detections ?? {} };
    } catch (error) {
      return {
        detections: {},
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  getStatus(): PythonHealthStatus {
    return {
      ...this.status,
      models: [...this.status.models],
      gpus: [...this.status.gpus],
      loadedModels: [...this.status.loadedModels],
    };
  }

  private markUnavailable(): void {
    this.status = {
      ...this.status,
      available: false,
      models: [],
      gpuAvailable: false,
      gpuCount: 0,
      gpus: [],
      loadedModels: [],
    };
  }

  private errorResult(input: PythonGenerationInput, error: string): PythonGenerationResult {
    return {
      imageBase64: null,
      imageUrl: null,
      seed: input.seed ?? 42,
      model: input.model,
      timingsMs: { modelLoadMs: 0, inferenceMs: 0, totalMs: 0 },
      vramPeakGb: null,
      ramPeakGb: null,
      promptUsed: input.prompt,
      error,
    };
  }
}

export const pythonDiffusionClient = new PythonDiffusionClient(
  process.env.PYTHON_SERVICE_URL ?? 'http://localhost:8100',
  Number(process.env.PYTHON_SERVICE_TIMEOUT_MS ?? 300000),
);
