import { ImageEngineId, PYTHON_ENGINE_IDS } from '../types/studio';

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
  loadedModels: string[];
}

export class PythonDiffusionClient {
  private baseUrl: string;
  private available: boolean = false;
  private models: string[] = [];
  private healthChecked: boolean = false;
  private timeoutMs: number;

  constructor(baseUrl: string = 'http://localhost:8100', timeoutMs: number = 300000) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
  }

  async checkHealth(): Promise<boolean> {
    if (this.healthChecked) {
      return this.available;
    }
    this.healthChecked = true;
    try {
      const res = await fetch(`${this.baseUrl}/v1/health`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) {
        this.available = false;
        return false;
      }
      const data = (await res.json()) as {
        loaded_models: string[];
        gpu_available: boolean;
      };
      this.available = true;
      this.models = data.loaded_models ?? [];
      return true;
    } catch (err) {
      this.available = false;
      return false;
    }
  }

  async listModels(): Promise<string[]> {
    await this.checkHealth();
    return this.models;
  }

  async generate(input: PythonGenerationInput): Promise<PythonGenerationResult> {
    const isPythonEngine = PYTHON_ENGINE_IDS.includes(input.model as ImageEngineId);
    if (!isPythonEngine) {
      return {
        imageBase64: null,
        imageUrl: null,
        seed: input.seed ?? 42,
        model: input.model,
        timingsMs: { modelLoadMs: 0, inferenceMs: 0, totalMs: 0 },
        vramPeakGb: null,
        ramPeakGb: null,
        promptUsed: input.prompt,
        error: `Model '${input.model}' is not a Python service engine`,
      };
    }

    const healthy = await this.checkHealth();
    if (!healthy) {
      return {
        imageBase64: null,
        imageUrl: null,
        seed: input.seed ?? 42,
        model: input.model,
        timingsMs: { modelLoadMs: 0, inferenceMs: 0, totalMs: 0 },
        vramPeakGb: null,
        ramPeakGb: null,
        promptUsed: input.prompt,
        error: 'Python inference service unavailable',
      };
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(`${this.baseUrl}/v1/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: input.model,
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
      clearTimeout(timeoutId);

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        return {
          imageBase64: null,
          imageUrl: null,
          seed: input.seed ?? 42,
          model: input.model,
          timingsMs: { modelLoadMs: 0, inferenceMs: 0, totalMs: 0 },
          vramPeakGb: null,
          ramPeakGb: null,
          promptUsed: input.prompt,
          error: `Python service error (${res.status}): ${errBody.detail ?? errBody.error ?? 'unknown'}`,
        };
      }

      const data = (await res.json()) as PythonGenerationResult;
      return {
        imageBase64: data.imageBase64 ?? null,
        imageUrl: data.imageUrl ?? null,
        seed: data.seed ?? input.seed ?? 42,
        model: data.model ?? input.model,
        timingsMs: data.timingsMs ?? { modelLoadMs: 0, inferenceMs: 0, totalMs: 0 },
        vramPeakGb: data.vramPeakGb ?? null,
        ramPeakGb: data.ramPeakGb ?? null,
        promptUsed: data.promptUsed ?? input.prompt,
        optimizedPrompt: data.optimizedPrompt,
        error: data.error,
      };
    } catch (err) {
      clearTimeout(timeoutId);
      const message = err instanceof Error ? err.message : String(err);
      return {
        imageBase64: null,
        imageUrl: null,
        seed: input.seed ?? 42,
        model: input.model,
        timingsMs: { modelLoadMs: 0, inferenceMs: 0, totalMs: 0 },
        vramPeakGb: null,
        ramPeakGb: null,
        promptUsed: input.prompt,
        error: `Python service request failed: ${message}`,
      };
    }
  }

  async parsePrompt(prompt: string): Promise<any> {
    const healthy = await this.checkHealth();
    if (!healthy) {
      return { error: 'Python inference service unavailable', originalPrompt: prompt, optimizedPrompt: prompt };
    }

    try {
      const res = await fetch(`${this.baseUrl}/v1/prompt/parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, return_json: true, mode: 'faithful' }),
      });
      if (!res.ok) {
        return { error: `Parse failed (${res.status})`, originalPrompt: prompt, optimizedPrompt: prompt };
      }
      return await res.json();
    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err), originalPrompt: prompt, optimizedPrompt: prompt };
    }
  }

  async optimizePrompt(prompt: string, targetModel: string): Promise<string> {
    const result = await this.parsePrompt(prompt);
    return result.optimizedPrompt ?? result.original_prompt ?? prompt;
  }

  getStatus(): PythonHealthStatus {
    return {
      available: this.available,
      models: this.models,
      baseUrl: this.baseUrl,
      gpuAvailable: false,
      loadedModels: this.models,
    };
  }
}

export const pythonDiffusionClient = new PythonDiffusionClient(
  process.env.PYTHON_SERVICE_URL ?? 'http://localhost:8100',
  Number(process.env.PYTHON_SERVICE_TIMEOUT_MS ?? 300000),
);
