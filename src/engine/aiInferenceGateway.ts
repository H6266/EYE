import {
  AiModelDefinition,
  AiProvider,
  AiTask,
  AI_MODEL_REGISTRY,
  getAiModel,
} from './aiModelRegistry';
import { S3ImageStore } from './s3ImageStore';

const IMAGE_MAX_BYTES = 12 * 1024 * 1024;
const RESULT_MAX_BYTES = 24 * 1024 * 1024;
const RESULT_CACHE_MAX_BYTES = 96 * 1024 * 1024;
const JOB_TTL_MS = 60 * 60 * 1000;
const RESULT_TTL_MS = 15 * 60 * 1000;
const MAX_JOBS = 32;

export type AiJobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface AiJobRequest {
  task: AiTask;
  modelId: string;
  provider?: AiProvider;
  prompt?: string;
  instructions?: string;
  parameters?: Record<string, unknown>;
  referenceImage?: {
    url?: string;
    data?: string;
    mimeType?: string;
  };
  projectId?: string;
}

export interface AiJob {
  id: string;
  task: AiTask;
  modelId: string;
  repository: string;
  provider: AiProvider;
  status: AiJobStatus;
  progress: number | null;
  createdAt: string;
  updatedAt: string;
  projectId?: string;
  request: {
    prompt?: string;
    instructions?: string;
    parameters: Record<string, unknown>;
  };
  resultUrl?: string;
  resultMetadata?: Record<string, unknown>;
  error?: string;
}

export interface AiInferenceResult {
  image?: {
    bytes: Uint8Array;
    contentType: string;
  };
  url?: string;
  data?: Record<string, unknown>;
}

export interface AiModelStatus extends AiModelDefinition {
  status: 'configured' | 'unconfigured' | 'unsupported';
  missingConfiguration: string[];
}

export class AiGatewayError extends Error {
  constructor(
    message: string,
    readonly statusCode: number = 400
  ) {
    super(message);
    this.name = 'AiGatewayError';
  }
}

interface GatewayOptions {
  env?: NodeJS.ProcessEnv;
  fetcher?: typeof fetch;
  pollIntervalMs?: number;
  mockDelayMs?: number;
  concurrency?: number;
}

interface QueuedJob {
  id: string;
  request: AiJobRequest;
  model: AiModelDefinition;
  provider: AiProvider;
}

interface StoredResult {
  bytes: Uint8Array;
  contentType: string;
  expiresAt: number;
}

export class AiInferenceGateway {
  private readonly env: NodeJS.ProcessEnv;
  private readonly fetcher: typeof fetch;
  private readonly pollIntervalMs: number;
  private readonly mockDelayMs: number;
  private readonly concurrency: number;
  private readonly s3ImageStore: S3ImageStore;
  private readonly jobs = new Map<string, AiJob>();
  private readonly results = new Map<string, StoredResult>();
  private readonly queue: QueuedJob[] = [];
  private activeJobs = 0;

  constructor(options: GatewayOptions = {}) {
    this.env = options.env ?? process.env;
    this.fetcher = options.fetcher ?? fetch;
    this.pollIntervalMs = options.pollIntervalMs ?? 1000;
    this.mockDelayMs = options.mockDelayMs ?? 10;
    this.concurrency = options.concurrency ?? 2;
    this.s3ImageStore = new S3ImageStore({
      env: this.env,
      fetcher: this.fetcher,
    });
  }

  listModels(provider?: AiProvider): AiModelStatus[] {
    return AI_MODEL_REGISTRY.map((model) => {
      const selectedProvider = provider ?? this.defaultProvider();
      const missingConfiguration = this.getMissingConfiguration(model, selectedProvider);
      return {
        ...model,
        status: !model.providers.includes(selectedProvider)
          ? 'unsupported'
          : missingConfiguration.length
            ? 'unconfigured'
            : 'configured',
        missingConfiguration,
      };
    });
  }

  async createJob(request: AiJobRequest): Promise<AiJob> {
    this.cleanupExpired();
    if (!request || typeof request !== 'object' || Array.isArray(request)) {
      throw new AiGatewayError('An inference job request object is required.');
    }
    const model = getAiModel(request.modelId);
    if (!model) {
      throw new AiGatewayError(`Unknown AI model '${request.modelId}'.`, 404);
    }
    if (model.task !== request.task) {
      throw new AiGatewayError(
        `Model '${model.id}' supports '${model.task}', not '${request.task}'.`
      );
    }
    this.validateRequest(model, request);

    const provider = request.provider ?? this.defaultProvider();
    if (!model.providers.includes(provider)) {
      throw new AiGatewayError(
        `Model '${model.id}' is not configured for provider '${provider}'.`,
        400
      );
    }
    const missingConfiguration = this.getMissingConfiguration(model, provider);
    if (missingConfiguration.length > 0) {
      throw new AiGatewayError(
        `Provider '${provider}' is not configured. Missing or invalid: ${missingConfiguration.join(', ')}.`,
        503
      );
    }
    if (this.jobs.size >= MAX_JOBS) {
      throw new AiGatewayError('The inference job queue is full. Try again shortly.', 429);
    }
    if (this.queue.length + this.activeJobs >= 16) {
      throw new AiGatewayError('The inference job queue is full. Try again shortly.', 429);
    }

    const now = new Date().toISOString();
    const id = crypto.randomUUID();
    const job: AiJob = {
      id,
      task: request.task,
      modelId: model.id,
      repository: model.repository,
      provider,
      status: 'queued',
      progress: 0,
      createdAt: now,
      updatedAt: now,
      projectId: request.projectId,
      request: {
        prompt: request.prompt,
        instructions: request.instructions,
        parameters: request.parameters ?? {},
      },
    };
    this.jobs.set(id, job);
    this.queue.push({ id, request, model, provider });
    this.drainQueue();
    return this.copyJob(job);
  }

  getJob(id: string): AiJob | undefined {
    this.cleanupExpired();
    const job = this.jobs.get(id);
    return job ? this.copyJob(job) : undefined;
  }

  getResult(id: string): StoredResult | undefined {
    this.cleanupExpired();
    const result = this.results.get(id);
    return result ? { ...result, bytes: result.bytes.slice() } : undefined;
  }

  private defaultProvider(): AiProvider {
    const configured = this.env.AI_INFERENCE_PROVIDER;
    if (!configured) return 'mock';
    if (['mock', 'hf-inference', 'hf-endpoint', 'runpod'].includes(configured)) {
      return configured as AiProvider;
    }
    throw new AiGatewayError(
      `Invalid AI_INFERENCE_PROVIDER '${configured}'.`,
      503
    );
  }

  private validateRequest(model: AiModelDefinition, request: AiJobRequest): void {
    if (['image.generate', 'image.edit'].includes(request.task)) {
      if (typeof request.prompt !== 'string' || !request.prompt.trim()) {
        throw new AiGatewayError(`A prompt is required for '${request.task}'.`);
      }
    }
    if (['image.edit', 'image.analyze', 'image.segment'].includes(request.task)) {
      if (!request.referenceImage) {
        throw new AiGatewayError(`A reference image is required for '${request.task}'.`);
      }
    }
    if (request.task === 'image.analyze' && !request.instructions?.trim()) {
      throw new AiGatewayError('Instructions are required for image analysis.');
    }
    if (request.referenceImage) {
      const { data, url } = request.referenceImage;
      if (!data && !url) {
        throw new AiGatewayError('Reference image must include a URL or image data.');
      }
      if (data) {
        const base64 = data.replace(/^data:image\/[a-zA-Z0-9.+-]+;base64,/, '');
        if (!/^[A-Za-z0-9+/]*={0,2}$/.test(base64) || base64.length % 4 !== 0) {
          throw new AiGatewayError('Reference image data must be valid base64.');
        }
        if (Buffer.byteLength(base64, 'base64') > IMAGE_MAX_BYTES) {
          throw new AiGatewayError('Reference images must be 12 MB or smaller.', 413);
        }
      }
      if (url && !url.startsWith('data:image/')) {
        let parsed: URL;
        try {
          parsed = new URL(url);
        } catch {
          throw new AiGatewayError('Reference image URL is invalid.');
        }
        if (parsed.protocol !== 'https:') {
          throw new AiGatewayError('Reference image URLs must use HTTPS.');
        }
      }
      const mimeType = request.referenceImage.mimeType;
      if (mimeType && !mimeType.startsWith('image/')) {
        throw new AiGatewayError('Reference image MIME type must be an image type.');
      }
    }
    if (request.parameters && typeof request.parameters !== 'object') {
      throw new AiGatewayError('Generation parameters must be an object.');
    }
    for (const key of Object.keys(request.parameters ?? {})) {
      if (!model.supportedParameters.includes(key)) {
        throw new AiGatewayError(
          `Parameter '${key}' is not supported by model '${model.id}'.`
        );
      }
    }
  }

  private getMissingConfiguration(
    model: AiModelDefinition,
    provider: AiProvider
  ): string[] {
    if (!model.providers.includes(provider)) return [];
    if (provider === 'mock') return [];
    if (provider === 'hf-inference') {
      return this.env.HF_TOKEN ? [] : ['HF_TOKEN'];
    }
    if (provider === 'hf-endpoint') {
      const endpoint = model.endpointEnv ? this.env[model.endpointEnv] : undefined;
      const missing = this.env.HF_TOKEN ? [] : ['HF_TOKEN'];
      if (!endpoint) {
        missing.push(model.endpointEnv ?? 'HF endpoint URL');
      } else if (!this.isValidEndpoint(endpoint)) {
        missing.push(`${model.endpointEnv} (must be a valid HTTPS URL)`);
      }
      return missing;
    }
    if (provider === 'runpod') {
      const endpointEnv = this.runpodEndpointEnv(model.task);
      const endpointId = this.env[endpointEnv];
      const missing = this.env.RUNPOD_API_KEY ? [] : ['RUNPOD_API_KEY'];
      if (!endpointId) missing.push(endpointEnv);
      else if (!/^[A-Za-z0-9_-]+$/.test(endpointId)) {
        missing.push(`${endpointEnv} (invalid endpoint ID)`);
      }
      return missing;
    }
    return ['provider configuration'];
  }

  private isValidEndpoint(value: string): boolean {
    try {
      const endpoint = new URL(value);
      return endpoint.protocol === 'https:' && !endpoint.username && !endpoint.password;
    } catch {
      return false;
    }
  }

  private runpodEndpointEnv(task: AiTask): string {
    if (task === 'image.analyze') return 'RUNPOD_FLORENCE_ENDPOINT_ID';
    if (task === 'image.segment') return 'RUNPOD_SAM2_ENDPOINT_ID';
    return 'RUNPOD_GENERATION_ENDPOINT_ID';
  }

  private drainQueue(): void {
    while (this.activeJobs < this.concurrency && this.queue.length > 0) {
      const queued = this.queue.shift();
      if (!queued) return;
      this.activeJobs += 1;
      void this.runJob(queued).finally(() => {
        this.activeJobs -= 1;
        this.drainQueue();
      });
    }
  }

  private async runJob(queued: QueuedJob): Promise<void> {
    const job = this.jobs.get(queued.id);
    if (!job) return;
    this.updateJob(job, { status: 'processing', progress: 5 });
    try {
      const result = await this.invokeProvider(
        queued.provider,
        queued.model,
        queued.request,
        (progress) => this.updateJob(job, { progress })
      );
      if (result.image) {
        if (result.image.bytes.byteLength > RESULT_MAX_BYTES) {
          throw new Error('Generated image exceeds the 24 MB result limit.');
        }
        const persistentUrl = await this.s3ImageStore.putResult(
          job.id,
          result.image.contentType,
          result.image.bytes
        );
        if (persistentUrl) {
          job.resultUrl = persistentUrl;
        } else {
          this.storeResult(job.id, result.image.bytes, result.image.contentType);
          job.resultUrl = `/api/ai/jobs/${job.id}/result`;
        }
        job.resultMetadata = {
          contentType: result.image.contentType,
          sizeBytes: result.image.bytes.byteLength,
          persistent: Boolean(persistentUrl),
          ...result.data,
        };
      } else if (result.url) {
        job.resultUrl = result.url;
        job.resultMetadata = { ...result.data, remote: true };
      } else {
        const bytes = Buffer.from(JSON.stringify(result.data ?? {}));
        if (bytes.byteLength > RESULT_MAX_BYTES) {
          throw new Error('Inference JSON result exceeds the 24 MB result limit.');
        }
        const persistentUrl = await this.s3ImageStore.putResult(
          job.id,
          'application/json',
          bytes
        );
        if (persistentUrl) {
          job.resultUrl = persistentUrl;
        } else {
          this.storeResult(job.id, bytes, 'application/json');
          job.resultUrl = `/api/ai/jobs/${job.id}/result`;
        }
        job.resultMetadata = {
          contentType: 'application/json',
          sizeBytes: bytes.byteLength,
          resultType: 'json',
          persistent: Boolean(persistentUrl),
        };
      }
      this.updateJob(job, { status: 'completed', progress: 100 });
    } catch (error) {
      this.updateJob(job, {
        status: 'failed',
        progress: null,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async invokeProvider(
    provider: AiProvider,
    model: AiModelDefinition,
    request: AiJobRequest,
    onProgress: (progress: number) => void
  ): Promise<AiInferenceResult> {
    if (provider === 'mock') return this.invokeMock(model, request, onProgress);
    if (provider === 'hf-inference') return this.invokeHfInference(model, request, onProgress);
    if (provider === 'hf-endpoint') return this.invokeHfEndpoint(model, request, onProgress);
    return this.invokeRunpod(model, request, onProgress);
  }

  private async invokeMock(
    model: AiModelDefinition,
    request: AiJobRequest,
    onProgress: (progress: number) => void
  ): Promise<AiInferenceResult> {
    onProgress(25);
    await new Promise((resolve) => setTimeout(resolve, this.mockDelayMs));
    onProgress(75);
    if (model.task === 'image.generate' || model.task === 'image.edit') {
      const prompt = escapeXml(request.prompt ?? request.instructions ?? 'Mock generated image');
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><defs><linearGradient id="g" x2="1" y2="1"><stop stop-color="#161b2e"/><stop offset="1" stop-color="#d89f3d"/></linearGradient></defs><rect width="512" height="512" fill="url(#g)"/><text x="256" y="235" text-anchor="middle" fill="white" font-family="sans-serif" font-size="24">Mock inference</text><text x="256" y="278" text-anchor="middle" fill="white" font-family="sans-serif" font-size="14">${prompt.slice(0, 120)}</text></svg>`;
      return {
        image: { bytes: Buffer.from(svg), contentType: 'image/svg+xml' },
        data: {
          mock: true,
          modelId: model.id,
          prompt: request.prompt,
          referenceImageReceived: Boolean(request.referenceImage),
          parameters: request.parameters ?? {},
        },
      };
    }
    return {
      data: {
        mock: true,
        modelId: model.id,
        task: model.task,
        instructions: request.instructions,
        referenceImageReceived: Boolean(request.referenceImage),
        output:
          model.task === 'image.segment'
            ? { masks: [], note: 'Mock segmentation returns no real masks.' }
            : { caption: 'Mock image analysis result.' },
      },
    };
  }

  private async invokeHfInference(
    model: AiModelDefinition,
    request: AiJobRequest,
    onProgress: (progress: number) => void
  ): Promise<AiInferenceResult> {
    const url = `https://router.huggingface.co/hf-inference/models/${model.repository}`;
    const imageInput = request.referenceImage?.data ?? request.referenceImage?.url;
    const inputs =
      model.task === 'image.analyze'
        ? { image: imageInput, text: request.instructions }
        : model.task === 'image.segment'
          ? imageInput
          : request.prompt ?? '';
    const body: Record<string, unknown> = { inputs };
    if (request.parameters && Object.keys(request.parameters).length > 0) {
      body.parameters = {
        ...request.parameters,
        ...(request.parameters.steps !== undefined
          ? { num_inference_steps: request.parameters.steps }
          : {}),
        ...(request.parameters.guidanceScale !== undefined
          ? { guidance_scale: request.parameters.guidanceScale }
          : {}),
      };
      delete (body.parameters as Record<string, unknown>).steps;
      delete (body.parameters as Record<string, unknown>).guidanceScale;
    }
    onProgress(20);
    const response = await this.fetchWithRetry(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.env.HF_TOKEN}`,
        'Content-Type': 'application/json',
        Accept: 'application/json, image/*',
      },
      body: JSON.stringify(body),
    });
    onProgress(80);
    return this.parseProviderResponse(response);
  }

  private async invokeHfEndpoint(
    model: AiModelDefinition,
    request: AiJobRequest,
    onProgress: (progress: number) => void
  ): Promise<AiInferenceResult> {
    const url = this.env[model.endpointEnv ?? ''];
    if (!url) throw new Error(`Missing ${model.endpointEnv}.`);
    onProgress(20);
    const response = await this.fetchWithRetry(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.env.HF_TOKEN}`,
        'Content-Type': 'application/json',
        Accept: 'application/json, image/*',
      },
      body: JSON.stringify({
        task: model.task,
        model_id: model.id,
        repository: model.repository,
        inputs: {
          prompt: request.prompt,
          instructions: request.instructions,
          image_url: request.referenceImage?.url,
          image: request.referenceImage?.data,
          mime_type: request.referenceImage?.mimeType,
        },
        parameters: request.parameters ?? {},
      }),
    });
    onProgress(80);
    return this.parseProviderResponse(response);
  }

  private async invokeRunpod(
    model: AiModelDefinition,
    request: AiJobRequest,
    onProgress: (progress: number) => void
  ): Promise<AiInferenceResult> {
    const endpointId = this.env[this.runpodEndpointEnv(model.task)];
    if (!endpointId) throw new Error(`Missing ${this.runpodEndpointEnv(model.task)}.`);
    const baseUrl = `https://api.runpod.ai/v2/${endpointId}`;
    const timeoutMs = this.timeoutMs();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const start = await this.fetcher(`${baseUrl}/run`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.env.RUNPOD_API_KEY}`,
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          input: {
            task: model.task,
            model_id: model.id,
            repository: model.repository,
            prompt: request.prompt,
            instructions: request.instructions,
            reference_image: request.referenceImage,
            parameters: request.parameters ?? {},
            project_id: request.projectId,
          },
        }),
        signal: controller.signal,
      });
      const initial = await this.readJsonResponse(start);
      if (!start.ok) throw providerError('RunPod', start.status, initial);
      if (initial.status === 'COMPLETED') {
        onProgress(90);
        return normalizeProviderOutput(initial.output);
      }
      const runId = typeof initial.id === 'string' ? initial.id : undefined;
      if (!runId) throw new Error('RunPod returned no job ID.');
      onProgress(25);
      const expiresAt = Date.now() + timeoutMs;

      while (Date.now() < expiresAt) {
        await delay(this.pollIntervalMs, controller.signal);
        const statusResponse = await this.fetcher(`${baseUrl}/status/${encodeURIComponent(runId)}`, {
          headers: {
            Authorization: `Bearer ${this.env.RUNPOD_API_KEY}`,
            Accept: 'application/json',
          },
          signal: controller.signal,
        });
        const status = await this.readJsonResponse(statusResponse);
        if (!statusResponse.ok) {
          throw providerError('RunPod status', statusResponse.status, status);
        }
        if (status.status === 'COMPLETED') {
          onProgress(90);
          return normalizeProviderOutput(status.output);
        }
        if (status.status === 'FAILED' || status.status === 'CANCELLED' || status.status === 'TIMED_OUT') {
          throw new Error(
            `RunPod job ${status.status.toLowerCase()}: ${String(status.error ?? 'no provider details')}`
          );
        }
        onProgress(Math.min(85, 25 + Number(status.progress ?? 0) * 0.6));
      }
      throw new Error(`RunPod inference timed out after ${timeoutMs} ms.`);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`RunPod inference timed out after ${timeoutMs} ms.`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  private async fetchWithRetry(url: string, init: RequestInit): Promise<Response> {
    const timeoutMs = this.timeoutMs();
    const expiresAt = Date.now() + timeoutMs;
    let lastError: unknown;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const remainingMs = expiresAt - Date.now();
      if (remainingMs <= 0) {
        throw new Error(`Inference provider request timed out after ${timeoutMs} ms.`);
      }
      try {
        const response = await this.fetcher(url, {
          ...init,
          signal: AbortSignal.timeout(remainingMs),
        });
        if (response.ok || (response.status < 500 && response.status !== 429)) {
          return response;
        }
        if (attempt === 2) return response;
        await delay(Math.min(250 * 2 ** attempt, Math.max(0, expiresAt - Date.now())));
      } catch (error) {
        lastError = error;
        if (error instanceof Error && error.name === 'TimeoutError') break;
        if (attempt === 2) break;
        await delay(Math.min(250 * 2 ** attempt, Math.max(0, expiresAt - Date.now())));
      }
    }
    if (Date.now() >= expiresAt || (lastError instanceof Error && lastError.name === 'TimeoutError')) {
      throw new Error(`Inference provider request timed out after ${timeoutMs} ms.`);
    }
    throw new Error(
      `Inference provider request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`
    );
  }

  private timeoutMs(): number {
    const configured = Number(this.env.AI_PROVIDER_TIMEOUT_MS);
    return Number.isFinite(configured) && configured >= 1000 ? configured : 600_000;
  }

  private async parseProviderResponse(response: Response): Promise<AiInferenceResult> {
    const contentType = response.headers.get('content-type')?.split(';')[0].trim() ?? '';
    if (!response.ok) {
      const body = await response.text();
      throw providerError(response.url || 'Inference provider', response.status, body.slice(0, 2000));
    }
    if (contentType.startsWith('image/')) {
      return {
        image: { bytes: new Uint8Array(await response.arrayBuffer()), contentType },
      };
    }
    return normalizeProviderOutput(await this.readJsonResponse(response));
  }

  private async readJsonResponse(response: Response): Promise<Record<string, unknown>> {
    const text = await response.text();
    if (!text) throw new Error('Inference provider returned an empty response.');
    try {
      const value: unknown = JSON.parse(text);
      if (!value || typeof value !== 'object') {
        throw new Error('Inference provider returned an invalid JSON object.');
      }
      return value as Record<string, unknown>;
    } catch (error) {
      throw new Error(
        `Inference provider returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private updateJob(job: AiJob, update: Partial<AiJob>): void {
    Object.assign(job, update, { updatedAt: new Date().toISOString() });
  }

  private storeResult(id: string, bytes: Uint8Array, contentType: string): void {
    this.cleanupExpired();
    let cachedBytes = [...this.results.values()].reduce(
      (total, result) => total + result.bytes.byteLength,
      0
    );
    while (
      this.results.size > 0 &&
      cachedBytes + bytes.byteLength > RESULT_CACHE_MAX_BYTES
    ) {
      const oldest = this.results.entries().next().value as
        | [string, StoredResult]
        | undefined;
      if (!oldest) break;
      this.results.delete(oldest[0]);
      cachedBytes -= oldest[1].bytes.byteLength;
    }
    this.results.set(id, {
      bytes: bytes.slice(),
      contentType,
      expiresAt: Date.now() + RESULT_TTL_MS,
    });
  }

  private copyJob(job: AiJob): AiJob {
    return {
      ...job,
      request: {
        ...job.request,
        parameters: { ...job.request.parameters },
      },
      resultMetadata: job.resultMetadata ? { ...job.resultMetadata } : undefined,
    };
  }

  private cleanupExpired(): void {
    const cutoff = Date.now() - JOB_TTL_MS;
    for (const [id, job] of this.jobs) {
      if (Date.parse(job.createdAt) < cutoff && job.status !== 'queued' && job.status !== 'processing') {
        this.jobs.delete(id);
        this.results.delete(id);
      }
    }
    for (const [id, result] of this.results) {
      if (result.expiresAt <= Date.now()) this.results.delete(id);
    }
  }
}

function normalizeProviderOutput(value: unknown): AiInferenceResult {
  if (typeof value === 'string') {
    if (value.startsWith('https://')) return { url: value };
    if (value.startsWith('data:image/')) {
      return decodeImageData(value);
    }
    return { data: { output: value } };
  }
  if (Array.isArray(value)) {
    if (value.length === 1) return normalizeProviderOutput(value[0]);
    return { data: { output: value } };
  }
  if (!value || typeof value !== 'object') {
    throw new Error('Inference provider returned no usable output.');
  }
  const record = value as Record<string, unknown>;
  const directUrl = record.image_url ?? record.result_url ?? record.url;
  if (typeof directUrl === 'string' && /^https:\/\//.test(directUrl)) {
    return { url: directUrl, data: withoutImageFields(record) };
  }
  const base64 =
    record.image_base64 ??
    record.generated_image ??
    record.image ??
    record.output_image ??
    record.png_base64;
  if (typeof base64 === 'string') {
    const mimeType =
      typeof record.content_type === 'string' && record.content_type.startsWith('image/')
        ? record.content_type
        : 'image/png';
    const dataUri = base64.startsWith('data:image/')
      ? base64
      : `data:${mimeType};base64,${base64}`;
    return { ...decodeImageData(dataUri), data: withoutImageFields(record) };
  }
  if (record.output && typeof record.output === 'object') {
    return normalizeProviderOutput(record.output);
  }
  if (record.result && typeof record.result === 'object') {
    return normalizeProviderOutput(record.result);
  }
  if (Array.isArray(record.images) && record.images.length > 0) {
    const first = normalizeProviderOutput(record.images[0]);
    return {
      ...first,
      data: { ...first.data, imageCount: record.images.length },
    };
  }
  return { data: record };
}

function decodeImageData(dataUri: string): AiInferenceResult {
  const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/]+=*)$/.exec(dataUri);
  if (!match) throw new Error('Inference provider returned malformed image data.');
  return {
    image: {
      contentType: match[1],
      bytes: Buffer.from(match[2], 'base64'),
    },
  };
}

function withoutImageFields(value: Record<string, unknown>): Record<string, unknown> {
  const { image, image_base64, generated_image, output_image, png_base64, ...rest } = value;
  void image;
  void image_base64;
  void generated_image;
  void output_image;
  void png_base64;
  return rest;
}

function providerError(provider: string, status: number, body: unknown): Error {
  const detail =
    typeof body === 'string'
      ? body
      : JSON.stringify(body);
  return new Error(`${provider} request failed (${status}): ${detail.slice(0, 2000)}`);
}

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function delay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const timer = setTimeout(resolve, milliseconds);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true }
    );
  });
}

