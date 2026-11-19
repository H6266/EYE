/**
 * Local video generation service.
 *
 * EYE is a local-first platform. Video generation is not yet supported
 * locally. This service provides an honest, informative response when
 * video generation is requested — it does NOT silently fall back to
 * any external API, and it does NOT simulate successful generation.
 *
 * When a local video engine becomes available, this file should be
 * extended to implement actual local video generation.
 */

export interface VideoGenerationRequest {
  prompt: string;
  durationSeconds: number;
  aspectRatio: '16:9' | '9:16';
  resolution: '720p' | '1080p';
  referenceImageBase64?: string;
  referenceImageMimeType?: string;
}

export type VideoJobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface VideoJob {
  id: string;
  status: VideoJobStatus;
  progress: number | null;
  createdAt: string;
  updatedAt: string;
  durationSeconds: number;
  model: string;
  resultUrl?: string;
  error?: string;
}

interface VideoResult {
  bytes: Buffer;
  mimeType: string;
  expiresAt: number;
}

const JOB_TTL_MS = 60 * 60 * 1000;
const RESULT_TTL_MS = 15 * 60 * 1000;
const RESULT_MAX_BYTES = 64 * 1024 * 1024;
const RESULT_CACHE_MAX_BYTES = 128 * 1024 * 1024;
const MAX_ACTIVE_JOBS = 2;

export class VideoGenerationService {
  private readonly jobs = new Map<string, VideoJob>();
  private readonly results = new Map<string, VideoResult>();
  private readonly pollIntervalMs: number;
  private readonly timeoutMs: number;
  private activeJobs = 0;

  constructor(
    private readonly model: string = 'local-not-configured',
    options: { pollIntervalMs?: number; timeoutMs?: number } = {}
  ) {
    this.pollIntervalMs = options.pollIntervalMs ?? 10_000;
    this.timeoutMs = options.timeoutMs ?? 10 * 60 * 1000;
  }

  createJob(request: VideoGenerationRequest): VideoJob {
    this.cleanup();
    if (!request.prompt.trim()) throw new Error('A video motion prompt is required.');
    if (![4, 6, 8].includes(request.durationSeconds)) {
      throw new Error('Video duration must be 4, 6, or 8 seconds.');
    }
    if (this.activeJobs >= MAX_ACTIVE_JOBS) {
      throw new Error('Video generation queue is full. Try again shortly.');
    }

    const now = new Date().toISOString();
    const job: VideoJob = {
      id: crypto.randomUUID(),
      status: 'failed',
      progress: null,
      createdAt: now,
      updatedAt: now,
      durationSeconds: request.durationSeconds,
      model: this.model,
      error: 'Video generation is not available in local-only mode. A local video engine is not yet implemented.',
    };
    this.jobs.set(job.id, job);
    return { ...job };
  }

  getJob(id: string): VideoJob | undefined {
    this.cleanup();
    const job = this.jobs.get(id);
    return job ? { ...job } : undefined;
  }

  getResult(id: string): { bytes: Buffer; mimeType: string } | undefined {
    this.cleanup();
    const result = this.results.get(id);
    return result
      ? { bytes: Buffer.from(result.bytes), mimeType: result.mimeType }
      : undefined;
  }

  isAvailable(): boolean {
    return false;
  }

  private cleanup(): void {
    const cutoff = Date.now() - JOB_TTL_MS;
    for (const [id, job] of this.jobs) {
      if (
        Date.parse(job.createdAt) < cutoff &&
        job.status !== 'queued' &&
        job.status !== 'processing'
      ) {
        this.jobs.delete(id);
        this.results.delete(id);
      }
    }
    for (const [id, result] of this.results) {
      if (result.expiresAt <= Date.now()) this.results.delete(id);
    }
  }
}