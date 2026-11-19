import assert from 'node:assert/strict';
import test from 'node:test';
import {
  VideoGenerationService,
  VideoJob,
} from './videoGeneration';

function isVideoJob(value: Record<string, unknown>): VideoJob {
  return value as unknown as VideoJob;
}

async function waitForJob(
  service: VideoGenerationService,
  id: string
): Promise<VideoJob> {
  const end = Date.now() + 1000;
  while (Date.now() < end) {
    const job = service.getJob(id);
    if (!job) throw new Error('Expected the video job to be available.');
    if (job.status === 'completed' || job.status === 'failed') return job;
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
  throw new Error('Video job did not finish in time.');
}

test('video jobs are not available in local-only mode', async () => {
  const service = new VideoGenerationService('local-not-configured', {
    pollIntervalMs: 1,
  });

  const job = service.createJob({
    prompt: 'A red kite over a field',
    durationSeconds: 6,
    aspectRatio: '16:9',
    resolution: '720p',
    referenceImageBase64: 'data:image/png;base64,aGVsbG8=',
    referenceImageMimeType: 'image/png',
  });

  assert.equal(job.status, 'failed');
  assert.match(job.error ?? '', /not available in local-only mode/);
  assert.equal(service.isAvailable(), false);
});

test('video jobs reject unsupported durations', async () => {
  const service = new VideoGenerationService('local-not-configured', {
    pollIntervalMs: 1,
  });

  assert.throws(
    () =>
      service.createJob({
        prompt: 'A scene',
        durationSeconds: 5,
        aspectRatio: '16:9',
        resolution: '720p',
      }),
    /duration must be 4, 6, or 8/
  );
});

test('video jobs report provider failures when local engine is unavailable', async () => {
  const service = new VideoGenerationService('local-not-configured', {
    pollIntervalMs: 1,
  });

  const job = service.createJob({
    prompt: 'A scene',
    durationSeconds: 4,
    aspectRatio: '16:9',
    resolution: '720p',
  });
  const failed = await waitForJob(service, job.id);
  assert.equal(failed.status, 'failed');
  assert.match(failed.error ?? '', /not available in local-only mode/);
});

test('video service reports not available', () => {
  const service = new VideoGenerationService();
  assert.equal(service.isAvailable(), false);
});