import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AiGatewayError,
  AiInferenceGateway,
  AiJob,
} from './aiInferenceGateway';

async function waitForJob(gateway: AiInferenceGateway, id: string): Promise<AiJob> {
  const end = Date.now() + 3000;
  while (Date.now() < end) {
    const job = gateway.getJob(id);
    if (!job) throw new Error('Expected the job to remain available.');
    if (job.status === 'completed' || job.status === 'failed') return job;
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
  throw new Error('Inference job did not finish in time.');
}

test('mock generation creates a job and returns an image without retaining reference data', async () => {
  const gateway = new AiInferenceGateway({ mockDelayMs: 1 });
  const job = await gateway.createJob({
    task: 'image.generate',
    modelId: 'z-image-turbo',
    provider: 'mock',
    prompt: 'A lantern in a forest',
    parameters: { seed: 7 },
    referenceImage: {
      data: `data:image/png;base64,${Buffer.from('test image').toString('base64')}`,
    },
  });

  assert.ok(['queued', 'processing'].includes(job.status));
  assert.equal(job.provider, 'mock');
  assert.equal(JSON.stringify(job).includes('test image'), false);

  const completed = await waitForJob(gateway, job.id);
  assert.equal(completed.status, 'completed');
  assert.equal(completed.request.prompt, 'A lantern in a forest');
  assert.equal(completed.resultMetadata?.mock, true);
  assert.equal(completed.resultMetadata?.referenceImageReceived, true);
  assert.equal(completed.resultUrl, `/api/ai/jobs/${job.id}/result`);
  const image = gateway.getResult(job.id);
  assert.equal(image?.contentType, 'image/svg+xml');
  assert.ok(Buffer.from(image?.bytes ?? []).toString().includes('Mock inference'));
});

test('mock analyze and segment tasks return JSON results', async () => {
  const gateway = new AiInferenceGateway({ mockDelayMs: 1 });
  const analyze = await gateway.createJob({
    task: 'image.analyze',
    modelId: 'florence-2-base-ft',
    provider: 'mock',
    instructions: 'Describe the image',
    referenceImage: { url: 'https://assets.example/image.png' },
  });
  const segment = await gateway.createJob({
    task: 'image.segment',
    modelId: 'sam2-hiera-base-plus',
    provider: 'mock',
    referenceImage: { url: 'https://assets.example/image.png' },
  });

  const completedAnalyze = await waitForJob(gateway, analyze.id);
  const completedSegment = await waitForJob(gateway, segment.id);
  assert.deepEqual(
    JSON.parse(Buffer.from(gateway.getResult(completedAnalyze.id)!.bytes).toString()).output,
    {
    caption: 'Mock image analysis result.',
    }
  );
  assert.deepEqual(
    JSON.parse(Buffer.from(gateway.getResult(completedSegment.id)!.bytes).toString()).output,
    {
      masks: [],
      note: 'Mock segmentation returns no real masks.',
    }
  );
});

test('validates task, model, input, and reference image before enqueueing', async () => {
  const gateway = new AiInferenceGateway();
  await assert.rejects(
    gateway.createJob({
      task: 'image.analyze',
      modelId: 'z-image-turbo',
      provider: 'mock',
      prompt: 'wrong task',
    }),
    /supports 'image.generate'/
  );
  await assert.rejects(
    gateway.createJob({
      task: 'image.edit',
      modelId: 'flux-2-klein-4b-edit',
      provider: 'mock',
      prompt: 'edit this',
    }),
    /reference image is required/
  );
  await assert.rejects(
    gateway.createJob({
      task: 'image.generate',
      modelId: 'z-image-turbo',
      provider: 'mock',
      prompt: 'bad URL',
      referenceImage: { url: 'http://example.test/image.png' },
    }),
    /must use HTTPS/
  );
});

test('reports missing tokens and rejects invalid dedicated endpoint URLs', async () => {
  const noToken = new AiInferenceGateway({
    env: { AI_INFERENCE_PROVIDER: 'hf-inference' },
  });
  await assert.rejects(
    noToken.createJob({
      task: 'image.analyze',
      modelId: 'florence-2-base-ft',
      instructions: 'Describe',
      referenceImage: { data: 'aGVsbG8=' },
    }),
    (error: unknown) =>
      error instanceof AiGatewayError &&
      error.statusCode === 503 &&
      error.message.includes('HF_TOKEN')
  );

  const invalidEndpoint = new AiInferenceGateway({
    env: {
      AI_INFERENCE_PROVIDER: 'hf-endpoint',
      HF_TOKEN: 'test-token',
      HF_GENERATION_ENDPOINT_URL: 'http://example.test/inference',
    },
  });
  await assert.rejects(
    invalidEndpoint.createJob({
      task: 'image.generate',
      modelId: 'z-image-turbo',
      prompt: 'A remote image',
    }),
    /valid HTTPS URL/
  );
});

test('dedicated Hugging Face endpoint forwards normalized inputs and retries transient failures', async () => {
  let attempts = 0;
  let receivedPayload: Record<string, unknown> | undefined;
  const fetcher: typeof fetch = async (_input, init) => {
    attempts += 1;
    receivedPayload = JSON.parse(String(init?.body)) as Record<string, unknown>;
    if (attempts < 3) return Response.json({ error: 'warming up' }, { status: 503 });
    return new Response(new Uint8Array([137, 80, 78, 71]), {
      headers: { 'Content-Type': 'image/png' },
    });
  };
  const gateway = new AiInferenceGateway({
    env: {
      HF_TOKEN: 'server-only-token',
      HF_GENERATION_ENDPOINT_URL: 'https://endpoint.example.test/generate',
      AI_PROVIDER_TIMEOUT_MS: '1000',
    },
    fetcher,
    pollIntervalMs: 1,
  });
  const job = await gateway.createJob({
    task: 'image.generate',
    modelId: 'z-image-turbo',
    provider: 'hf-endpoint',
    prompt: 'A remote lighthouse',
    referenceImage: { data: 'data:image/png;base64,aGVsbG8=' },
    parameters: { width: 512, seed: 42 },
  });

  const completed = await waitForJob(gateway, job.id);
  assert.equal(attempts, 3);
  assert.equal(completed.status, 'completed');
  assert.equal(completed.resultMetadata?.contentType, 'image/png');
  assert.equal(
    (receivedPayload?.inputs as Record<string, unknown>).prompt,
    'A remote lighthouse'
  );
  assert.equal(
    (receivedPayload?.inputs as Record<string, unknown>).image,
    'data:image/png;base64,aGVsbG8='
  );
  assert.equal(JSON.stringify(receivedPayload).includes('server-only-token'), false);
  assert.deepEqual([...gateway.getResult(job.id)!.bytes], [137, 80, 78, 71]);
});

test('Hugging Face Inference Providers use task payloads and snake_case parameters', async () => {
  let requestUrl = '';
  let requestBody: Record<string, unknown> | undefined;
  const fetcher: typeof fetch = async (input, init) => {
    requestUrl = String(input);
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return Response.json({ generated_text: 'A detailed image caption' });
  };
  const gateway = new AiInferenceGateway({
    env: { HF_TOKEN: 'test-token', AI_PROVIDER_TIMEOUT_MS: '1000' },
    fetcher,
  });
  const job = await gateway.createJob({
    task: 'image.analyze',
    modelId: 'florence-2-base-ft',
    provider: 'hf-inference',
    instructions: 'Describe visible objects',
    referenceImage: { data: 'data:image/png;base64,aGVsbG8=' },
    parameters: { taskPrompt: '<MORE_DETAILED_CAPTION>' },
  });
  const completed = await waitForJob(gateway, job.id);
  assert.equal(completed.status, 'completed');
  assert.equal(
    requestUrl,
    'https://router.huggingface.co/hf-inference/models/microsoft/Florence-2-base-ft'
  );
  assert.deepEqual(requestBody?.inputs, {
    image: 'data:image/png;base64,aGVsbG8=',
    text: 'Describe visible objects',
  });
  assert.deepEqual(
    JSON.parse(Buffer.from(gateway.getResult(job.id)!.bytes).toString()),
    { generated_text: 'A detailed image caption' }
  );
});

test('remote provider errors become failed jobs with explicit messages', async () => {
  const fetcher: typeof fetch = async () =>
    Response.json({ error: 'endpoint is unavailable' }, { status: 502 });
  const gateway = new AiInferenceGateway({
    env: {
      HF_TOKEN: 'test-token',
      HF_GENERATION_ENDPOINT_URL: 'https://endpoint.example.test/generate',
      AI_PROVIDER_TIMEOUT_MS: '1000',
    },
    fetcher,
  });
  const job = await gateway.createJob({
    task: 'image.generate',
    modelId: 'flux-2-klein-4b',
    provider: 'hf-endpoint',
    prompt: 'A red boat',
  });
  const failed = await waitForJob(gateway, job.id);
  assert.equal(failed.status, 'failed');
  assert.match(failed.error ?? '', /502/);
  assert.match(failed.error ?? '', /unavailable/);
});

test('RunPod adapter polls asynchronous serverless jobs and returns worker images', async () => {
  let polls = 0;
  let runInput: Record<string, unknown> | undefined;
  const fetcher: typeof fetch = async (input, init) => {
    const url = String(input);
    if (url.endsWith('/run')) {
      runInput = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return Response.json({ id: 'run-123', status: 'IN_PROGRESS' });
    }
    if (url.endsWith('/status/run-123')) {
      polls += 1;
      if (polls === 1) return Response.json({ status: 'IN_PROGRESS', progress: 50 });
      return Response.json({
        status: 'COMPLETED',
        output: { image_base64: Buffer.from('mock PNG').toString('base64') },
      });
    }
    throw new Error(`Unexpected RunPod URL: ${url}`);
  };
  const gateway = new AiInferenceGateway({
    env: {
      RUNPOD_API_KEY: 'server-only-key',
      RUNPOD_GENERATION_ENDPOINT_ID: 'endpoint-123',
      AI_PROVIDER_TIMEOUT_MS: '1000',
    },
    fetcher,
    pollIntervalMs: 1,
  });
  const job = await gateway.createJob({
    task: 'image.generate',
    modelId: 'qwen-image-2512-lightning',
    provider: 'runpod',
    prompt: 'A paper moon',
    referenceImage: { url: 'https://assets.example/image.png' },
    parameters: { steps: 4 },
  });
  const completed = await waitForJob(gateway, job.id);
  assert.equal(completed.status, 'completed');
  assert.equal(completed.provider, 'runpod');
  assert.equal(polls, 2);
  assert.equal(
    ((runInput?.input as Record<string, unknown>).reference_image as Record<string, unknown>)
      .url,
    'https://assets.example/image.png'
  );
  assert.equal(gateway.getResult(job.id)?.contentType, 'image/png');
});

test('optional S3 storage persists binary results and avoids the in-memory cache', async () => {
  let uploaded = false;
  const fetcher: typeof fetch = async (input, init) => {
    uploaded = String(input).includes('/assets/auteur-studio/');
    assert.equal(init?.method, 'PUT');
    assert.match(new Headers(init?.headers).get('authorization') ?? '', /^AWS4-HMAC-SHA256/);
    return new Response(null, { status: 200 });
  };
  const gateway = new AiInferenceGateway({
    env: {
      S3_ENDPOINT_URL: 'https://objects.example.test',
      S3_BUCKET: 'assets',
      S3_REGION: 'test-region-1',
      S3_ACCESS_KEY_ID: 'test-access-key',
      S3_SECRET_ACCESS_KEY: 'test-secret-key',
    },
    fetcher,
    mockDelayMs: 1,
  });
  const job = await gateway.createJob({
    task: 'image.generate',
    modelId: 'z-image-turbo',
    provider: 'mock',
    prompt: 'A generated result',
  });
  const completed = await waitForJob(gateway, job.id);
  assert.equal(completed.status, 'completed');
  assert.equal(uploaded, true);
  assert.match(completed.resultUrl ?? '', /^https:\/\/objects\.example\.test\/assets\//);
  assert.match(completed.resultUrl ?? '', /X-Amz-Signature=/);
  assert.equal(completed.resultMetadata?.persistent, true);
  assert.equal(gateway.getResult(job.id), undefined);
});

test('remote provider timeouts become failed jobs and invalid model IDs are rejected', async () => {
  const fetcher: typeof fetch = async () => {
    throw new DOMException('Request timed out', 'TimeoutError');
  };
  const gateway = new AiInferenceGateway({
    env: {
      HF_TOKEN: 'test-token',
      HF_GENERATION_ENDPOINT_URL: 'https://endpoint.example.test/generate',
      AI_PROVIDER_TIMEOUT_MS: '1000',
    },
    fetcher,
  });
  const job = await gateway.createJob({
    task: 'image.generate',
    modelId: 'z-image-turbo',
    provider: 'hf-endpoint',
    prompt: 'A timed request',
  });
  const failed = await waitForJob(gateway, job.id);
  assert.equal(failed.status, 'failed');
  assert.match(failed.error ?? '', /timed out/);
  await assert.rejects(
    gateway.createJob({
      task: 'image.generate',
      modelId: 'unknown',
      provider: 'mock',
      prompt: 'not registered',
    }),
    /Unknown AI model/
  );
});
