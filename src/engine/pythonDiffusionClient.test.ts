import test from 'node:test';
import assert from 'node:assert/strict';
import { PythonDiffusionClient } from './pythonDiffusionClient';

test('maps studio engine IDs and decodes the inference service snake_case response', async () => {
  const originalFetch = globalThis.fetch;
  const requestedModels: string[] = [];

  globalThis.fetch = async (input, init) => {
    const url = String(input);
    if (url.endsWith('/v1/health')) {
      return Response.json({
        gpu_available: true,
        gpu_count: 1,
        gpus: [{ index: 0, name: 'Test GPU', vram_gb: 48 }],
        loaded_models: ['flux-klein-4b'],
      });
    }
    if (url.endsWith('/v1/models')) {
      return Response.json({ models: [{ model_id: 'flux-klein-4b', enabled: true }] });
    }
    if (url.endsWith('/v1/generate')) {
      const request = JSON.parse(String(init?.body)) as { model: string };
      requestedModels.push(request.model);
      return Response.json({
        image_base64: 'data:image/png;base64,abc',
        seed: 12,
        model: request.model,
        timings: { model_load_ms: 100, inference_ms: 200, total_ms: 300 },
        vram_peak_gb: 8,
        ram_peak_gb: 10,
        prompt_used: 'scene',
      });
    }
    throw new Error(`Unexpected request: ${url}`);
  };

  try {
    const client = new PythonDiffusionClient('http://inference.test');
    const result = await client.generate({
      model: 'python-flux-klein',
      prompt: 'scene',
      seed: 12,
    });

    assert.deepEqual(requestedModels, ['flux-klein-4b']);
    assert.equal(result.imageBase64, 'data:image/png;base64,abc');
    assert.deepEqual(result.timingsMs, {
      modelLoadMs: 100,
      inferenceMs: 200,
      totalMs: 300,
    });
    assert.equal(client.getStatus().gpuAvailable, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('does not silently substitute an unsupported engine', async () => {
  const client = new PythonDiffusionClient('http://inference.test');
  const result = await client.generate({ model: 'not-an-engine', prompt: 'scene' });
  assert.match(result.error ?? '', /Unsupported Python image engine/);
  assert.equal(result.imageBase64, null);
});

test('proxies optional image inspection through the Python service', async () => {
  const originalFetch = globalThis.fetch;
  const requestedUrls: string[] = [];

  globalThis.fetch = async (input) => {
    const url = String(input);
    requestedUrls.push(url);
    if (url.endsWith('/v1/health')) return Response.json({ gpu_available: true });
    if (url.endsWith('/v1/models')) return Response.json({ models: [] });
    if (url.endsWith('/v1/quality/objects')) {
      return Response.json({ detections: { '<OD>': { labels: ['cat'] } } });
    }
    throw new Error(`Unexpected request: ${url}`);
  };

  try {
    const client = new PythonDiffusionClient('http://inference.test');
    const result = await client.detectObjects('data:image/png;base64,abc');
    assert.deepEqual(result.detections, { '<OD>': { labels: ['cat'] } });
    assert.ok(requestedUrls.some((url) => url.endsWith('/v1/quality/objects')));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
