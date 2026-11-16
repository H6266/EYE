import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyClassifierFreeGuidance,
  buildEulerTrailingSchedule,
  eulerStep,
  generateSeededLatents,
  LATENT_SIZE,
  scaleModelInput,
  SDXS_SIGMA_MAX,
} from './diffusionSampler';

test('generateSeededLatents is deterministic per seed and matches expected tensor size', () => {
  const a = generateSeededLatents(4242, SDXS_SIGMA_MAX);
  const b = generateSeededLatents(4242, SDXS_SIGMA_MAX);
  const c = generateSeededLatents(9999, SDXS_SIGMA_MAX);

  assert.equal(a.length, LATENT_SIZE);
  assert.deepEqual(Array.from(a.slice(0, 16)), Array.from(b.slice(0, 16)));
  assert.notDeepEqual(Array.from(a.slice(0, 16)), Array.from(c.slice(0, 16)));
});

test('buildEulerTrailingSchedule produces valid decreasing sigmas ending at 0', () => {
  const oneStep = buildEulerTrailingSchedule(1);
  assert.equal(oneStep.length, 1);
  assert.equal(oneStep[0].timestep, 999);
  assert.ok(Math.abs(oneStep[0].sigma - SDXS_SIGMA_MAX) < 1e-4);
  assert.equal(oneStep[0].nextSigma, 0);

  const fourStep = buildEulerTrailingSchedule(4);
  assert.equal(fourStep.length, 4);
  assert.ok(fourStep[0].sigma > fourStep[1].sigma);
  assert.ok(fourStep[1].sigma > fourStep[2].sigma);
  assert.ok(fourStep[2].sigma > fourStep[3].sigma);
  assert.equal(fourStep[3].nextSigma, 0);
});

test('scaleModelInput, CFG, and eulerStep compute accurate updates', () => {
  const x = new Float32Array([10, -10]);
  const scaled = scaleModelInput(x, Math.sqrt(3)); // factor = 1/2
  assert.ok(Math.abs(scaled[0] - 5) < 1e-5);
  assert.ok(Math.abs(scaled[1] + 5) < 1e-5);

  const cond = new Float32Array([2, 4]);
  const uncond = new Float32Array([1, 2]);
  const guided = applyClassifierFreeGuidance(cond, uncond, 3.0);
  // 1 + 3*(2-1) = 4, 2 + 3*(4-2) = 8
  assert.ok(Math.abs(guided[0] - 4) < 1e-5);
  assert.ok(Math.abs(guided[1] - 8) < 1e-5);

  const next = eulerStep(new Float32Array([14.6, 0]), new Float32Array([1, -0.5]), 14.6, 0);
  assert.ok(Math.abs(next[0] - 0) < 1e-4);
  assert.ok(Math.abs(next[1] - 7.3) < 1e-4);
});
