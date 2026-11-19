import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildVideoPrompt,
  buildRichVisualPrompt,
  sanitizeNegativePrompt,
} from './imagePromptBuilder';
import { getDimensionsForAspect, IMAGE_ENGINES } from '../data/imageEngines';

test('buildRichVisualPrompt integrates character traits, camera framing, lighting, and style', () => {
  const out = buildRichVisualPrompt({
    prompt: 'Standing on a cobblestone pier',
    style: 'cartoon2d',
    cameraFraming: 'low-hero',
    lighting: 'golden',
    characterName: 'Captain Fox',
    characterTraits: 'red fox with brass goggles',
  });

  assert.match(out, /Standing on a cobblestone pier/);
  assert.match(out, /Captain Fox: red fox with brass goggles/);
  assert.match(out, /low-angle hero framing/i);
  assert.match(out, /golden hour/i);
  assert.match(out, /2D animated/i);
});

test('buildVideoPrompt keeps the requested scene distinct from added visual direction', () => {
  const description =
    'A courier rides a red bicycle through a rainy Tokyo street at night.';
  const out = buildVideoPrompt({
    prompt: description,
    style: 'cinematic',
    cameraFraming: 'low-hero',
    lighting: 'golden',
  });

  assert.ok(out.startsWith(`Primary scene description (follow faithfully):\n${description}`));
  assert.match(out, /Additional visual direction:/);
  assert.match(out, /Camera:/);
  assert.match(out, /Lighting:/);
  assert.match(out, /Visual style:/);
});

test('getDimensionsForAspect returns native 512x512 / aspect-framed local diffusion sizes', () => {
  const wide = getDimensionsForAspect('16:9');
  assert.equal(wide.width, 512);
  assert.equal(wide.height, 288);

  const portrait = getDimensionsForAspect('9:16');
  assert.equal(portrait.width, 288);
  assert.equal(portrait.height, 512);

  const square = getDimensionsForAspect('1:1');
  assert.equal(square.width, 512);
  assert.equal(square.height, 512);
});

test('sanitizeNegativePrompt supplies clean defaults when omitted', () => {
  assert.match(sanitizeNegativePrompt(''), /blurry/i);
  assert.equal(sanitizeNegativePrompt('extra fingers, text'), 'extra fingers, text');
});

test('IMAGE_ENGINES configures self-contained local neural diffusion modes without third-party APIs', () => {
  const ids = IMAGE_ENGINES.map((e) => e.id);
  const localIds = ids.filter((id) => id.startsWith('local-sdxs'));
  assert.deepEqual(localIds, ['local-sdxs', 'local-sdxs-cfg', 'local-sdxs-img2img']);
  assert.ok(ids.includes('remote-zimage-turbo'));
  assert.ok(ids.includes('remote-flux-klein'));
  assert.ok(ids.includes('remote-qwen-2512'));
  assert.ok(ids.includes('remote-qwen-2512-lightning'));
});
