import test from 'node:test';
import assert from 'node:assert/strict';
import {
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
  assert.ok(ids.includes('local-sdxs'));
  assert.ok(ids.includes('local-sdxs-cfg'));
  assert.ok(ids.includes('local-sdxs-img2img'));
  assert.equal(ids.length, 3);
});
