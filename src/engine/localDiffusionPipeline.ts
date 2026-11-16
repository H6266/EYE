import fs from 'fs';
import path from 'path';
import * as ort from 'onnxruntime-node';
import { PNG } from 'pngjs';
import { ClipBpeTokenizer } from './clipTokenizer';
import {
  addLatentNoiseForImg2Img,
  applyClassifierFreeGuidance,
  buildEulerTrailingSchedule,
  eulerStep,
  generateSeededLatents,
  LATENT_CHANNELS,
  LATENT_HEIGHT,
  LATENT_WIDTH,
  scaleModelInput,
  VAE_LATENT_SCALE,
} from './diffusionSampler';

const MODEL_DIR = path.resolve(process.cwd(), '.cache/models/sdxs-512');

const WEIGHT_SOURCES: Array<{ relPath: string; url: string }> = [
  {
    relPath: 'tokenizer/vocab.json',
    url: 'https://huggingface.co/skillsafeai/sdxs-512-dreamshaper-onnx-webgpu/resolve/main/tokenizer/vocab.json',
  },
  {
    relPath: 'tokenizer/merges.txt',
    url: 'https://huggingface.co/skillsafeai/sdxs-512-dreamshaper-onnx-webgpu/resolve/main/tokenizer/merges.txt',
  },
  {
    relPath: 'text_encoder.onnx',
    url: 'https://huggingface.co/skillsafeai/sdxs-512-dreamshaper-onnx-webgpu/resolve/main/text_encoder.onnx',
  },
  {
    relPath: 'unet.onnx',
    url: 'https://huggingface.co/skillsafeai/sdxs-512-dreamshaper-onnx-webgpu/resolve/main/unet.onnx',
  },
  {
    relPath: 'vae_decoder.onnx',
    url: 'https://huggingface.co/skillsafeai/sdxs-512-dreamshaper-onnx-webgpu/resolve/main/vae_decoder.onnx',
  },
  {
    relPath: 'vae_encoder.onnx',
    url: 'https://huggingface.co/jdp8/sd-turbo-onnx/resolve/main/vae_encoder/model.onnx',
  },
];

async function ensureWeightFile(relPath: string, url: string): Promise<void> {
  const fullPath = path.join(MODEL_DIR, relPath);
  if (fs.existsSync(fullPath) && fs.statSync(fullPath).size > 1024) {
    return;
  }
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  const res = await fetch(url, { signal: AbortSignal.timeout(180000) });
  if (!res.ok) {
    throw new Error(`Failed to initialize local model weight ${relPath}: HTTP ${res.status}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(fullPath, buf);
}

function decodePngToChw512(pngBuffer: Buffer): Float32Array {
  const decoded = PNG.sync.read(pngBuffer);
  const target = 512;
  const chw = new Float32Array(3 * target * target);
  const stride = target * target;

  // Nearest/bilinear-clean center resize from arbitrary PNG dimensions to 512x512
  for (let y = 0; y < target; y++) {
    const sy = Math.min(decoded.height - 1, Math.floor((y / target) * decoded.height));
    for (let x = 0; x < target; x++) {
      const sx = Math.min(decoded.width - 1, Math.floor((x / target) * decoded.width));
      const srcIdx = (sy * decoded.width + sx) * 4;
      const dstIdx = y * target + x;
      chw[dstIdx] = (decoded.data[srcIdx] / 255) * 2 - 1;
      chw[stride + dstIdx] = (decoded.data[srcIdx + 1] / 255) * 2 - 1;
      chw[2 * stride + dstIdx] = (decoded.data[srcIdx + 2] / 255) * 2 - 1;
    }
  }
  return chw;
}

export interface LocalDiffusionInput {
  prompt: string;
  negativePrompt?: string;
  guidanceScale?: number;
  seed?: number;
  aspectRatio?: '16:9' | '9:16' | '1:1';
  referenceImagePngBase64?: string;
  img2imgStrength?: number;
}

export interface LocalDiffusionResult {
  imageDataUrl: string;
  pngBuffer: Buffer;
  seed: number;
  guidanceUsed: number;
  mode: 'txt2img' | 'img2img';
  width: number;
  height: number;
  tokenStats: {
    activeTokens: number;
    maxTokens: number;
    truncated: boolean;
  };
  timingsMs: {
    tokenizeAndEncode: number;
    vaeEncode: number;
    unetDenoise: number;
    vaeDecode: number;
    total: number;
  };
}

interface LoadedEngine {
  tokenizer: ClipBpeTokenizer;
  textEncoder: ort.InferenceSession;
  unet: ort.InferenceSession;
  vaeDecoder: ort.InferenceSession;
  vaeEncoder: ort.InferenceSession;
}

class LocalDiffusionEngine {
  private enginePromise: Promise<LoadedEngine> | null = null;
  private inferQueue: Promise<void> = Promise.resolve();

  async ensureReady(): Promise<LoadedEngine> {
    if (!this.enginePromise) {
      this.enginePromise = (async () => {
        for (const item of WEIGHT_SOURCES) {
          await ensureWeightFile(item.relPath, item.url);
        }
        const sessionOpts: ort.InferenceSession.SessionOptions = {
          logSeverityLevel: 3,
          intraOpNumThreads: 4,
        };
        const tokenizer = new ClipBpeTokenizer(path.join(MODEL_DIR, 'tokenizer'));
        const [textEncoder, unet, vaeDecoder, vaeEncoder] = await Promise.all([
          ort.InferenceSession.create(path.join(MODEL_DIR, 'text_encoder.onnx'), sessionOpts),
          ort.InferenceSession.create(path.join(MODEL_DIR, 'unet.onnx'), sessionOpts),
          ort.InferenceSession.create(path.join(MODEL_DIR, 'vae_decoder.onnx'), sessionOpts),
          ort.InferenceSession.create(path.join(MODEL_DIR, 'vae_encoder.onnx'), sessionOpts),
        ]);
        return { tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder };
      })();
    }
    return this.enginePromise;
  }

  getStatus(): {
    ready: boolean;
    weightsPresent: boolean;
    ingredients: string[];
  } {
    const weightsPresent = WEIGHT_SOURCES.every((w) => {
      const p = path.join(MODEL_DIR, w.relPath);
      return fs.existsSync(p) && fs.statSync(p).size > 1024;
    });
    return {
      ready: Boolean(this.enginePromise),
      weightsPresent,
      ingredients: [
        'CLIP BPE Tokenizer (vocab + merges)',
        'CLIP ViT-L/14 Text Encoder',
        'SDXS-512-DreamShaper UNet Brain',
        'Euler Sampler + Calibrated CFG Guidance',
        'TAESD VAE Decoder + SD VAE Encoder',
      ],
    };
  }

  async generate(input: LocalDiffusionInput): Promise<LocalDiffusionResult> {
    const runTask = async (): Promise<LocalDiffusionResult> => {
      const tStart = Date.now();
      const { tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder } =
        await this.ensureReady();

      // Calibrated guidance: SDXS distilled UNet excels at 0 (pure distilled) or 1.05..1.35 mild negative repulsion
      const rawGuidance = Number(input.guidanceScale ?? 0);
      const guidanceScale = rawGuidance > 1.02 ? Math.min(1.35, rawGuidance) : 0;
      const useCfg = guidanceScale > 1.02;
      const seed = Math.floor(Number(input.seed) || Math.random() * 1_000_000_000);

      // 1. CLIP BPE Tokenizer + CLIP ViT-L/14 Text Encoder
      const tEnc0 = Date.now();
      const tokenized = tokenizer.encode(input.prompt);
      const condOut = await textEncoder.run({
        input_ids: new ort.Tensor('int64', tokenized.inputIds, [1, 77]),
      });
      const condEmbeds = condOut.last_hidden_state;

      let uncondEmbeds: ort.Tensor | null = null;
      if (useCfg) {
        const negTokenized = tokenizer.encode(input.negativePrompt || '');
        const uncondOut = await textEncoder.run({
          input_ids: new ort.Tensor('int64', negTokenized.inputIds, [1, 77]),
        });
        uncondEmbeds = uncondOut.last_hidden_state;
      }
      const tokenizeAndEncodeMs = Date.now() - tEnc0;

      // 2. Latent Initialization (Pure Gaussian Noise OR VAE-Encoded Img2Img + Noise)
      let vaeEncodeMs = 0;
      let scheduleStep = buildEulerTrailingSchedule(1)[0];
      let latents: Float32Array;
      let mode: 'txt2img' | 'img2img' = 'txt2img';

      if (input.referenceImagePngBase64) {
        const tVaeEnc0 = Date.now();
        try {
          const pngBuf = Buffer.from(input.referenceImagePngBase64, 'base64');
          const chw = decodePngToChw512(pngBuf);
          const encOut = await vaeEncoder.run({
            [vaeEncoder.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, 512, 512]),
          });
          const rawLatent = encOut[vaeEncoder.outputNames[0]].data as Float32Array;
          const scaledClean = new Float32Array(rawLatent.length);
          for (let i = 0; i < rawLatent.length; i++) {
            scaledClean[i] = rawLatent[i] * VAE_LATENT_SCALE;
          }
          const noised = addLatentNoiseForImg2Img(
            scaledClean,
            seed,
            input.img2imgStrength ?? 0.72
          );
          latents = noised.noisyLatent;
          scheduleStep = {
            timestep: noised.timestep,
            sigma: noised.sigma,
            nextSigma: 0,
          };
          mode = 'img2img';
          vaeEncodeMs = Date.now() - tVaeEnc0;
        } catch {
          latents = generateSeededLatents(seed, scheduleStep.sigma);
        }
      } else {
        latents = generateSeededLatents(seed, scheduleStep.sigma);
      }

      // 3. UNet Brain + Euler Sampler + Classifier-Free Guidance
      const tUnet0 = Date.now();
      const scaledSample = scaleModelInput(latents, scheduleStep.sigma);
      const sampleTensor = new ort.Tensor('float32', scaledSample, [
        1,
        LATENT_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
      ]);
      const timestepTensor = new ort.Tensor(
        'float32',
        new Float32Array([scheduleStep.timestep]),
        [1]
      );

      const condUnet = await unet.run({
        sample: sampleTensor,
        timestep: timestepTensor,
        encoder_hidden_states: condEmbeds,
      });
      let noisePred = condUnet.out_sample.data as Float32Array;

      if (useCfg && uncondEmbeds) {
        const uncondUnet = await unet.run({
          sample: sampleTensor,
          timestep: timestepTensor,
          encoder_hidden_states: uncondEmbeds,
        });
        const uncondPred = uncondUnet.out_sample.data as Float32Array;
        noisePred = applyClassifierFreeGuidance(noisePred, uncondPred, guidanceScale);
      }

      const denoisedLatents = eulerStep(
        latents,
        noisePred,
        scheduleStep.sigma,
        scheduleStep.nextSigma
      );
      const unetDenoiseMs = Date.now() - tUnet0;

      // 4. VAE Latent Decoder -> RGBA PNG
      const tVae0 = Date.now();
      const vaeOut = await vaeDecoder.run({
        latent: new ort.Tensor('float32', denoisedLatents, [
          1,
          LATENT_CHANNELS,
          LATENT_HEIGHT,
          LATENT_WIDTH,
        ]),
      });
      const rgb = vaeOut.image.data as Float32Array;
      const vaeDecodeMs = Date.now() - tVae0;

      const fullSize = 512;
      const outW = input.aspectRatio === '9:16' ? 288 : fullSize;
      const outH = input.aspectRatio === '16:9' ? 288 : fullSize;
      const offsetX = Math.floor((fullSize - outW) / 2);
      const offsetY = Math.floor((fullSize - outH) / 2);

      const png = new PNG({ width: outW, height: outH });
      const channelStride = fullSize * fullSize;

      for (let y = 0; y < outH; y++) {
        const sy = y + offsetY;
        for (let x = 0; x < outW; x++) {
          const sx = x + offsetX;
          const srcIdx = sy * fullSize + sx;
          const r = Math.max(0, Math.min(255, Math.round((rgb[srcIdx] * 0.5 + 0.5) * 255)));
          const g = Math.max(
            0,
            Math.min(255, Math.round((rgb[channelStride + srcIdx] * 0.5 + 0.5) * 255))
          );
          const b = Math.max(
            0,
            Math.min(255, Math.round((rgb[channelStride * 2 + srcIdx] * 0.5 + 0.5) * 255))
          );

          const dstIdx = (y * outW + x) * 4;
          png.data[dstIdx] = r;
          png.data[dstIdx + 1] = g;
          png.data[dstIdx + 2] = b;
          png.data[dstIdx + 3] = 255;
        }
      }

      const pngBuffer = PNG.sync.write(png);
      const imageDataUrl = `data:image/png;base64,${pngBuffer.toString('base64')}`;

      return {
        imageDataUrl,
        pngBuffer,
        seed,
        guidanceUsed: useCfg ? guidanceScale : 0,
        mode,
        width: outW,
        height: outH,
        tokenStats: {
          activeTokens: tokenized.activeTokenCount,
          maxTokens: tokenized.maxTokens,
          truncated: tokenized.truncated,
        },
        timingsMs: {
          tokenizeAndEncode: tokenizeAndEncodeMs,
          vaeEncode: vaeEncodeMs,
          unetDenoise: unetDenoiseMs,
          vaeDecode: vaeDecodeMs,
          total: Date.now() - tStart,
        },
      };
    };

    const next = this.inferQueue.then(runTask, runTask);
    this.inferQueue = next.then(
      () => undefined,
      () => undefined
    );
    return next;
  }
}

export const localDiffusionEngine = new LocalDiffusionEngine();
