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

const REQUIRED_WEIGHT_FILES = [
  'tokenizer/vocab.json',
  'tokenizer/merges.txt',
  'text_encoder.onnx',
  'unet.onnx',
  'vae_decoder.onnx',
  'vae_encoder.onnx',
];

function getMissingWeightFiles(): string[] {
  return REQUIRED_WEIGHT_FILES.filter((relPath) => {
    const fullPath = path.join(MODEL_DIR, relPath);
    try {
      return !fs.statSync(fullPath).isFile() || fs.statSync(fullPath).size <= 1024;
    } catch {
      return true;
    }
  });
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
  vaeEncoder: ort.InferenceSession | null;
}

class LocalDiffusionEngine {
  private enginePromise: Promise<LoadedEngine> | null = null;
  private vaeEncoderPromise: Promise<ort.InferenceSession> | null = null;
  private inferQueue: Promise<void> = Promise.resolve();
  private state: 'idle' | 'loading' | 'ready' | 'error' = 'idle';
  private readonly sessionOptions: ort.InferenceSession.SessionOptions = {
    logSeverityLevel: 3,
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
    executionMode: 'sequential',
    graphOptimizationLevel: 'basic',
    enableCpuMemArena: false,
    enableMemPattern: false,
  };

  async ensureReady(): Promise<LoadedEngine> {
    if (!this.enginePromise) {
      this.state = 'loading';
      this.enginePromise = (async () => {
        const missingFiles = getMissingWeightFiles();
        if (missingFiles.length > 0) {
          throw new Error(
            `Local SDXS model files are missing or incomplete in ${MODEL_DIR}: ${missingFiles.join(', ')}. ` +
            'Automatic downloads are disabled. Install the model files locally before generating.'
          );
        }

        const tokenizer = new ClipBpeTokenizer(path.join(MODEL_DIR, 'tokenizer'));
        const textEncoder = await ort.InferenceSession.create(
          path.join(MODEL_DIR, 'text_encoder.onnx'),
          this.sessionOptions
        );
        const unet = await ort.InferenceSession.create(
          path.join(MODEL_DIR, 'unet.onnx'),
          this.sessionOptions
        );
        const vaeDecoder = await ort.InferenceSession.create(
          path.join(MODEL_DIR, 'vae_decoder.onnx'),
          this.sessionOptions
        );
        return { tokenizer, textEncoder, unet, vaeDecoder, vaeEncoder: null };
      })().then(
        (engine) => {
          this.state = 'ready';
          return engine;
        },
        (error: unknown) => {
          this.enginePromise = null;
          this.state = 'error';
          throw error;
        }
      );
    }
    return this.enginePromise;
  }

  private async ensureVaeEncoder(engine: LoadedEngine): Promise<ort.InferenceSession> {
    if (engine.vaeEncoder) return engine.vaeEncoder;
    if (!this.vaeEncoderPromise) {
      this.vaeEncoderPromise = ort.InferenceSession.create(
        path.join(MODEL_DIR, 'vae_encoder.onnx'),
        this.sessionOptions
      ).then(
        (session) => {
          engine.vaeEncoder = session;
          return session;
        },
        (error: unknown) => {
          this.vaeEncoderPromise = null;
          throw error;
        }
      );
    }
    return this.vaeEncoderPromise;
  }

  getStatus(): {
    ready: boolean;
    state: 'idle' | 'loading' | 'ready' | 'error';
    weightsPresent: boolean;
    ingredients: string[];
  } {
    const weightsPresent = getMissingWeightFiles().length === 0;
    return {
      ready: this.state === 'ready',
      state: this.state,
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
      const engine = await this.ensureReady();
      const { tokenizer, textEncoder, unet, vaeDecoder } = engine;

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
        const vaeEncoder = await this.ensureVaeEncoder(engine);
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
