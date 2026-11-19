import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { localDiffusionEngine } from './src/engine/localDiffusionPipeline';
import { generateImage } from './src/engine/generationRouter';
import { pythonDiffusionClient } from './src/engine/pythonDiffusionClient';
import {
  AiGatewayError,
  AiInferenceGateway,
  AiJobRequest,
} from './src/engine/aiInferenceGateway';
import { AiProvider } from './src/engine/aiModelRegistry';
import { VideoGenerationService } from './src/engine/videoGeneration';
import { IMAGE_ENGINES } from './src/data/imageEngines';
import { ImageEngineId, PYTHON_ENGINE_IDS } from './src/types/studio';
import { buildAutoSubtitleCues } from './src/utils/timelineEngine';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const aiInferenceGateway = new AiInferenceGateway();
const videoGenerationService = new VideoGenerationService();

const STYLE_DIRECTIVES: Record<string, string> = {
  original: 'High-fidelity studio composition, crisp lighting and natural texture.',
  cinematic:
    'Anamorphic 35mm cinema frame, shallow depth of field, rich film grain, volumetric atmospheric lighting, teal and warm amber grading.',
  cartoon2d:
    'Hand-drawn 2D feature animation frame, bold expressive ink outlines, cel-shaded vibrant watercolor backgrounds, dynamic character silhouette.',
  anime:
    'High-budget theatrical Japanese anime still, luminous twilight sky, crisp cel shading, expressive lens bloom, detailed painted background.',
  claymation:
    'Tactile stop-motion claymation miniature set, handcrafted sculpted textures, practical warm tungsten studio lighting, shallow miniature lens focus.',
  noir: 'Classic 35mm neo-noir cinema still, dramatic chiaroscuro shadows, rain-slick reflections, atmospheric mist, high-contrast silver monochrome.',
  pixar3d:
    'Stylized 3D animated feature film frame, soft subsurface scattering, expressive stylized character proportions, global illumination.',
  watercolor:
    'Hand-painted gouache and watercolor storybook illustration, cold-press paper texture, delicate ink linework.',
};

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '60mb' }));

  app.get('/api/health', async (_req: Request, res: Response) => {
    res.json({
      ok: true,
      hasApiKey: false,
      localDiffusion: localDiffusionEngine.getStatus(),
      pythonService: pythonDiffusionClient.getStatus(),
    });
  });

  app.get('/api/ai/models', (req: Request, res: Response) => {
    const provider = typeof req.query.provider === 'string'
      ? req.query.provider as AiProvider
      : undefined;
    res.json({
      defaultProvider: process.env.AI_INFERENCE_PROVIDER ?? 'mock',
      models: aiInferenceGateway.listModels(provider),
    });
  });

  app.post('/api/ai/jobs', async (req: Request, res: Response) => {
    try {
      const job = await aiInferenceGateway.createJob(req.body as AiJobRequest);
      res.status(202).json(job);
    } catch (error) {
      if (error instanceof AiGatewayError) {
        res.status(error.statusCode).json({ error: error.message });
        return;
      }
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Failed to create inference job.',
      });
    }
  });

  app.get('/api/ai/jobs/:jobId', (req: Request, res: Response) => {
    const job = aiInferenceGateway.getJob(req.params.jobId);
    if (!job) {
      res.status(404).json({ error: 'AI job not found or expired.' });
      return;
    }
    res.json(job);
  });

  app.get('/api/ai/jobs/:jobId/result', (req: Request, res: Response) => {
    const job = aiInferenceGateway.getJob(req.params.jobId);
    if (!job) {
      res.status(404).json({ error: 'AI job not found or expired.' });
      return;
    }
    if (job.status !== 'completed') {
      res.status(409).json({ error: `AI job is ${job.status}.` });
      return;
    }
    const result = aiInferenceGateway.getResult(job.id);
    if (!result) {
      if (job.resultUrl && /^https:\/\//.test(job.resultUrl)) {
        res.redirect(job.resultUrl);
        return;
      }
      res.status(404).json({ error: 'This AI job has no image result.' });
      return;
    }
    res
      .status(200)
      .type(result.contentType)
      .set('Cache-Control', 'private, max-age=900')
      .send(Buffer.from(result.bytes));
  });

  // 1. Local text-only prompt enhancement
  app.post('/api/prompts/enhance', async (req: Request, res: Response) => {
    const {
      textPrompt = '',
      audioBase64,
      style = 'cinematic',
    } = req.body ?? {};

    if (audioBase64) {
      res.status(503).json({
        error: 'Audio transcription is unavailable in local-only mode. Enter the spoken direction as text.',
        feature: 'local-audio-transcription-unavailable',
      });
      return;
    }

    const cleaned = String(textPrompt || '').trim();
    if (!cleaned) {
      res.status(400).json({
        error: 'Enter a text prompt to enhance locally.',
      });
      return;
    }

    const styleNote = STYLE_DIRECTIVES[style] ?? STYLE_DIRECTIVES.cinematic;
    res.json({
      transcribedAudio: '',
      title: cleaned.split(/\\s+/).slice(0, 4).join(' '),
      visualPrompt: `${cleaned}, ${styleNote}, intricate focal detail, balanced depth of field`,
      narrationLine: `Behold ${cleaned.toLowerCase()}.`,
      musicCue: `Expressive ${style} orchestral soundtrack matching "${cleaned.slice(0, 48)}"`,
      mode: 'local-text',
    });
  });

  // Character Trait Extractor (local profile descriptor only)
  app.post('/api/characters/analyze', async (req: Request, res: Response) => {
    const { imageBase64, mimeType = 'image/jpeg', name = 'Character' } = req.body ?? {};
    // Local-only mode: provide a descriptive fallback trait string.
    // A local vision model is not yet implemented.
    res.json({
      traits: `${name}, consistent facial features, expressive eyes, distinctive signature outfit and colors`,
      mode: 'local-fallback',
      ...(imageBase64 ? {} : { note: 'Image provided but local vision analysis is not yet available.' }),
    });
  });

  // 2. Self-Contained Local Neural Diffusion Text-to-Picture & Img2Img Engine
  // Ingredients: CLIP BPE Tokenizer -> CLIP ViT-L/14 Text Encoder -> SDXS-512 UNet Brain -> Euler Sampler + CFG -> TAESD VAE Decoder (+ SD VAE Encoder)
  app.post('/api/media/image', async (req: Request, res: Response) => {
    try {
      const {
        prompt,
        negativePrompt = 'blurry, distorted anatomy, extra limbs, watermark, low quality',
        engineId = 'local-sdxs',
        aspectRatio = '16:9',
        guidance = 0,
        seed = 0,
        randomizeSeed = true,
        img2imgStrength = 0.78,
        batchCount = 1,
        referenceImagePngBase64,
      } = req.body ?? {};

      if (!prompt || typeof prompt !== 'string') {
        res.status(400).json({ error: 'A scene prompt is required.' });
        return;
      }
      const selectedEngine = IMAGE_ENGINES.find((engine) => engine.id === engineId);
      if (!selectedEngine) {
        res.status(400).json({ error: `Unknown image engine '${String(engineId)}'.` });
        return;
      }
      const selectedEngineId = selectedEngine.id;

      const safeCount = Math.max(1, Math.min(4, Number(batchCount) || 1));
      const baseSeed =
        randomizeSeed || !seed
          ? Math.floor(Math.random() * 1_000_000_000)
          : Number(seed);

      const effectiveGuidance =
        engineId === 'local-sdxs-cfg' || engineId === 'local-sdxs-img2img'
          ? Math.max(1.05, Math.min(1.35, Number(guidance) || 1.2))
          : 0;

      const useReferenceLatent =
        Boolean(referenceImagePngBase64) && engineId === 'local-sdxs-img2img';

      const takes = [];
      for (let i = 0; i < safeCount; i++) {
        const takeSeed = baseSeed + i * 137;
        const isPythonEngine = PYTHON_ENGINE_IDS.includes(selectedEngineId);
        const result = await generateImage(
          isPythonEngine
            ? {
                model: selectedEngineId,
                prompt: prompt.trim(),
                negativePrompt: String(negativePrompt || '').trim(),
                width: aspectRatio === '9:16' ? 576 : 1024,
                height: aspectRatio === '16:9' ? 576 : 1024,
                numInferenceSteps:
                  engineId === 'python-zimage-turbo'
                    ? 8
                    : engineId === 'python-flux-klein'
                      ? 4
                      : engineId === 'python-zimage-base'
                        ? 50
                        : 50,
                guidanceScale:
                  engineId === 'python-zimage-turbo'
                    ? 0
                    : engineId === 'python-flux-klein'
                      ? 1
                      : engineId === 'python-qwen-2512'
                        ? 4
                        : engineId === 'python-moss-sft'
                          ? 3
                          : 5,
                seed: takeSeed,
                returnBase64: true,
              }
            : {
                prompt: prompt.trim(),
                negativePrompt: String(negativePrompt || '').trim(),
                guidanceScale: effectiveGuidance,
                seed: takeSeed,
                aspectRatio,
                referenceImagePngBase64: useReferenceLatent
                  ? String(referenceImagePngBase64)
                  : undefined,
                img2imgStrength: Number(img2imgStrength) || 0.78,
              },
          selectedEngineId
        );

        const modelLabel = selectedEngine.name;
        if ('imageDataUrl' in result) {
          takes.push({
            imageDataUrl: result.imageDataUrl,
            seed: result.seed,
            modelLabel:
              result.mode === 'img2img'
                ? 'Local VAE Img2Img + UNet'
                : effectiveGuidance > 1
                  ? 'Local SDXS-512 + CFG'
                  : 'Local SDXS-512 Neural',
            tokenStats: result.tokenStats,
            timingsMs: result.timingsMs,
          });
        } else {
          if (result.error) {
            res.status(502).json({ error: result.error });
            return;
          }
          const imageDataUrl =
            result.imageBase64 ??
            (result.imageUrl?.startsWith('data:') ? result.imageUrl : '');
          if (!imageDataUrl) {
            res.status(502).json({ error: 'The selected engine returned no image.' });
            return;
          }
          takes.push({
            imageDataUrl,
            seed: result.seed,
            modelLabel,
            timingsMs: {
              tokenizeAndEncode: 0,
              vaeEncode: 0,
              unetDenoise: result.timingsMs.inferenceMs,
              vaeDecode: 0,
              total: result.timingsMs.totalMs,
            },
          });
        }
      }

      res.json({
        imageDataUrl: takes[0].imageDataUrl,
        images: takes,
        promptUsed: prompt.trim(),
        model: takes[0].modelLabel,
        seed: takes[0].seed,
        tokenStats: takes[0].tokenStats,
        timingsMs: takes[0].timingsMs,
      });
    } catch (err) {
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : 'Local diffusion inference failed.',
      });
    }
  });

  app.post('/api/quality/objects', async (req: Request, res: Response) => {
    const { imageBase64 } = req.body ?? {};
    if (typeof imageBase64 !== 'string' || !imageBase64.trim()) {
      res.status(400).json({ error: 'An image is required for inspection.' });
      return;
    }

    const result = await pythonDiffusionClient.detectObjects(imageBase64);
    if (result.error) {
      res.status(503).json({ error: result.error });
      return;
    }
    res.json({ detections: result.detections });
  });

  // 3. Video generation (local engine not yet implemented)
  app.post('/api/media/video/jobs', (req: Request, res: Response) => {
    try {
      const {
        prompt,
        aspectRatio = '16:9',
        durationSeconds = 4,
        resolution = '720p',
        referenceImageBase64,
        referenceImageMimeType = 'image/jpeg',
      } = req.body ?? {};

      if (typeof prompt !== 'string' || !prompt.trim()) {
        res.status(400).json({ error: 'A video motion prompt is required.' });
        return;
      }
      const job = videoGenerationService.createJob({
        prompt: prompt.trim(),
        aspectRatio: aspectRatio === '9:16' ? '9:16' : '16:9',
        durationSeconds: Number(durationSeconds),
        resolution: resolution === '1080p' ? '1080p' : '720p',
        referenceImageBase64:
          typeof referenceImageBase64 === 'string'
            ? referenceImageBase64
            : undefined,
        referenceImageMimeType:
          typeof referenceImageMimeType === 'string'
            ? referenceImageMimeType
            : undefined,
      });
      res.status(202).json(job);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const status = message.includes('queue is full')
        ? 429
        : 400;
      res.status(status).json({ error: message });
    }
  });

  app.get('/api/media/video/jobs/:jobId', (req: Request, res: Response) => {
    const job = videoGenerationService.getJob(req.params.jobId);
    if (!job) {
      res.status(404).json({ error: 'Video job not found or expired.' });
      return;
    }
    res.json(job);
  });

  app.get('/api/media/video/jobs/:jobId/result', (req: Request, res: Response) => {
    const result = videoGenerationService.getResult(req.params.jobId);
    if (!result) {
      res.status(404).json({ error: 'Video result not found or expired.' });
      return;
    }

    const range = req.header('range');
    res.set({
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'private, max-age=900',
      'Content-Type': result.mimeType,
    });
    if (range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (!match || (!match[1] && !match[2])) {
        res.status(416).set('Content-Range', `bytes */${result.bytes.length}`).end();
        return;
      }
      const start = match[1] ? Number(match[1]) : 0;
      const suffixLength = Number(match[2]);
      const end = match[1]
        ? match[2]
          ? Math.min(suffixLength, result.bytes.length - 1)
          : result.bytes.length - 1
        : result.bytes.length - 1;
      const rangeStart =
        !match[1] && suffixLength > 0
          ? Math.max(0, result.bytes.length - suffixLength)
          : start;
      if (rangeStart > end || rangeStart >= result.bytes.length) {
        res.status(416).set('Content-Range', `bytes */${result.bytes.length}`).end();
        return;
      }
      res
        .status(206)
        .set({
          'Content-Length': String(end - rangeStart + 1),
          'Content-Range': `bytes ${rangeStart}-${end}/${result.bytes.length}`,
        })
        .send(result.bytes.subarray(rangeStart, end + 1));
      return;
    }
    res.set('Content-Length', String(result.bytes.length)).send(result.bytes);
  });

  // 4. AI Music Generation (local audio engine not yet implemented)
  app.post('/api/media/music', async (req: Request, res: Response) => {
    const { prompt } = req.body ?? {};
    if (!prompt || typeof prompt !== 'string') {
      res.status(400).json({ error: 'A soundtrack description is required.' });
      return;
    }
    res.status(503).json({
      error: 'Music generation is not available in local-only mode. A local audio engine is not yet implemented.',
      feature: 'local-music-generation-unavailable',
    });
  });

  // 5. AI-Driven Voice Acting (local TTS engine not yet implemented)
  app.post('/api/media/voice', async (req: Request, res: Response) => {
    const { mode = 'single', text = '' } = req.body ?? {};
    if (mode === 'dual') {
      const { speakerA, speakerB } = req.body ?? {};
      if (!speakerA?.line?.trim() || !speakerB?.line?.trim()) {
        res.status(400).json({
          error: 'Both speaker dialogue lines are required in dual-speaker mode.',
        });
        return;
      }
       }
    if (mode === 'single' && (!text || !text.trim())) {
      res.status(400).json({ error: 'Dialogue or narration script is required.' });
      return;
    }
    res.status(503).json({
      error: 'Voice generation is not available in local-only mode. A local TTS engine is not yet implemented.',
      feature: 'local-tts-unavailable',
    });
  });

  // 6. Local script-to-subtitle cue generation
  app.post('/api/subtitles/generate', async (req: Request, res: Response) => {
    const {
      audioBase64,
      scriptText = '',
      totalDuration = 8,
      startTimeOffset = 0,
    } = req.body ?? {};

    if (audioBase64) {
      res.status(503).json({
        error: 'Audio transcription is unavailable in local-only mode. Provide the transcript as script text.',
        feature: 'local-audio-transcription-unavailable',
      });
      return;
    }

    const cleaned = String(scriptText || '').trim();
    if (!cleaned) {
      res.status(400).json({
        error: 'Provide script text to generate timed subtitle cues locally.',
      });
      return;
    }

    const duration = Number(totalDuration);
    const offset = Number(startTimeOffset);
    const cues = buildAutoSubtitleCues(
      cleaned,
      Number.isFinite(offset) ? offset : 0,
      Number.isFinite(duration) && duration > 0 ? duration : 8
    ).map((cue) => ({
      startTime: cue.startTime,
      duration: cue.duration,
      speaker: cue.speaker || '',
      text: cue.text,
    }));

    res.json({
      transcript: cleaned,
      cues,
      mode: 'local-script',
    });
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Auteur Studio server listening on http://localhost:${PORT}`);
    if (process.env.PREWARM_LOCAL_DIFFUSION === 'true') {
      localDiffusionEngine.ensureReady().catch((err) => {
        console.warn('Local diffusion warmup deferred:', err?.message || err);
      });
    }
  });
}

startServer();
