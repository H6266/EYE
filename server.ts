import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { localDiffusionEngine } from './src/engine/localDiffusionPipeline';
import { buildAutoSubtitleCues } from './src/utils/timelineEngine';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);

function createAiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function formatGeminiError(err: unknown): {
  status: number;
  message: string;
  requiresPaidKey: boolean;
} {
  const raw = err instanceof Error ? err.message : String(err);
  const lower = raw.toLowerCase();
  const isAuthOrQuota =
    lower.includes('403') ||
    lower.includes('permission_denied') ||
    lower.includes('429') ||
    lower.includes('resource_exhausted') ||
    lower.includes('paid') ||
    lower.includes('billing') ||
    lower.includes('quota');

  const isInvalidKey =
    lower.includes('api_key_invalid') || lower.includes('api key not valid');

  if (isInvalidKey) {
    return {
      status: 400,
      message:
        'Gemini API key is missing or invalid. Configure GEMINI_API_KEY in Settings > Secrets.',
      requiresPaidKey: false,
    };
  }
  if (isAuthOrQuota) {
    return {
      status: 402,
      message:
        'This generation model requires a billing-enabled Gemini API key in Settings > Secrets, or quota was reached.',
      requiresPaidKey: true,
    };
  }
  return {
    status: 500,
    message: raw || 'Unexpected error communicating with Gemini API.',
    requiresPaidKey: false,
  };
}

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

  app.get('/api/health', (_req: Request, res: Response) => {
    const hasKey = Boolean(
      process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'
    );
    res.json({
      ok: true,
      hasApiKey: hasKey,
      localDiffusion: localDiffusionEngine.getStatus(),
    });
  });

  // 1. Prompt Enhancer (Local Semantic Director fallback + optional Gemini Audio Transcription)
  app.post('/api/prompts/enhance', async (req: Request, res: Response) => {
    const {
      textPrompt = '',
      audioBase64,
      audioMimeType = 'audio/webm',
      style = 'cinematic',
    } = req.body ?? {};

    const ai = createAiClient();
    if (!ai) {
      const cleaned = String(textPrompt || '').trim();
      if (!cleaned) {
        res.status(400).json({
          error:
            'Enter a text prompt to enhance locally (audio transcription requires GEMINI_API_KEY).',
        });
        return;
      }
      const styleNote = STYLE_DIRECTIVES[style] ?? STYLE_DIRECTIVES.cinematic;
      res.json({
        transcribedAudio: '',
        title: cleaned.split(/\s+/).slice(0, 4).join(' '),
        visualPrompt: `${cleaned}, ${styleNote}, intricate focal detail, balanced depth of field`,
        narrationLine: `Behold ${cleaned.toLowerCase()}.`,
        musicCue: `Expressive ${style} orchestral soundtrack matching "${cleaned.slice(0, 48)}"`,
      });
      return;
    }

    try {
      let transcribedAudio = '';
      if (audioBase64) {
        const txResp = await ai.models.generateContent({
          model: 'gemini-3.5-transcribe',
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType: audioMimeType,
                  data: audioBase64,
                },
              },
              { text: 'Transcribe this spoken creative direction verbatim.' },
            ],
          },
        });
        transcribedAudio = (txResp.text ?? '').trim();
      }

      const combinedIdea = [textPrompt.trim(), transcribedAudio]
        .filter(Boolean)
        .join(' — Spoken direction: ');

      if (!combinedIdea) {
        res.status(400).json({ error: 'Provide a text prompt or audio recording.' });
        return;
      }

      const styleNote = STYLE_DIRECTIVES[style] ?? STYLE_DIRECTIVES.cinematic;

      const resp = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Transform this creative idea into a concise production package for a ${style} movie/animation scene.\nIdea: "${combinedIdea}"\nStyle guide: ${styleNote}`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING, description: 'Short 2-4 word scene title' },
              visualPrompt: {
                type: Type.STRING,
                description:
                  'Rich single-paragraph visual prompt optimized for image/video generation',
              },
              narrationLine: {
                type: Type.STRING,
                description: 'One expressive voiceover or character dialogue line for this scene',
              },
              musicCue: {
                type: Type.STRING,
                description: 'Concise soundtrack mood & instrumentation prompt',
              },
            },
            required: ['title', 'visualPrompt', 'narrationLine', 'musicCue'],
          },
        },
      });

      const parsed = JSON.parse((resp.text ?? '{}').trim());
      res.json({
        transcribedAudio,
        ...parsed,
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({
        error: formatted.message,
        requiresPaidKey: formatted.requiresPaidKey,
      });
    }
  });

  // Character Trait Extractor (local profile descriptor fallback + optional Gemini vision)
  app.post('/api/characters/analyze', async (req: Request, res: Response) => {
    const { imageBase64, mimeType = 'image/jpeg', name = 'Character' } = req.body ?? {};
    const ai = createAiClient();
    if (!ai) {
      res.json({
        traits: `${name}, consistent facial features, expressive eyes, distinctive signature outfit and colors`,
      });
      return;
    }
    try {
      if (!imageBase64) {
        res.status(400).json({ error: 'Character image is required.' });
        return;
      }
      const resp = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            { inlineData: { data: imageBase64, mimeType } },
            {
              text: `Describe the visual appearance of ${name} in this image in one concise comma-separated phrase (25-40 words) suitable for a text-to-image prompt: age/species, hair, face, outfit, colors, accessories. Return plain text only.`,
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
      });
      res.json({ traits: (resp.text ?? '').trim() });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
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
        const result = await localDiffusionEngine.generate({
          prompt: prompt.trim(),
          negativePrompt: String(negativePrompt || '').trim(),
          guidanceScale: effectiveGuidance,
          seed: takeSeed,
          aspectRatio,
          referenceImagePngBase64: useReferenceLatent
            ? String(referenceImagePngBase64)
            : undefined,
          img2imgStrength: Number(img2imgStrength) || 0.78,
        });

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

  // 3. Video Generation / Animation / Extension via Interactions API (Omni Flash)
  app.post('/api/media/video', async (req: Request, res: Response) => {
    const ai = createAiClient();
    if (!ai) {
      res.status(400).json({
        error:
          'GEMINI_API_KEY is not configured. Add your key in Settings > Secrets.',
        requiresPaidKey: false,
      });
      return;
    }

    try {
      const {
        prompt,
        style = 'cinematic',
        aspectRatio = '16:9',
        durationSeconds = 5,
        resolution = '720p',
        referenceImageBase64,
        referenceImageMimeType = 'image/jpeg',
        previousInteractionId,
      } = req.body ?? {};

      if (!prompt || typeof prompt !== 'string') {
        res.status(400).json({ error: 'A video motion prompt is required.' });
        return;
      }

      const clampedDur = Math.max(3, Math.min(15, Number(durationSeconds) || 5));
      const styleDirective = STYLE_DIRECTIVES[style] ?? '';
      const fullPrompt = `${prompt.trim()}. Visual direction: ${styleDirective}`;

      const inputPayload:
        | string
        | Array<
            | { type: 'image'; mime_type: string; data: string }
            | { type: 'text'; text: string }
          > = referenceImageBase64
        ? [
            {
              type: 'image',
              mime_type: referenceImageMimeType,
              data: referenceImageBase64,
            },
            { type: 'text', text: fullPrompt },
          ]
        : fullPrompt;

      const interaction = await ai.interactions.create(
        {
          model: 'gemini-omni-1.2-flash',
          ...(previousInteractionId
            ? { previous_interaction_id: previousInteractionId }
            : {}),
          input: inputPayload,
          background: false,
          store: true,
          stream: false,
          response_format: {
            type: 'video',
            aspect_ratio: aspectRatio === '9:16' ? '9:16' : '16:9',
            duration: `${clampedDur}s`,
            resolution: resolution === '1080p' ? '1080p' : '720p',
          },
        },
        { timeout: 300000 }
      );

      const videoPart = interaction.output_video;
      if (!videoPart?.data && !videoPart?.uri) {
        res.status(502).json({ error: 'Video model returned no playable video output.' });
        return;
      }

      const mimeType = videoPart.mime_type || 'video/mp4';
      const videoDataUrl = videoPart.data
        ? `data:${mimeType};base64,${videoPart.data}`
        : videoPart.uri || '';

      res.json({
        videoDataUrl,
        mimeType,
        interactionId: interaction.id,
        durationSeconds: clampedDur,
        promptUsed: fullPrompt,
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({
        error: formatted.message,
        requiresPaidKey: formatted.requiresPaidKey,
      });
    }
  });

  // 4. AI Music Generation (Google Lyria)
  app.post('/api/media/music', async (req: Request, res: Response) => {
    try {
      const {
        prompt,
        proMode = false,
        referenceImageBase64,
        referenceImageMimeType = 'image/jpeg',
      } = req.body ?? {};

      if (!prompt || typeof prompt !== 'string') {
        res.status(400).json({ error: 'A soundtrack description is required.' });
        return;
      }

      const ai = createAiClient();
      if (!ai) {
        res.status(400).json({
          error:
            'GEMINI_API_KEY is not configured. Upload your own music or use built-in studio stems.',
          requiresPaidKey: false,
        });
        return;
      }

      const model = proMode ? 'lyria-3-pro-preview' : 'lyria-3-clip-preview';
      const contents = referenceImageBase64
        ? {
            parts: [
              { text: prompt.trim() },
              {
                inlineData: {
                  data: referenceImageBase64,
                  mimeType: referenceImageMimeType,
                },
              },
            ],
          }
        : prompt.trim();

      const stream = await ai.models.generateContentStream({
        model,
        contents,
      });

      const audioChunks: Buffer[] = [];
      let lyrics = '';
      let mimeType = 'audio/mpeg';

      for await (const chunk of stream) {
        const parts = chunk.candidates?.[0]?.content?.parts;
        if (!parts) continue;
        for (const part of parts) {
          if (part.inlineData?.data) {
            if (audioChunks.length === 0 && part.inlineData.mimeType) {
              mimeType = part.inlineData.mimeType;
            }
            audioChunks.push(Buffer.from(part.inlineData.data, 'base64'));
          }
          if (part.text) {
            lyrics += part.text;
          }
        }
      }

      if (audioChunks.length === 0) {
        res.status(502).json({ error: 'Lyria returned no audio chunks.' });
        return;
      }

      const base64Audio = Buffer.concat(audioChunks).toString('base64');
      res.json({
        audioDataUrl: `data:${mimeType};base64,${base64Audio}`,
        mimeType,
        lyrics: lyrics.trim(),
        model,
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({
        error: formatted.message,
        requiresPaidKey: formatted.requiresPaidKey,
      });
    }
  });

  // 5. AI-Driven Voice Acting (Single-speaker & Dual-speaker Screenplay TTS)
  app.post('/api/media/voice', async (req: Request, res: Response) => {
    const ai = createAiClient();
    if (!ai) {
      res.status(400).json({
        error:
          'GEMINI_API_KEY is not configured. Add your key in Settings > Secrets.',
        requiresPaidKey: false,
      });
      return;
    }

    try {
      const {
        mode = 'single',
        text = '',
        voice = 'Kore',
        styleDirection = 'Expressive cinematic narrator',
        speakerA = { name: 'Elena', voice: 'Kore', style: 'Warm cinematic lead', line: '' },
        speakerB = { name: 'Julian', voice: 'Puck', style: 'Wry energetic co-star', line: '' },
      } = req.body ?? {};

      if (mode === 'dual') {
        if (!speakerA.line?.trim() || !speakerB.line?.trim()) {
          res.status(400).json({
            error: 'Both speaker dialogue lines are required in dual-speaker mode.',
          });
          return;
        }

        const interaction = await ai.interactions.create({
          model: 'gemini-3.8-flash-tts',
          input: [
            {
              type: 'text',
              text: `${speakerA.name}: ${speakerA.line.trim()}`,
              annotations: [
                {
                  type: 'speech_metadata',
                  speaker: speakerA.name,
                  style: speakerA.style,
                },
              ],
            },
            {
              type: 'text',
              text: `${speakerB.name}: ${speakerB.line.trim()}`,
              annotations: [
                {
                  type: 'speech_metadata',
                  speaker: speakerB.name,
                  style: speakerB.style,
                },
              ],
            },
          ],
          response_format: { type: 'audio' },
          generation_config: {
            speech_config: {
              speakers: [
                { speaker: speakerA.name, voice: speakerA.voice },
                { speaker: speakerB.name, voice: speakerB.voice },
              ],
            },
          },
        });

        const audio = interaction.output_audio;
        if (!audio?.data) {
          res.status(502).json({ error: 'Voice model returned no audio output.' });
          return;
        }

        const combinedScript = `${speakerA.name}: ${speakerA.line.trim()}\n${speakerB.name}: ${speakerB.line.trim()}`;
        const wordCount = combinedScript.split(/\s+/).filter(Boolean).length;
        res.json({
          audioDataUrl: `data:${audio.mime_type || 'audio/wav'};base64,${audio.data}`,
          mimeType: audio.mime_type || 'audio/wav',
          transcript: combinedScript,
          durationEstimate: Math.max(2.5, Number((wordCount * 0.42).toFixed(1))),
        });
        return;
      }

      if (!text || !text.trim()) {
        res.status(400).json({ error: 'Dialogue or narration script is required.' });
        return;
      }

      const useFlagship =
        /<laugh>|<gasp>|<breath>|\|mhm\||\|yeah\|/i.test(text) ||
        styleDirection.toLowerCase().includes('character');

      const interaction = await ai.interactions.create({
        model: useFlagship ? 'gemini-3.8-flash-tts' : 'gemini-3.8-flash-lite-tts',
        input: [
          {
            type: 'text',
            text: text.trim(),
            annotations: [
              {
                type: 'speech_metadata',
                style: styleDirection,
              },
            ],
          },
        ],
        response_format: { type: 'audio' },
        generation_config: {
          speech_config: [
            {
              language: 'en-US',
              voice,
            },
          ],
        },
      });

      const audio = interaction.output_audio;
      if (!audio?.data) {
        res.status(502).json({ error: 'TTS model returned no audio output.' });
        return;
      }

      const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
      res.json({
        audioDataUrl: `data:${audio.mime_type || 'audio/wav'};base64,${audio.data}`,
        mimeType: audio.mime_type || 'audio/wav',
        transcript: text.trim(),
        durationEstimate: Math.max(2, Number((wordCount * 0.4).toFixed(1))),
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({
        error: formatted.message,
        requiresPaidKey: formatted.requiresPaidKey,
      });
    }
  });

  // 6. Automated Subtitle Generation (Local Cue Alignment + Optional Audio Transcription)
  app.post('/api/subtitles/generate', async (req: Request, res: Response) => {
    const {
      audioBase64,
      audioMimeType = 'audio/wav',
      scriptText = '',
      totalDuration = 8,
      startTimeOffset = 0,
    } = req.body ?? {};

    const ai = createAiClient();
    if (!ai) {
      const cleaned = String(scriptText || '').trim();
      if (!cleaned) {
        res.status(400).json({
          error:
            'Provide script text to generate timed subtitle cues locally (audio transcription requires GEMINI_API_KEY).',
        });
        return;
      }
      const cues = buildAutoSubtitleCues(
        cleaned,
        Number(startTimeOffset) || 0,
        Number(totalDuration) || 8
      ).map((c) => ({
        startTime: c.startTime,
        duration: c.duration,
        speaker: c.speaker || '',
        text: c.text,
      }));
      res.json({ transcript: cleaned, cues });
      return;
    }

    try {

      let sourceTranscript = scriptText.trim();
      if (audioBase64) {
        const txResp = await ai.models.generateContent({
          model: 'gemini-3.5-transcribe',
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType: audioMimeType,
                  data: audioBase64,
                },
              },
              {
                text: 'Transcribe the speech accurately with punctuation and speaker labels if present.',
              },
            ],
          },
        });
        sourceTranscript = (txResp.text ?? '').trim();
      }

      if (!sourceTranscript) {
        res.status(400).json({ error: 'No speech or transcript found to subtitle.' });
        return;
      }

      const cueResp = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Break this dialogue/narration into clean, readable cinema subtitle cues spanning ${totalDuration} seconds starting at 0s.\nTranscript:\n"""${sourceTranscript}"""`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                startOffset: {
                  type: Type.NUMBER,
                  description: 'Start time in seconds relative to 0',
                },
                duration: {
                  type: Type.NUMBER,
                  description: 'Duration of this cue in seconds (min 1.2)',
                },
                speaker: {
                  type: Type.STRING,
                  description: 'Optional speaker name or empty string',
                },
                text: {
                  type: Type.STRING,
                  description: 'Subtitle caption line (max 75 characters)',
                },
              },
              required: ['startOffset', 'duration', 'text'],
            },
          },
        },
      });

      const rawCues = JSON.parse((cueResp.text ?? '[]').trim()) as Array<{
        startOffset: number;
        duration: number;
        speaker?: string;
        text: string;
      }>;

      const cues = rawCues.map((c) => ({
        startTime: Number((Number(startTimeOffset) + Math.max(0, c.startOffset)).toFixed(2)),
        duration: Math.max(1.2, Number(c.duration || 2.5)),
        speaker: c.speaker?.trim() || '',
        text: c.text.trim(),
      }));

      res.json({ transcript: sourceTranscript, cues });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({
        error: formatted.message,
        requiresPaidKey: formatted.requiresPaidKey,
      });
    }
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
    // Pre-warm local neural diffusion ingredients (CLIP Text Encoder, UNet Brain, VAE Decoder/Encoder)
    localDiffusionEngine.ensureReady().catch((err) => {
      console.warn('Local diffusion warmup deferred:', err?.message || err);
    });
  });
}

startServer();
