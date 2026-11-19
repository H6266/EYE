# EYE — Local-First AI Studio

EYE is a self-contained, local-first AI generation platform. No external AI API
credentials or cloud login are required to start and operate its locally
supported features.

## Quick Start

```bash
cp .env.example .env   # no credentials needed
npm install
npm run build
NODE_ENV=production npx tsx server.ts
```

Open http://localhost:3000

## Local Features (No Credentials Required)

- **Image generation:** SDXS-512 ONNX CPU engine
- **Prompt enhancement:** Local text-only processing
- **Subtitle generation:** Local script-to-cue timing
- **Character analysis:** Local fallback descriptors
- **AI inference gateway:** Mock provider (local)

## Optional Remote Providers

The following are entirely optional and not required for core functionality:

- **Hugging Face:** `HF_TOKEN` for Florence-2 / SAM2 hosted endpoints
- **RunPod:** `RUNPOD_API_KEY` for custom serverless workers
- **S3-compatible storage:** `S3_*` variables for result persistence

## Features Not Yet Available Locally

- Video generation (honest 503 response, no silent fallback)
- Music generation (honest 503 response)
- Voice generation / TTS (honest 503 response)
- Audio transcription (honest 503 response)
- Cloud vision analysis (honest 503 response)

## Environment Variables

See `.env.example` for the complete list. No external AI credentials are required.