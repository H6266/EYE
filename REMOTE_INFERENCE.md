# EYE — Remote Inference Documentation (Historical)
#
# This document describes the original remote-inference architecture.
# EYE is now a local-first platform. All generation features use local engines.
# Remote providers (Hugging Face, RunPod) remain optional for advanced features
# like Florence-2 object detection and SAM2 segmentation.

## Current Architecture

EYE uses a local-first architecture:

- **Image generation:** SDXS-512 ONNX CPU engine (local, always available)
- **AI inference gateway:** Mock provider by default; HF/RunPod optional
- **Video generation:** Not yet available locally (honest 503 response)
- **Music generation:** Not yet available locally (honest 503 response)
- **Voice generation:** Not yet available locally (honest 503 response)
- **Subtitle generation:** Local script-to-cue timing (always available)
- **Prompt enhancement:** Local text-only processing (always available)
- **Character analysis:** Local fallback descriptor (always available)

## Optional Remote Providers

The following providers can be configured for advanced features:

- **Hugging Face Inference API:** `HF_TOKEN` + `HF_*_ENDPOINT_URL` for Florence-2, SAM2
- **RunPod Serverless:** `RUNPOD_API_KEY` + `RUNPOD_*_ENDPOINT_ID` for custom workers

These are entirely optional and not required for core functionality.

## Historical Reference

The original architecture used Google Gemini Developer API (`GEMINI_API_KEY`)
for text/image generation and Vertex AI (`VERTEX_PROJECT_ID`, `VERTEX_LOCATION`)
for video generation via Veo. These have been removed in favor of local engines.