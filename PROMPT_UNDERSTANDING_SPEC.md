# PROMPT UNDERSTANDING & STRUCTURED PROMPT PROCESSING SPECIFICATION

> Historical specification. Use [IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md](./IMAGE_GENERATION_INTELLIGENCE_UPGRADE_PLAN.md) for the current prompt-interpreter architecture and implementation gates.

**Part of:** Auteur Studio Image Generation Intelligence Upgrade  
**Status:** SPECIFICATION (not yet implemented)  
**Date:** 2026-11-16

---

## 1. OBJECTIVE

When a user describes an image in ordinary human language, Auteur Studio must generate an image that follows the description as accurately as possible. The prompt understanding layer ensures that:

- All requested subjects, objects, colors, positions, camera angles, lighting, and environment details are preserved
- No important user instructions are discarded or invented
- The original prompt is preserved so users can compare with the optimized version
- Lightweight local processing (no excessive inference overhead)

---

## 2. PROMPT PROCESSING PIPELINE

```
USER INPUT (natural language)
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 1: Subject & Entity Extraction   │
│  - Identify all requested subjects      │
│  - Extract visual characteristics       │
│  - Count objects (specific quantities)  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 2: Action & Relationship Parsing │
│  - Identify actions and poses           │
│  - Extract spatial relationships        │
│  - Identify object positions            │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 3: Technical Instruction Extraction│
│  - Camera angle, framing, lens          │
│  - Lighting conditions                  │
│  - Environment and background           │
│  - Weather and atmospheric conditions   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 4: Constraint & Negative Parsing │
│  - What should NOT be in the image      │
│  - What must NOT be invented            │
│  - Security/privacy constraints         │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 5: Structured Output Generation  │
│  - Preserve original prompt             │
│  - Produce optimized generation prompt  │
│  - Output structured scene specification│
└─────────────────────────────────────────┘
        │
        ▼
GENERATION MODEL (Z-Image-Turbo / FLUX.2 klein 4B)
```

---

## 3. STRUCTURED SCENE SPECIFICATION (JSON SCHEMA)

```json
{
  "original_prompt": "string (preserved verbatim)",
  "optimized_prompt": "string (rewritten for model, preserving all elements)",
  "subjects": [
    {
      "type": "person | animal | object",
      "count": 1,
      "attributes": {
        "age": "string",
        "gender": "string",
        "ethnicity": "string",
        "clothing": "string",
        "accessories": ["string"],
        "expression": "string",
        "pose": "string"
      },
      "position": "foreground | middle-ground | background | specific",
      "spatial_relationship": "string (relative to other subjects)"
    }
  ],
  "environment": {
    "location": "string",
    "background_elements": ["string"],
    "weather": "string",
    "time_of_day": "string",
    "season": "string"
  },
  "camera": {
    "angle": "string",
    "framing": "string",
    "lens": "string",
    "depth_of_field": "string"
  },
  "lighting": {
    "type": "string",
    "color_temperature": "string",
    "direction": "string"
  },
  "style": {
    "visual_style": "cinematic | anime | cartoon | watercolor | ...",
    "quality": "photorealistic | stylized",
    "color_palette": ["string"]
  },
  "constraints": {
    "must_include": ["string"],
    "must_not_include": ["string"],
    "colors_required": ["string"],
    "object_counts": { "object_name": number }
  },
  "security": {
    "no_pin_visible": true,
    "no_password_spoken": true,
    "user_retains_control": true
  },
  "model_recommendation": "z-image-turbo | flux-klein-4b | qwen-2512 | moss-sft",
  "estimated_vram_gb": 16,
  "estimated_steps": 8
}
```

---

## 4. LIGHTWEIGHT PROMPT PROCESSING MODEL

### 4.1 Model Selection Criteria

| Criterion | Requirement |
|-----------|-------------|
| Size | < 2B parameters (fast, low overhead) |
| License | Apache-2.0 or MIT (commercial) |
| Capability | Instruction following, entity extraction, structured output |
| VRAM | < 4GB (can run alongside generation model or on CPU) |
| Speed | < 1s inference on CPU |

### 4.2 Candidate Models

| Model | Params | License | VRAM | Use Case |
|-------|--------|---------|------|----------|
| Qwen2.5-0.5B-Instruct | 0.5B | Apache-2.0 | ~1GB | Prompt parsing, entity extraction |
| Qwen2.5-1.5B-Instruct | 1.5B | Apache-2.0 | ~2GB | Complex prompt understanding |
| Phi-3.5-mini-Instruct | 3.8B | MIT | ~4GB | Advanced reasoning, structured output |
| Llama-3.2-1B-Instruct | 1B | Meta License | ~1GB | Simple prompt rewriting |

### 4.3 Recommended: Qwen2.5-0.5B-Instruct

- **HF:** https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- **License:** Apache-2.0
- **Size:** ~1GB (FP16), ~0.5GB (INT4)
- **Speed:** Fast on CPU (< 500ms for short prompts)
- **Capability:** Good instruction following, JSON output support

---

## 5. PROMPT INTERPRETER IMPLEMENTATION SPEC

### 5.1 TypeScript Interface (`src/engine/promptInterpreter.ts`)

```typescript
export interface ParsedSubject {
  type: 'person' | 'animal' | 'object';
  count: number;
  attributes: Record<string, string>;
  position: 'foreground' | 'middle-ground' | 'background';
  spatialRelationship?: string;
}

export interface ParsedScene {
  originalPrompt: string;
  optimizedPrompt: string;
  subjects: ParsedSubject[];
  environment: {
    location: string;
    backgroundElements: string[];
    weather: string;
    timeOfDay: string;
  };
  camera: {
    angle: string;
    framing: string;
    lens: string;
  };
  lighting: {
    type: string;
    colorTemperature: string;
  };
  constraints: {
    mustInclude: string[];
    mustNotInclude: string[];
    colorsRequired: string[];
    objectCounts: Record<string, number>;
  };
  modelRecommendation: string;
  estimatedVramGb: number;
}

export class PromptInterpreter {
  async parse(prompt: string): Promise<ParsedScene>;
  async optimize(prompt: string, modelId: string): Promise<string>;
}
```

### 5.2 Python Inference Service Endpoint

```
POST /api/v1/vlm/parse-prompt
Request: { "prompt": "string", "return_json": true }
Response: ParsedScene JSON

POST /api/v1/vlm/optimize-prompt
Request: { "prompt": "string", "target_model": "z-image-turbo" }
Response: { "optimized_prompt": "string", "model_recommendation": "string" }
```

---

## 6. RULES FOR PROMPT PROCESSING

### 6.1 Preservation Rules (MUST)
1. **Never discard** requested subjects, objects, or details
2. **Never invent** new objects, characters, clothing, or scene elements not requested
3. **Preserve** specific object counts and positions
4. **Preserve** camera angles, lighting, environment, weather
5. **Preserve** color specifications
6. **Preserve** the original prompt for user comparison

### 6.2 Optimization Rules (MAY)
1. Reorder terms for better model attention (front-load important subjects)
2. Add technical quality descriptors (photorealistic, 8k, cinematic) only if user hasn't specified style
3. Expand abbreviations (e.g., "DSLR" → "digital single-lens reflex camera")
4. Normalize terminology for the target model's training distribution

### 6.3 Security Rules (MUST)
1. Never include PIN, OTP, password, or credentials in prompts
2. Never suggest speaking confidential information aloud
3. Always preserve user control of the transaction/device
4. Flag prompts that request insecure behaviors

---

## 7. INTEGRATION POINTS WITH AUTEUR STUDIO

### 7.1 GeneratePanel Modifications
- Add "Prompt Understanding" toggle (faithful vs enhanced interpretation)
- Show original prompt vs optimized prompt side-by-side
- Allow user to edit optimized prompt before generation
- Display model recommendation and estimated VRAM

### 7.2 API Changes (`server.ts`)
- Add `POST /api/prompts/parse` endpoint (calls Python VLM service)
- Add `POST /api/prompts/optimize` endpoint (calls Python VLM service)
- Modify `POST /api/media/image` to accept optional `parsed_scene` and `optimized_prompt`

### 7.3 Fallback Behavior
- If Python VLM service is unavailable, use original prompt unchanged
- If VLM parsing fails, log error and continue with original prompt
- Never block generation due to prompt understanding failure

---

## 8. EVALUATION CRITERIA FOR PROMPT UNDERSTANDING

| Metric | Target |
|--------|--------|
| Subject recall | > 95% (all requested subjects present) |
| Object count accuracy | > 90% |
| Color accuracy | > 90% |
| No hallucinated elements | > 95% |
| Inference overhead | < 1s on CPU |
| Original prompt preservation | 100% (always available for comparison) |

---

## 9. IMPLEMENTATION PHASE

**Phase 1 (Week 3-4 of integration):**
- Deploy Qwen2.5-0.5B-Instruct in Python inference service
- Implement `/api/v1/vlm/parse-prompt` and `/api/v1/vlm/optimize-prompt` endpoints
- Create TypeScript `PromptInterpreter` class
- Add "Prompt Understanding" UI toggle in GeneratePanel

**Phase 2 (Week 5-6):**
- Connect optimized prompts to generation router
- Add model recommendation based on parsed scene complexity
- Human evaluation of prompt fidelity improvement