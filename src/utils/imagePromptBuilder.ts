import {
  CAMERA_FRAMING_OPTIONS,
  LIGHTING_OPTIONS,
  STYLE_DIRECTIVES,
} from '../data/imageEngines';
import { VisualStyle } from '../types/studio';

export interface PromptBuildInput {
  prompt: string;
  style: VisualStyle;
  cameraFraming?: string;
  lighting?: string;
  characterName?: string;
  characterTraits?: string;
}

export function buildRichVisualPrompt(input: PromptBuildInput): string {
  const parts: string[] = [input.prompt.trim()];

  if (input.characterName || input.characterTraits) {
    const charDesc = [input.characterName, input.characterTraits]
      .filter(Boolean)
      .join(': ');
    if (charDesc) {
      parts.push(`featuring character (${charDesc})`);
    }
  }

  const framing = CAMERA_FRAMING_OPTIONS.find((f) => f.id === input.cameraFraming);
  if (framing?.directive) {
    parts.push(framing.directive);
  }

  const light = LIGHTING_OPTIONS.find((l) => l.id === input.lighting);
  if (light?.directive) {
    parts.push(light.directive);
  }

  const styleDirective = STYLE_DIRECTIVES[input.style] || STYLE_DIRECTIVES.original;
  parts.push(styleDirective);

  return parts.filter(Boolean).join(', ');
}

export function sanitizeNegativePrompt(neg?: string): string {
  const base = 'blurry, distorted anatomy, extra limbs, watermark, text artifacts, low quality';
  const trimmed = (neg || '').trim();
  return trimmed ? trimmed : base;
}
