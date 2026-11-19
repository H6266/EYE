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
  const details = getVisualPromptDetails(input);
  const parts = [
    input.prompt.trim(),
    details.character ? `featuring character (${details.character})` : '',
    details.camera,
    details.lighting,
    details.style,
  ];
  return parts.filter(Boolean).join(', ');
}

export function buildVideoPrompt(input: PromptBuildInput): string {
  const sceneDescription = input.prompt.trim();
  const details = getVisualPromptDetails(input);
  const visualDetails = [
    details.character ? `Character reference: ${details.character}` : '',
    details.camera ? `Camera: ${details.camera}` : '',
    details.lighting ? `Lighting: ${details.lighting}` : '',
    details.style ? `Visual style: ${details.style}` : '',
  ].filter(Boolean);
  const parts = [
    `Primary scene description (follow faithfully):\n${sceneDescription}`,
  ];

  if (visualDetails.length > 0) {
    parts.push(`Additional visual direction:\n${visualDetails.join('\n')}`);
  }

  return parts.join('\n\n');
}

function getVisualPromptDetails(input: PromptBuildInput): {
  character?: string;
  camera?: string;
  lighting?: string;
  style: string;
} {
  let character: string | undefined;
  if (input.characterName || input.characterTraits) {
    character = [input.characterName, input.characterTraits]
      .filter(Boolean)
      .join(': ');
  }

  const framing = CAMERA_FRAMING_OPTIONS.find((f) => f.id === input.cameraFraming);
  const light = LIGHTING_OPTIONS.find((l) => l.id === input.lighting);
  return {
    character,
    camera: framing?.directive,
    lighting: light?.directive,
    style: STYLE_DIRECTIVES[input.style] || STYLE_DIRECTIVES.original,
  };
}

export function sanitizeNegativePrompt(neg?: string): string {
  const base = 'blurry, distorted anatomy, extra limbs, watermark, text artifacts, low quality';
  const trimmed = (neg || '').trim();
  return trimmed ? trimmed : base;
}
