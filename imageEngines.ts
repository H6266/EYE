import { AspectRatio, ImageEngineId, VisualStyle } from '../types/studio';

export interface ImageEngineSpec {
  id: ImageEngineId;
  name: string;
  architecture: string;
  ingredientsSummary: string;
  summary: string;
  supportsNegativePrompt: boolean;
  supportsCharacterImg2Img: boolean;
  supportsGuidance: boolean;
  defaultGuidance: number;
  maxGuidance: number;
}

export const IMAGE_ENGINES: ImageEngineSpec[] = [
  {
    id: 'local-sdxs',
    name: 'Local SDXS-512 Neural',
    architecture: 'CLIP + UNet + TAESD VAE',
    ingredientsSummary: 'Zero APIs · On-Server ONNX',
    summary:
      'Self-contained 4-ingredient pipeline (CLIP ViT-L/14 Text Encoder, SDXS DreamShaper UNet Brain, Euler Sampler, TAESD VAE Decoder). Fast ~2.5s high-accuracy generation.',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: true,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
  },
  {
    id: 'local-sdxs-cfg',
    name: 'Local SDXS-512 + CFG Guidance',
    architecture: 'Dual-Pass UNet + Negative Prompt',
    ingredientsSummary: 'Zero APIs · Negative Filtering',
    summary:
      'Runs conditional and unconditional UNet Brain passes with calibrated Classifier-Free Guidance to suppress unwanted elements from your Negative Prompt.',
    supportsNegativePrompt: true,
    supportsCharacterImg2Img: true,
    supportsGuidance: true,
    defaultGuidance: 1.2,
    maxGuidance: 1.35,
  },
  {
    id: 'local-sdxs-img2img',
    name: 'Local VAE Latent Img2Img',
    architecture: 'VAE Encoder + UNet + VAE Decoder',
    ingredientsSummary: 'Zero APIs · Photo/Character Guided',
    summary:
      'Encodes your uploaded character photo or reference frame via the local SD VAE Encoder into 64×64 latents, applies controlled noise, and restyles with the UNet Brain.',
    supportsNegativePrompt: true,
    supportsCharacterImg2Img: true,
    supportsGuidance: true,
    defaultGuidance: 1.15,
    maxGuidance: 1.35,
  },
];

export const CAMERA_FRAMING_OPTIONS = [
  { id: 'auto', label: 'Director Choice', directive: '' },
  {
    id: 'wide',
    label: 'Wide Establishing',
    directive: 'wide-angle establishing shot, deep environmental scale',
  },
  {
    id: 'closeup',
    label: 'Close-Up Portrait',
    directive: 'expressive close-up portrait, crisp facial detail, shallow depth of field',
  },
  {
    id: 'low-hero',
    label: 'Low-Angle Hero',
    directive: 'dramatic low-angle hero framing looking upward, monumental silhouette',
  },
  {
    id: 'ots',
    label: 'Over-the-Shoulder',
    directive: 'over-the-shoulder cinematic framing, foreground depth',
  },
] as const;

export const LIGHTING_OPTIONS = [
  { id: 'auto', label: 'Scene Natural', directive: '' },
  {
    id: 'golden',
    label: 'Golden Hour',
    directive: 'warm golden hour volumetric sunbeams, long soft shadows',
  },
  {
    id: 'neon-rain',
    label: 'Rainy Lantern Glow',
    directive: 'wet reflective surfaces, warm glowing lanterns against cool twilight mist',
  },
  {
    id: 'studio-rim',
    label: 'Studio Rim Light',
    directive: 'crisp three-point studio lighting with warm rim highlights',
  },
  {
    id: 'chiaroscuro',
    label: 'Chiaroscuro Noir',
    directive: 'high-contrast chiaroscuro dramatic shadows and single key light',
  },
] as const;

export const STYLE_DIRECTIVES: Record<VisualStyle, string> = {
  original: 'clean studio composition, natural color balance and sharp detail',
  cinematic:
    'anamorphic 35mm cinema still, shallow depth of field, subtle film grain, teal and amber color grading',
  cartoon2d:
    'hand-drawn 2D animated feature film frame, bold expressive ink outlines, cel-shaded vibrant watercolor sky',
  anime:
    'high-budget theatrical Japanese anime still, luminous painted clouds, crisp cel shading, expressive bloom',
  claymation:
    'tactile stop-motion claymation miniature set, handcrafted sculpted textures, practical warm studio lights',
  noir: 'classic 35mm neo-noir cinema still, dramatic chiaroscuro shadows, rain-slicked reflections, monochrome silver',
  pixar3d:
    'stylized 3D animated feature film frame, subsurface scattering, expressive stylized proportions, rich global illumination',
  watercolor:
    'hand-painted gouache and watercolor storybook illustration, soft paper grain, delicate ink accents',
};

export function getDimensionsForAspect(
  aspectRatio: AspectRatio
): { width: number; height: number } {
  if (aspectRatio === '9:16') return { width: 288, height: 512 };
  if (aspectRatio === '1:1') return { width: 512, height: 512 };
  return { width: 512, height: 288 };
}
