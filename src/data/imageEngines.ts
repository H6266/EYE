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
  {
    id: 'python-zimage-turbo',
    name: 'Z-Image-Turbo (Python GPU)',
    architecture: 'ZImagePipeline · Qwen3-4B Text Encoder · DiT',
    ingredientsSummary: 'Apache-2.0 · 6B params · GPU required',
    summary:
      '6B DiT model with 8-step generation at 1024×1024. Zero CFG, zero negative prompts. Primary production model for fast, high-fidelity image generation. Requires GPU (Kaggle T4/P100 or production).',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: false,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
  },
  {
    id: 'python-flux-klein',
    name: 'FLUX.2-klein-4B (Python GPU)',
    architecture: 'Flux2KleinPipeline · Qwen3-4B Text Encoder · Flux DiT',
    ingredientsSummary: 'Apache-2.0 · 4B params · GPU required',
    summary:
      '4B Flux model with 4-step generation at up to 4MP resolution. CFG and negative prompts supported. Fast fallback for production generation. Requires GPU.',
    supportsNegativePrompt: true,
    supportsCharacterImg2Img: false,
    supportsGuidance: true,
    defaultGuidance: 1.0,
    maxGuidance: 4.0,
  },
  {
    id: 'python-zimage-base',
    name: 'Z-Image Base (Python GPU)',
    architecture: 'ZImagePipeline · Qwen3-4B Text Encoder · DiT',
    ingredientsSummary: 'Apache-2.0 · Base model · GPU required',
    summary:
      'Base Z-Image model with 50-step generation, CFG, and negative prompts at 1024×1024. Higher quality than Z-Image-Turbo but slower. Requires GPU.',
    supportsNegativePrompt: true,
    supportsCharacterImg2Img: false,
    supportsGuidance: true,
    defaultGuidance: 5.0,
    maxGuidance: 10.0,
  },
  {
    id: 'python-qwen-2512',
    name: 'Qwen-Image-2512 (Python GPU)',
    architecture: 'QwenImagePipeline · Qwen2.5-VL-7B Text Encoder · MMDiT',
    ingredientsSummary: 'Apache-2.0 · 20B params · 24-48GB VRAM required',
    summary:
      '20B MMDiT model with best-in-class fidelity. Requires 24-48GB VRAM production GPU. Does NOT fit Kaggle 16GB T4/P100. Disabled by default.',
    supportsNegativePrompt: true,
    supportsCharacterImg2Img: false,
    supportsGuidance: true,
    defaultGuidance: 4.0,
    maxGuidance: 10.0,
  },
  {
    id: 'python-moss-sft',
    name: 'MOSS-Image-SFT (Python GPU)',
    architecture: 'DiffusionPipeline + pipeline_moss_image · DiT',
    ingredientsSummary: 'Apache-2.0 · 5B params · GPU required',
    summary:
      '5B model optimized for text rendering and layout control. 50-step generation with CFG at 1024×1024. VAE in FP32. Requires GPU.',
    supportsNegativePrompt: true,
    supportsCharacterImg2Img: false,
    supportsGuidance: true,
    defaultGuidance: 3.0,
    maxGuidance: 10.0,
  },
  {
    id: 'remote-zimage-turbo',
    name: 'Z-Image-Turbo (Remote)',
    architecture: 'Remote ZImagePipeline · Tongyi-MAI/Z-Image-Turbo',
    ingredientsSummary: 'Remote inference · HF Endpoint or RunPod',
    summary:
      'Runs on a configured Hugging Face Inference Endpoint or RunPod worker. No model weights are loaded in Codespaces.',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: false,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
  },
  {
    id: 'remote-flux-klein',
    name: 'FLUX.2 Klein 4B (Remote)',
    architecture: 'Remote Flux2KleinPipeline · black-forest-labs/FLUX.2-klein-4B',
    ingredientsSummary: 'Remote inference · HF Endpoint or RunPod',
    summary:
      'Runs on configured remote infrastructure. No FLUX weights are loaded in Codespaces.',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: false,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
  },
  {
    id: 'remote-flux-klein-edit',
    name: 'FLUX.2 Klein Edit (Remote)',
    architecture: 'Remote image edit · black-forest-labs/FLUX.2-klein-4B',
    ingredientsSummary: 'Remote image edit · HF Endpoint or RunPod',
    summary:
      'Edits a reference frame on the configured remote worker. The source image is sent to that provider only when you submit.',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: true,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
  },
  {
    id: 'remote-qwen-2512',
    name: 'Qwen-Image-2512 (Remote)',
    architecture: 'Remote QwenImagePipeline · Qwen/Qwen-Image-2512',
    ingredientsSummary: 'Remote inference · HF Endpoint or RunPod',
    summary:
      'Runs on configured remote infrastructure with production GPU memory. No Qwen weights are loaded in Codespaces.',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: false,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
  },
  {
    id: 'remote-qwen-2512-lightning',
    name: 'Qwen-Image-2512 Lightning (Remote)',
    architecture: 'Remote Qwen base + Lightning LoRA',
    ingredientsSummary: 'Remote inference · base model required',
    summary:
      'Uses the Lightning adapter with Qwen-Image-2512 loaded on the remote worker. No weights are loaded in Codespaces.',
    supportsNegativePrompt: false,
    supportsCharacterImg2Img: false,
    supportsGuidance: false,
    defaultGuidance: 0,
    maxGuidance: 0,
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
  bioluminescent:
    'bioluminescent organisms glowing in deep darkness, ethereal neon flora, soft internal light emission, iridescent hues, magical atmospheric glow, photorealistic macro detail',
  cyberpunk:
    'cyberpunk neo-tokyo street at night, neon signage, rainy wet pavement reflections, chrome and chrome accents, synthwave color palette, high-contrast futuristic cityscape, 8k photorealistic',
  macro:
    'extreme macro photography, razor-sharp fine detail, microscopic texture, shallow depth of field, vivid natural colors, scientific illustration precision, 8k UHD',
  steampunk:
    'steampunk Victorian industrial aesthetic, brass fittings, copper pipes, clockwork gears, leather and steam, warm sepia tones, intricate mechanical detail, photorealistic',
  'art-nouveau':
    'art nouveau decorative style, flowing organic lines, elegant stylized figures, ornamental botanical motifs, soft pastel color palette, intricate line work, vintage poster art',
  origami:
    'origami paper craft art, folded paper geometry, crisp creases, layered papercraft diorama, soft studio lighting, pastel colors, miniature paper world, photorealistic',
  gothic:
    'gothic dark fantasy, ornate cathedral architecture, gargoyles, moonlight through stained glass, dark moody atmosphere, deep shadows, dramatic fantasy illustration, high detail',
  minimalist:
    'minimalist zen composition, abundant negative space, clean geometric shapes, muted earth tones, serene simplicity, balanced asymmetry, high-end editorial photography',
  holographic:
    'holographic iridescent surface, shimmering rainbow light refraction, futuristic chrome, pearlescent gradients, sci-fi interface glow, crystalline detail, 8k photorealistic',
  infrared:
    'infrared thermal photography, false-color heatmap, surreal dreamlike landscape, glowing foliage, deep sky contrast, scientific imaging aesthetic, ethereal atmosphere',
  polaroid:
    'vintage polaroid instant film, warm faded colors, soft vignette, slight film grain, nostalgic 1970s aesthetic, candid snapshot, soft focus edges',
  isometric:
    'isometric miniature diorama, tiny world perspective, clean 3D vector style, bright flat lighting, toy-like proportions, architectural model, crisp edges, vibrant colors',
};

export const QUALITY_PRESETS: Array<{ id: string; label: string; suffix: string }> = [
  { id: 'standard', label: 'Standard', suffix: 'photorealistic, sharp focus, detailed' },
  {
    id: 'high',
    label: 'High Detail',
    suffix: 'ultra-detailed, 8k resolution, masterpiece, highly intricate, professional photography, crisp detail',
  },
  {
    id: 'ultra',
    label: 'Ultra Quality',
    suffix: 'ultra-detailed, 8k UHD, masterpiece, award-winning photography, insanely intricate detail, sharp focus, cinematic lighting, photorealistic, trending on artstation',
  },
] as const;

export function getDimensionsForAspect(
  aspectRatio: AspectRatio
): { width: number; height: number } {
  if (aspectRatio === '9:16') return { width: 288, height: 512 };
  if (aspectRatio === '1:1') return { width: 512, height: 512 };
  return { width: 512, height: 288 };
}
