export type TrackId = 'video' | 'overlay' | 'voice' | 'music' | 'subtitle';

export type VisualStyle =
  | 'original'
  | 'cinematic'
  | 'cartoon2d'
  | 'anime'
  | 'claymation'
  | 'noir'
  | 'pixar3d'
  | 'watercolor'
  | 'bioluminescent'
  | 'cyberpunk'
  | 'macro'
  | 'steampunk'
  | 'art-nouveau'
  | 'origami'
  | 'gothic'
  | 'minimalist'
  | 'holographic'
  | 'infrared'
  | 'polaroid'
  | 'isometric';

export type TransitionType = 'cut' | 'crossfade' | 'wipe';

export type AspectRatio = '16:9' | '9:16' | '1:1';

export type SubtitleStyle = 'cinematic' | 'cartoon' | 'minimal';

export type MediaKind = 'image' | 'video' | 'voice' | 'music';

export type ImageEngineId =
  | 'local-sdxs'
  | 'local-sdxs-cfg'
  | 'local-sdxs-img2img'
  | 'python-zimage-turbo'
  | 'python-flux-klein'
  | 'python-zimage-base'
  | 'python-qwen-2512'
  | 'python-moss-sft';

export const PYTHON_ENGINE_IDS: ImageEngineId[] = [
  'python-zimage-turbo',
  'python-flux-klein',
  'python-zimage-base',
  'python-qwen-2512',
  'python-moss-sft',
];

export interface CharacterProfile {
  id: string;
  name: string;
  imageUrl: string;
  traits: string;
  createdAt: number;
}

export interface MediaAsset {
  id: string;
  title: string;
  kind: MediaKind;
  url: string;
  duration: number;
  styleTag: VisualStyle | 'audio';
  prompt?: string;
  engineId?: ImageEngineId | string;
  seed?: number;
  aspectRatio?: AspectRatio;
  transcript?: string;
  lyrics?: string;
  interactionId?: string;
  createdAt: number;
}

export interface TimelineClip {
  id: string;
  assetId?: string;
  track: TrackId;
  title: string;
  startTime: number;
  duration: number;
  mediaUrl?: string;
  mediaKind?: MediaKind;
  visualStyle: VisualStyle;
  transition: TransitionType;
  transitionDuration: number;
  volume: number;
  fadeIn: number;
  fadeOut: number;
  subtitleText?: string;
  subtitleSpeaker?: string;
  subtitleStyle?: SubtitleStyle;
  kenBurns?: 'zoom-in' | 'zoom-out' | 'pan-right' | 'static';
  overlayPosition?: 'left' | 'center' | 'right';
  overlayScale?: number;
  overlayFrame?: 'full' | 'portrait' | 'badge';
}

export interface TrackConfig {
  id: TrackId;
  label: string;
  shortCode: string;
  muted: boolean;
  locked: boolean;
}

export type WorkspaceView = 'studio' | 'voice' | 'subtitles' | 'export';

export type LeftTab = 'generate' | 'characters' | 'media' | 'voice' | 'music' | 'subtitles';
