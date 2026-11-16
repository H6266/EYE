import { useEffect, useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Cpu,
  Eye,
  Film,
  Image as ImageIcon,
  Loader2,
  Mic,
  Plus,
  Square,
  Upload,
  UserCheck,
  Wand2,
  X,
} from 'lucide-react';
import {
  AspectRatio,
  CharacterProfile,
  ImageEngineId,
  MediaAsset,
  VisualStyle,
} from '../types/studio';
import {
  CAMERA_FRAMING_OPTIONS,
  IMAGE_ENGINES,
  LIGHTING_OPTIONS,
} from '../data/imageEngines';
import {
  buildRichVisualPrompt,
  sanitizeNegativePrompt,
} from '../utils/imagePromptBuilder';
import { QUALITY_PRESETS } from '../data/imageEngines';

interface GeneratePanelProps {
  visualAssets: MediaAsset[];
  characters: CharacterProfile[];
  selectedCharacterId: string;
  onSelectCharacter: (characterId: string) => void;
  onQuickUploadCharacter: (character: CharacterProfile) => void;
  onOpenCastTab: () => void;
  onAddAsset: (asset: MediaAsset, insertOnTimeline?: boolean) => void;
  onQuickCreateAudioAndSub?: (narrationLine: string, musicCue: string) => void;
}

type GenMode = 'image' | 'video' | 'restyle';

interface GeneratedTake {
  id: string;
  imageDataUrl: string;
  seed?: number;
  modelLabel: string;
  prompt: string;
  style: VisualStyle;
  aspectRatio: AspectRatio;
  engineId: ImageEngineId;
  timingsMs?: {
    tokenizeAndEncode: number;
    vaeEncode: number;
    unetDenoise: number;
    vaeDecode: number;
    total: number;
  };
  tokenStats?: {
    activeTokens: number;
    maxTokens: number;
    truncated: boolean;
  };
}

const STYLE_OPTIONS: Array<{ id: VisualStyle; label: string }> = [
  { id: 'cinematic', label: 'Cinematic 35mm' },
  { id: 'cartoon2d', label: '2D Cartoon' },
  { id: 'anime', label: 'Anime Cel' },
  { id: 'claymation', label: 'Stop-Motion Clay' },
  { id: 'pixar3d', label: '3D Animated' },
  { id: 'watercolor', label: 'Watercolor' },
  { id: 'noir', label: 'Film Noir' },
  { id: 'original', label: 'Natural Studio' },
  { id: 'bioluminescent', label: 'Bioluminescent Glow' },
  { id: 'cyberpunk', label: 'Cyberpunk Neon' },
  { id: 'macro', label: 'Macro Photography' },
  { id: 'steampunk', label: 'Steampunk Brass' },
  { id: 'art-nouveau', label: 'Art Nouveau' },
  { id: 'origami', label: 'Origami Paper Craft' },
  { id: 'gothic', label: 'Gothic Dark Fantasy' },
  { id: 'minimalist', label: 'Minimalist Zen' },
  { id: 'holographic', label: 'Holographic' },
  { id: 'infrared', label: 'Infrared Thermal' },
  { id: 'polaroid', label: 'Vintage Polaroid' },
  { id: 'isometric', label: 'Isometric Diorama' },
];

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = String(reader.result || '');
      const idx = result.indexOf(',');
      resolve(idx >= 0 ? result.slice(idx + 1) : result);
    };
    reader.onerror = () => reject(new Error('Failed to read media file'));
    reader.readAsDataURL(blob);
  });
}

/**
 * Converts any image URL/data URL into a 512x512 PNG base64 string for the local VAE Encoder.
 */
async function imageUrlToPng512Base64(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Canvas context unavailable'));
        return;
      }
      const scale = Math.max(512 / img.naturalWidth, 512 / img.naturalHeight);
      const w = img.naturalWidth * scale;
      const h = img.naturalHeight * scale;
      ctx.drawImage(img, (512 - w) / 2, (512 - h) / 2, w, h);
      const dataUrl = canvas.toDataURL('image/png');
      resolve(dataUrl.split(',')[1] || '');
    };
    img.onerror = () => reject(new Error('Failed to load reference image'));
    img.src = url;
  });
}

export function GeneratePanel({
  visualAssets,
  characters,
  selectedCharacterId,
  onSelectCharacter,
  onQuickUploadCharacter,
  onOpenCastTab,
  onAddAsset,
  onQuickCreateAudioAndSub,
}: GeneratePanelProps) {
  const [mode, setMode] = useState<GenMode>('image');
  const [engineId, setEngineId] = useState<ImageEngineId>('local-sdxs');
  const [prompt, setPrompt] = useState(
    'A glowing lantern airship drifting above misty pine valleys at dusk'
  );
  const [style, setStyle] = useState<VisualStyle>('cinematic');
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('16:9');
  const [qualityPreset, setQualityPreset] = useState('high');
  const [cameraFraming, setCameraFraming] = useState<string>('auto');
  const [lighting, setLighting] = useState<string>('auto');
  const [duration, setDuration] = useState(5);

  // Local Diffusion Controls (Sampler, CFG Guidance, VAE Img2Img Strength)
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [negativePrompt, setNegativePrompt] = useState(
    'blurry, distorted anatomy, extra limbs, watermark, low quality'
  );
  const [guidance, setGuidance] = useState(1.2);
  const [seed, setSeed] = useState(424242);
  const [randomizeSeed, setRandomizeSeed] = useState(true);
  const [img2imgStrength, setImg2imgStrength] = useState(0.78);
  const [useCharacterLatent, setUseCharacterLatent] = useState(false);
  const [batchCount, setBatchCount] = useState(1);

  const [referenceAssetId, setReferenceAssetId] = useState<string>(
    visualAssets[0]?.id || ''
  );

  // Generated gallery takes
  const [recentTakes, setRecentTakes] = useState<GeneratedTake[]>([]);

  // Audio prompt capture
  const [isRecording, setIsRecording] = useState(false);
  const [audioPromptData, setAudioPromptData] = useState<{
    base64: string;
    mimeType: string;
    name: string;
  } | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [suggestedNarration, setSuggestedNarration] = useState<string | null>(null);
  const [suggestedMusic, setSuggestedMusic] = useState<string | null>(null);

  // Media preview modal (image or video)
  const [preview, setPreview] = useState<{
    url: string;
    kind: 'image' | 'video';
    title: string;
  } | null>(null);

  const activeEngine =
    IMAGE_ENGINES.find((e) => e.id === engineId) || IMAGE_ENGINES[0];
  const activeCharacter =
    characters.find((c) => c.id === selectedCharacterId) || null;

  useEffect(() => {
    if (mode === 'restyle') {
      setEngineId('local-sdxs-img2img');
    } else if (engineId === 'local-sdxs-img2img' && !useCharacterLatent) {
      setEngineId('local-sdxs');
    }
  }, [mode, useCharacterLatent, engineId]);

  const toggleRecording = async () => {
    setErrorMsg(null);
    if (isRecording) {
      recorderRef.current?.stop();
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });
        const base64 = await blobToBase64(blob);
        setAudioPromptData({
          base64,
          mimeType: blob.type || 'audio/webm',
          name: 'Microphone recording',
        });
      };
      recorder.start();
      recorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setErrorMsg(
        'Microphone permission unavailable. Upload an audio prompt file instead.'
      );
    }
  };

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    const base64 = await blobToBase64(file);
    setAudioPromptData({
      base64,
      mimeType: file.type || 'audio/mp3',
      name: file.name,
    });
  };

  const handleInlineCharacterUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('Failed to load character image'));
      reader.readAsDataURL(file);
    });
    const newChar: CharacterProfile = {
      id: `char-${Date.now()}`,
      name: file.name.replace(/\.[^/.]+$/, '').slice(0, 22) || 'Uploaded Hero',
      imageUrl: dataUrl,
      traits: 'Character portrait reference uploaded for scene conditioning',
      createdAt: Date.now(),
    };
    onQuickUploadCharacter(newChar);
    setUseCharacterLatent(true);
    setEngineId('local-sdxs-img2img');
    setStatusMessage(
      `Loaded "${newChar.name}" into local VAE Encoder Img2Img mode.`
    );
    e.target.value = '';
  };

  const handleEnhancePrompt = async () => {
    if (!prompt.trim() && !audioPromptData) {
      setErrorMsg('Enter a text prompt or attach an audio prompt first.');
      return;
    }
    setErrorMsg(null);
    setStatusMessage(null);
    setIsEnhancing(true);
    try {
      const res = await fetch('/api/prompts/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          textPrompt: prompt,
          audioBase64: audioPromptData?.base64,
          audioMimeType: audioPromptData?.mimeType,
          style,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.error || 'Failed to enhance prompt.');
        return;
      }
      if (data.visualPrompt) {
        setPrompt(data.visualPrompt);
      }
      setSuggestedNarration(data.narrationLine || null);
      setSuggestedMusic(data.musicCue || null);
      setStatusMessage(
        data.transcribedAudio
          ? `Transcribed audio direction: "${data.transcribedAudio}"`
          : 'Director prompt refined for CLIP ViT-L/14 encoding.'
      );
    } catch (err) {
      setErrorMsg(
        err instanceof Error ? err.message : 'Network error enhancing prompt.'
      );
    } finally {
      setIsEnhancing(false);
    }
  };

  const addTakeToStudio = (take: GeneratedTake, insertOnTimeline: boolean) => {
    const asset: MediaAsset = {
      id: `img-${Date.now()}-${Math.floor(Math.random() * 999)}`,
      title: take.prompt.slice(0, 28) || 'Generated Frame',
      kind: 'image',
      url: take.imageDataUrl,
      duration: 4.5,
      styleTag: take.style,
      prompt: take.prompt,
      engineId: take.engineId,
      seed: take.seed,
      aspectRatio: take.aspectRatio,
      createdAt: Date.now(),
    };
    onAddAsset(asset, insertOnTimeline);
    setStatusMessage(
      insertOnTimeline
        ? `Rendered locally in ${((take.timingsMs?.total || 2500) / 1000).toFixed(1)}s and added to V1.`
        : `Saved local render to Media Bin.`
    );
  };

  const openPreview = (url: string, kind: 'image' | 'video', title: string) => {
    setPreview({ url, kind, title });
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() && !audioPromptData) {
      setErrorMsg('Enter a scene description or record an audio prompt.');
      return;
    }

    setErrorMsg(null);
    setStatusMessage(null);
    setIsGenerating(true);

    try {
      let activePrompt = prompt.trim();
      if (audioPromptData && !activePrompt) {
        const enhRes = await fetch('/api/prompts/enhance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            textPrompt: activePrompt,
            audioBase64: audioPromptData.base64,
            audioMimeType: audioPromptData.mimeType,
            style,
          }),
        });
        const enhData = await enhRes.json();
        if (!enhRes.ok) {
          setErrorMsg(enhData.error || 'Could not process audio prompt.');
          return;
        }
        activePrompt =
          enhData.visualPrompt || enhData.transcribedAudio || 'Cinematic scene';
        setPrompt(activePrompt);
      }

      if (mode === 'video') {
        const richPrompt = buildRichVisualPrompt({
          prompt: activePrompt,
          style,
          cameraFraming,
          lighting,
          characterName: activeCharacter?.name,
          characterTraits: activeCharacter?.traits,
        });
        let refBase64: string | undefined;
        if (activeCharacter?.imageUrl) {
          refBase64 = await imageUrlToPng512Base64(activeCharacter.imageUrl);
        }
        const res = await fetch('/api/media/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: richPrompt,
            style,
            aspectRatio,
            durationSeconds: duration,
            referenceImageBase64: refBase64,
            referenceImageMimeType: 'image/png',
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          setErrorMsg(data.error || 'Video generation failed.');
          return;
        }
        const asset: MediaAsset = {
          id: `vid-${Date.now()}`,
          title: activePrompt.slice(0, 28) || 'Generated Clip',
          kind: 'video',
          url: data.videoDataUrl,
          duration: data.durationSeconds || duration,
          styleTag: style,
          prompt: activePrompt,
          aspectRatio,
          interactionId: data.interactionId,
          createdAt: Date.now(),
        };
        onAddAsset(asset, true);
        setPreview({ url: data.videoDataUrl, kind: 'video', title: asset.title });
        setStatusMessage('Video clip generated and appended to V1 timeline.');
        return;
      }

      const composedPrompt = buildRichVisualPrompt({
        prompt: activePrompt,
        style,
        cameraFraming,
        lighting,
        characterName: activeCharacter?.name,
        characterTraits: activeCharacter?.traits,
      });

      const qualityPresetConfig = QUALITY_PRESETS.find(
        (q) => q.id === qualityPreset
      );
      const enhancedPrompt =
        composedPrompt +
        (qualityPresetConfig ? `, ${qualityPresetConfig.suffix}` : '');

      let refPngBase64: string | undefined;
      if (mode === 'restyle' && referenceAssetId) {
        const refAsset = visualAssets.find((a) => a.id === referenceAssetId);
        if (refAsset?.url) {
          refPngBase64 = await imageUrlToPng512Base64(refAsset.url);
        }
      } else if (
        engineId === 'local-sdxs-img2img' &&
        activeCharacter?.imageUrl
      ) {
        refPngBase64 = await imageUrlToPng512Base64(activeCharacter.imageUrl);
      }

      const res = await fetch('/api/media/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: enhancedPrompt,
          negativePrompt: sanitizeNegativePrompt(negativePrompt),
          engineId,
          style,
          aspectRatio,
          guidance,
          seed,
          randomizeSeed,
          img2imgStrength,
          batchCount,
          referenceImagePngBase64: refPngBase64,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.error || 'Local diffusion inference failed.');
        return;
      }

      const returnedTakes: Array<{
        imageDataUrl: string;
        seed?: number;
        modelLabel?: string;
        timingsMs?: GeneratedTake['timingsMs'];
        tokenStats?: GeneratedTake['tokenStats'];
      }> =
        Array.isArray(data.images) && data.images.length > 0
          ? data.images
          : [
              {
                imageDataUrl: data.imageDataUrl,
                seed: data.seed,
                modelLabel: data.model || activeEngine.name,
                timingsMs: data.timingsMs,
                tokenStats: data.tokenStats,
              },
            ];

      const formattedTakes: GeneratedTake[] = returnedTakes.map((t, idx) => ({
        id: `take-${Date.now()}-${idx}`,
        imageDataUrl: t.imageDataUrl,
        seed: t.seed,
        modelLabel: t.modelLabel || activeEngine.name,
        prompt: activePrompt,
        style,
        aspectRatio,
        engineId,
        timingsMs: t.timingsMs,
        tokenStats: t.tokenStats,
      }));

      setRecentTakes((prev) => [...formattedTakes, ...prev].slice(0, 8));
      addTakeToStudio(formattedTakes[0], true);
      if (formattedTakes.length > 1) {
        setStatusMessage(
          `Generated ${formattedTakes.length} takes locally via CLIP + UNet + VAE. Take 1 added to V1.`
        );
      }
    } catch (err) {
      setErrorMsg(
        err instanceof Error ? err.message : 'Generation request failed.'
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const latestTake = recentTakes[0];

  return (
    <form onSubmit={handleGenerate} className="flex flex-col gap-3.5 p-4 text-sm">
      <div className="grid grid-cols-3 gap-1 p-1 bg-[#0E1013] rounded-lg border border-white/8">
        {(
          [
            { id: 'image', label: 'Picture', Icon: ImageIcon },
            { id: 'restyle', label: 'Img2Img', Icon: Wand2 },
            { id: 'video', label: 'Video', Icon: Film },
          ] as const
        ).map(({ id, label, Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setMode(id)}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
              mode === id
                ? 'bg-[#222731] text-[#F1F3F5]'
                : 'text-[#8E96A4] hover:text-[#F1F3F5]'
            }`}
          >
            <Icon className="w-3.5 h-3.5 shrink-0" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Self-Contained Local Neural Diffusion Ingredients */}
      {mode !== 'video' && (
        <div className="flex flex-col gap-1.5 p-2.5 rounded-lg bg-[#0E1013] border border-white/8">
          <div className="flex items-center justify-between gap-2">
            <label
              htmlFor="t2i-engine"
              className="text-xs font-semibold text-[#F1F3F5] flex items-center gap-1.5"
            >
              <Cpu className="w-3.5 h-3.5 text-[#E5A93C]" />
              <span>Local Neural Pipeline</span>
            </label>
            <span className="text-[11px] text-[#8E96A4] truncate">
              {activeEngine.ingredientsSummary}
            </span>
          </div>
          <select
            id="t2i-engine"
            value={engineId}
            onChange={(e) => setEngineId(e.target.value as ImageEngineId)}
            className="w-full bg-[#14171D] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
          >
            {IMAGE_ENGINES.map((eng) => (
              <option key={eng.id} value={eng.id}>
                {eng.name} · {eng.architecture}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-[#7D8594] leading-snug">
            {activeEngine.summary}
          </p>
        </div>
      )}

      {/* Character Consistency Picker */}
      <div className="flex flex-col gap-1.5 p-2.5 rounded-lg bg-[#0E1013] border border-white/8">
        <div className="flex items-center justify-between gap-2">
          <label
            htmlFor="char-ref-select"
            className="text-xs font-medium text-[#9BA1AE] flex items-center gap-1.5"
          >
            <UserCheck className="w-3.5 h-3.5 text-[#E5A93C]" />
            <span>Character Lock</span>
          </label>
          <div className="flex items-center gap-2">
            <label className="text-[11px] text-[#E5A93C] hover:underline cursor-pointer whitespace-nowrap">
              Upload Photo
              <input
                type="file"
                accept="image/*"
                onChange={handleInlineCharacterUpload}
                className="sr-only"
              />
            </label>
            <button
              type="button"
              onClick={onOpenCastTab}
              className="text-[11px] text-[#8E96A4] hover:text-[#F1F3F5] whitespace-nowrap"
            >
              Manage Cast
            </button>
          </div>
        </div>

        <select
          id="char-ref-select"
          value={selectedCharacterId}
          onChange={(e) => onSelectCharacter(e.target.value)}
          className="w-full bg-[#14171D] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
        >
          <option value="">None (Environment / Free Cast)</option>
          {characters.map((ch) => (
            <option key={ch.id} value={ch.id}>
              {ch.name}
            </option>
          ))}
        </select>

        {activeCharacter && mode === 'image' && (
          <label className="flex items-center gap-2 pt-1 text-[11px] text-[#C4C9D4] cursor-pointer">
            <input
              type="checkbox"
              checked={useCharacterLatent}
              onChange={(e) => {
                setUseCharacterLatent(e.target.checked);
                setEngineId(
                  e.target.checked ? 'local-sdxs-img2img' : 'local-sdxs'
                );
              }}
              className="accent-[#E5A93C]"
            />
            <span>Encode character photo through local VAE Encoder</span>
          </label>
        )}
      </div>

      {mode === 'restyle' && (
        <div className="flex flex-col gap-1.5">
          <label
            htmlFor="restyle-source"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Source Frame for VAE Encoder
          </label>
          <select
            id="restyle-source"
            value={referenceAssetId}
            onChange={(e) => setReferenceAssetId(e.target.value)}
            className="w-full bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-2 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
          >
            {visualAssets.map((a) => (
              <option key={a.id} value={a.id}>
                {a.title}
              </option>
            ))}
          </select>
        </div>
      )}

      {engineId === 'local-sdxs-img2img' && mode !== 'video' && (
        <div className="flex flex-col gap-1 px-2.5 py-2 rounded-md bg-[#0E1013] border border-white/8">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[#9BA1AE]">Latent Transform Strength</span>
            <span className="font-mono text-[#F1F3F5]">
              {Math.round(img2imgStrength * 100)}%
            </span>
          </div>
          <input
            type="range"
            aria-label="Img2Img latent strength"
            min={0.35}
            max={0.95}
            step={0.05}
            value={img2imgStrength}
            onChange={(e) => setImg2imgStrength(Number(e.target.value))}
            className="accent-[#E5A93C]"
          />
        </div>
      )}

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <label
            htmlFor="scene-prompt"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Text or Audio Prompt
          </label>
          <button
            type="button"
            onClick={handleEnhancePrompt}
            disabled={isEnhancing}
            className="text-xs font-medium text-[#E5A93C] hover:text-[#F6C86E] disabled:opacity-50 whitespace-nowrap"
          >
            {isEnhancing ? 'Directing...' : 'Direct & Enhance'}
          </button>
        </div>
        <textarea
          id="scene-prompt"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe any scene, character, cartoon, object, lighting, or style..."
          className="w-full bg-[#0E1013] border border-white/10 rounded-md px-3 py-2 text-xs leading-relaxed text-[#F1F3F5] placeholder:text-[#5F6674] focus:outline-none focus:border-[#E5A93C] resize-none"
        />
        <div className="flex items-center gap-2 pt-0.5">
          <button
            type="button"
            onClick={toggleRecording}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium border transition-colors whitespace-nowrap ${
              isRecording
                ? 'bg-[#DC2626]/20 border-[#DC2626] text-[#FCA5A5]'
                : 'bg-[#0E1013] border-white/10 text-[#C4C9D4] hover:border-white/25'
            }`}
          >
            {isRecording ? (
              <>
                <Square className="w-3 h-3 fill-current" />
                <span>Stop Mic</span>
              </>
            ) : (
              <>
                <Mic className="w-3.5 h-3.5" />
                <span>Record Audio</span>
              </>
            )}
          </button>

          <label className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium bg-[#0E1013] border border-white/10 text-[#C4C9D4] hover:border-white/25 cursor-pointer whitespace-nowrap">
            <Upload className="w-3.5 h-3.5" />
            <span>Audio File</span>
            <input
              type="file"
              accept="audio/*"
              onChange={handleAudioUpload}
              className="sr-only"
            />
          </label>
        </div>
        {audioPromptData && (
          <div className="flex items-center justify-between text-xs text-[#E5A93C] bg-[#E5A93C]/10 border border-[#E5A93C]/25 rounded px-2.5 py-1.5">
            <span className="truncate">{audioPromptData.name} attached</span>
            <button
              type="button"
              onClick={() => setAudioPromptData(null)}
              className="text-[#9BA1AE] hover:text-white ml-2 shrink-0"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-[#9BA1AE]">
          Visual Style Preset
        </span>
        <div className="grid grid-cols-2 gap-1.5">
          {STYLE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => setStyle(opt.id)}
              className={`px-2.5 py-1.5 rounded-md text-xs font-medium text-left border transition-colors truncate ${
                style === opt.id
                  ? 'bg-[#E5A93C]/15 border-[#E5A93C] text-[#FDE68A]'
                  : 'bg-[#0E1013] border-white/8 text-[#9BA1AE] hover:text-[#F1F3F5]'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-1">
          <label
            htmlFor="gen-framing"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Camera Framing
          </label>
          <select
            id="gen-framing"
            value={cameraFraming}
            onChange={(e) => setCameraFraming(e.target.value)}
            className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5]"
          >
            {CAMERA_FRAMING_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label
            htmlFor="gen-lighting"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Lighting Setup
          </label>
          <select
            id="gen-lighting"
            value={lighting}
            onChange={(e) => setLighting(e.target.value)}
            className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5]"
          >
            {LIGHTING_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-1">
          <label
            htmlFor="gen-aspect"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Aspect Ratio
          </label>
          <select
            id="gen-aspect"
            value={aspectRatio}
            onChange={(e) => setAspectRatio(e.target.value as AspectRatio)}
            className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
          >
            <option value="16:9">16:9 Widescreen</option>
            <option value="9:16">9:16 Vertical</option>
            <option value="1:1">1:1 Square (Full 512px)</option>
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label
            htmlFor="gen-quality"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Image Quality
          </label>
          <select
            id="gen-quality"
            value={qualityPreset}
            onChange={(e) => setQualityPreset(e.target.value)}
            className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
          >
            {QUALITY_PRESETS.map((q) => (
              <option key={q.id} value={q.id}>
                {q.label}
              </option>
            ))}
          </select>
        </div>

        {mode === 'video' ? (
          <div className="flex flex-col gap-1">
            <label
              htmlFor="gen-duration"
              className="text-xs font-medium text-[#9BA1AE]"
            >
              Clip Duration
            </label>
            <select
              id="gen-duration"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5] font-mono focus:outline-none focus:border-[#E5A93C]"
            >
              <option value={4}>4s</option>
              <option value={6}>6s</option>
              <option value={10}>10s</option>
              <option value={15}>15s</option>
            </select>
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            <label
              htmlFor="gen-batch"
              className="text-xs font-medium text-[#9BA1AE]"
            >
              Variations
            </label>
            <select
              id="gen-batch"
              value={batchCount}
              onChange={(e) => setBatchCount(Number(e.target.value))}
              className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs font-mono text-[#F1F3F5]"
            >
              <option value={1}>1 Take (~2.5s)</option>
              <option value={2}>2 Takes (~5s)</option>
              <option value={4}>4 Takes Grid (~10s)</option>
            </select>
          </div>
        )}
      </div>

      {/* Sampler + CFG Guidance + Negative Prompt + Seed */}
      {mode !== 'video' && (
        <div className="rounded-lg border border-white/8 bg-[#0E1013]">
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="w-full px-3 py-2 flex items-center justify-between text-xs font-medium text-[#C4C9D4] hover:text-white"
          >
            <span>Sampler, CFG Guidance & Seed</span>
            {showAdvanced ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>

          {showAdvanced && (
            <div className="px-3 pb-3 pt-1 flex flex-col gap-2.5 border-t border-white/6">
              {activeEngine.supportsNegativePrompt && (
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="gen-neg"
                    className="text-[11px] font-medium text-[#9BA1AE]"
                  >
                    Negative Prompt (Unconditional Repulsion)
                  </label>
                  <input
                    id="gen-neg"
                    type="text"
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="blurry, distorted hands, watermark, text"
                    className="bg-[#14171D] border border-white/10 rounded px-2 py-1.5 text-xs text-[#F1F3F5] placeholder:text-[#5F6674]"
                  />
                </div>
              )}

              {activeEngine.supportsGuidance && (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-[#9BA1AE]">CFG Guidance Scale</span>
                    <span className="font-mono text-[#F1F3F5]">{guidance}</span>
                  </div>
                  <input
                    type="range"
                    aria-label="Guidance scale"
                    min={1.05}
                    max={1.35}
                    step={0.05}
                    value={guidance}
                    onChange={(e) => setGuidance(Number(e.target.value))}
                    className="accent-[#E5A93C]"
                  />
                </div>
              )}

              <div className="flex items-center gap-2">
                <input
                  type="number"
                  aria-label="Seed"
                  disabled={randomizeSeed}
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value) || 0)}
                  className="flex-1 bg-[#14171D] border border-white/10 rounded px-2 py-1 text-xs font-mono text-[#F1F3F5] disabled:opacity-45"
                />
                <label className="flex items-center gap-1.5 text-[11px] text-[#C4C9D4] cursor-pointer whitespace-nowrap">
                  <input
                    type="checkbox"
                    checked={randomizeSeed}
                    onChange={(e) => setRandomizeSeed(e.target.checked)}
                    className="accent-[#E5A93C]"
                  />
                  <span>Random Seed</span>
                </label>
              </div>
            </div>
          )}
        </div>
      )}

      {suggestedNarration && (
        <div className="p-2.5 rounded-md bg-[#0E1013] border border-white/10 text-xs flex flex-col gap-1.5">
          <p className="text-[#9BA1AE]">
            Suggested Line:{' '}
            <span className="text-[#F1F3F5]">{suggestedNarration}</span>
          </p>
          {onQuickCreateAudioAndSub && (
            <button
              type="button"
              onClick={() =>
                onQuickCreateAudioAndSub(
                  suggestedNarration,
                  suggestedMusic || ''
                )
              }
              className="text-left text-xs font-medium text-[#E5A93C] hover:underline"
            >
              Send line to Voice Actor & Music Studio
            </button>
          )}
        </div>
      )}

      {errorMsg && (
        <div
          role="alert"
          className="p-2.5 rounded-md bg-[#DC2626]/12 border border-[#DC2626]/40 text-xs text-[#FCA5A5] leading-relaxed"
        >
          {errorMsg}
        </div>
      )}

      {statusMessage && (
        <div className="p-2.5 rounded-md bg-[#16A34A]/12 border border-[#16A34A]/40 text-xs text-[#86EFAC] leading-relaxed">
          {statusMessage}
        </div>
      )}

      {preview?.kind === 'video' && (
        <button
          type="button"
          onClick={() => openPreview(preview.url, 'video', preview.title)}
          className="w-full py-2 px-4 rounded-md text-xs font-medium border border-white/10 text-[#C4C9D4] hover:text-white hover:border-white/25 mb-2 flex items-center justify-center gap-1.5"
        >
          <Eye className="w-3.5 h-3.5 shrink-0" />
          <span>Preview Video</span>
        </button>
      )}

      <button
        type="submit"
        disabled={isGenerating}
        className="w-full py-2.5 px-4 rounded-md bg-[#E5A93C] text-[#111317] text-xs font-semibold hover:bg-[#EEB955] disabled:opacity-60 transition-colors flex items-center justify-center gap-2"
      >
        {isGenerating ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>
              {mode === 'video'
                ? 'Rendering Video Clip...'
                : 'Denoising Locally (CLIP + UNet + VAE)...'}
            </span>
          </>
        ) : (
          <span>
            {mode === 'video'
              ? 'Generate Video Clip'
              : mode === 'restyle'
                ? 'Restyle Locally via VAE Img2Img'
                : `Generate ${batchCount > 1 ? `${batchCount} Takes` : 'Picture'} Locally`}
          </span>
        )}
      </button>

      {/* Ingredient Readout for latest local render */}
      {latestTake?.timingsMs && latestTake?.tokenStats && (
        <div className="px-2.5 py-2 rounded-md bg-[#0E1013] border border-white/8 text-[11px] text-[#9BA1AE] font-mono flex flex-col gap-0.5">
          <div className="flex items-center justify-between">
            <span>CLIP Tokens</span>
            <span className="text-[#F1F3F5]">
              {latestTake.tokenStats.activeTokens} / {latestTake.tokenStats.maxTokens}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Text Enc · UNet · VAE</span>
            <span className="text-[#F1F3F5]">
              {latestTake.timingsMs.tokenizeAndEncode}ms ·{' '}
              {latestTake.timingsMs.unetDenoise}ms · {latestTake.timingsMs.vaeDecode}ms
            </span>
          </div>
        </div>
      )}
      {/* Generated Takes Gallery */}
      {recentTakes.length > 0 && (
        <div className="pt-2 border-t border-white/8 flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-[#F1F3F5]">Recent Takes</span>
            <span className="font-mono text-[11px] text-[#8E96A4]">
              {recentTakes.length} frames
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {recentTakes.map((take) => (
              <div
                key={take.id}
                className="rounded-md overflow-hidden bg-[#0E1013] border border-white/10 flex flex-col"
              >
                <img
                  src={take.imageDataUrl}
                  alt={take.prompt}
                  className="w-full aspect-video object-cover"
                />
                <div className="p-1.5 flex items-center justify-between gap-1 text-[10px] text-[#8E96A4]">
                  <span className="truncate font-mono">
                    {take.seed ? `#${take.seed}` : take.modelLabel}
                  </span>
                  <div className="flex items-center gap-0.5 shrink-0">
                    <button
                      type="button"
                      title="Preview take"
                      onClick={() =>
                        openPreview(take.imageDataUrl, 'image', take.prompt)
                      }
                      className="px-1.5 py-0.5 rounded bg-[#1D2129] text-[#E5A93C] hover:bg-[#E5A93C] hover:text-[#111317] flex items-center gap-0.5 shrink-0"
                    >
                      <Eye className="w-2.5 h-2.5" />
                      <span>View</span>
                    </button>
                    <button
                      type="button"
                      title="Add take to V1 timeline"
                      onClick={() => addTakeToStudio(take, true)}
                      className="px-1.5 py-0.5 rounded bg-[#1D2129] text-[#E5A93C] hover:bg-[#E5A93C] hover:text-[#111317] flex items-center gap-0.5 shrink-0"
                    >
                      <Plus className="w-2.5 h-2.5" />
                      <span>V1</span>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Full-screen media preview modal */}
      {preview && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setPreview(null)}
        >
          <div
            className="bg-[#0E1013] border border-white/20 rounded-lg shadow-2xl max-w-2xl max-h-[90vh] flex flex-col items-center relative"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setPreview(null)}
              className="absolute top-2 right-2 text-[#8E96A4] hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
            <div className="text-xs text-[#8E96A4] mb-2 truncate max-w-[300px]">
              {preview.title}
            </div>
            {preview.kind === 'image' ? (
              <img
                src={preview.url}
                alt={preview.title}
                className="max-w-full max-h-[60vh] object-contain rounded-md border border-white/10"
              />
            ) : (
              <video
                src={preview.url}
                controls
                autoPlay
                className="max-w-full max-h-[60vh] rounded-md border border-white/10 bg-black"
              />
            )}
            <button
              type="button"
              onClick={() => setPreview(null)}
              className="mt-3 px-3 py-1.5 rounded-md text-xs font-medium border border-white/10 text-[#C4C9D4] hover:text-white hover:border-white/25"
            >
              Close
            </button>
          </div>
        </div>
      )}

    </form>
  );
}
