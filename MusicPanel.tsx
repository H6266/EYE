import { useEffect, useState } from 'react';
import { Loader2, Music, Plus, Upload } from 'lucide-react';
import { MediaAsset } from '../types/studio';

interface MusicPanelProps {
  initialCue?: string;
  visualAssets: MediaAsset[];
  builtInStems: MediaAsset[];
  onAddMusicAsset: (asset: MediaAsset, insertOnTimeline?: boolean) => void;
}

const MOOD_PRESETS = [
  {
    label: 'Cinematic Brass & Strings',
    prompt:
      'Sweeping cinematic orchestral score with warm French horns, soaring strings, and gentle clockwork chimes, 92 BPM',
  },
  {
    label: 'Playful Cartoon Adventure',
    prompt:
      'Bouncy playful 2D cartoon soundtrack with pizzicato strings, bright clarinet, marimba, and cheerful brass stabs, 124 BPM',
  },
  {
    label: 'Anime Twilight Lo-Fi',
    prompt:
      'Nostalgic Japanese anime piano and warm synth pad with gentle rain texture and soft acoustic bass, 78 BPM',
  },
  {
    label: 'Noir Midnight Jazz',
    prompt:
      'Smoky neo-noir upright bass, muted trumpet solo, and brushed snare drums in a midnight detective alley, 72 BPM',
  },
];

async function imageUrlToBase64(
  url: string
): Promise<{ base64: string; mimeType: string }> {
  const res = await fetch(url);
  const blob = await res.blob();
  const base64 = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const s = String(reader.result || '');
      resolve(s.includes(',') ? s.split(',')[1] : s);
    };
    reader.onerror = () => reject(new Error('Failed to encode image'));
    reader.readAsDataURL(blob);
  });
  return { base64, mimeType: blob.type || 'image/jpeg' };
}

function measureAudioDuration(url: string): Promise<number> {
  return new Promise((resolve) => {
    const audio = new Audio();
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => {
      const dur =
        Number.isFinite(audio.duration) && audio.duration > 0
          ? Math.min(180, Number(audio.duration.toFixed(1)))
          : 18;
      resolve(dur);
    };
    audio.onerror = () => resolve(18);
    audio.src = url;
  });
}

export function MusicPanel({
  initialCue,
  visualAssets,
  builtInStems,
  onAddMusicAsset,
}: MusicPanelProps) {
  const [prompt, setPrompt] = useState(MOOD_PRESETS[0].prompt);
  const [proMode, setProMode] = useState(false);
  const [inspireAssetId, setInspireAssetId] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [generatedTrack, setGeneratedTrack] = useState<{
    url: string;
    lyrics: string;
  } | null>(null);

  useEffect(() => {
    if (initialCue) setPrompt(initialCue);
  }, [initialCue]);

  const handleUploadMusicFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    setUploadMsg(null);
    const url = URL.createObjectURL(file);
    const detectedDuration = await measureAudioDuration(url);
    const cleanTitle =
      file.name.replace(/\.[^/.]+$/, '').slice(0, 28) || 'Custom Music Track';

    onAddMusicAsset(
      {
        id: `music-upload-${Date.now()}`,
        title: cleanTitle,
        kind: 'music',
        url,
        duration: detectedDuration,
        styleTag: 'audio',
        createdAt: Date.now(),
      },
      true
    );
    setUploadMsg(
      `Imported "${cleanTitle}" (${detectedDuration.toFixed(1)}s) and placed on A2.`
    );
    e.target.value = '';
  };

  const handleGenerateMusic = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setErrorMsg('Enter a music style, instrumentation, or mood.');
      return;
    }

    setErrorMsg(null);
    setUploadMsg(null);
    setIsGenerating(true);

    try {
      let refImg: { base64: string; mimeType: string } | null = null;
      if (inspireAssetId) {
        const found = visualAssets.find((a) => a.id === inspireAssetId);
        if (found?.url) {
          refImg = await imageUrlToBase64(found.url);
        }
      }

      const res = await fetch('/api/media/music', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          proMode,
          referenceImageBase64: refImg?.base64,
          referenceImageMimeType: refImg?.mimeType,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.error || 'Music generation failed.');
        return;
      }

      setGeneratedTrack({ url: data.audioDataUrl, lyrics: data.lyrics || '' });
      onAddMusicAsset(
        {
          id: `music-${Date.now()}`,
          title: prompt.slice(0, 26),
          kind: 'music',
          url: data.audioDataUrl,
          duration: proMode ? 45 : 24,
          styleTag: 'audio',
          lyrics: data.lyrics,
          createdAt: Date.now(),
        },
        true
      );
    } catch (err) {
      setErrorMsg(
        err instanceof Error ? err.message : 'Network error generating music.'
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4 text-sm">
      {/* Custom User Music Upload */}
      <div className="p-3 rounded-lg bg-[#0E1013] border border-white/8 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[#F1F3F5]">
            Upload Your Own Music
          </span>
          <span className="text-[11px] text-[#8E96A4]">MP3 · WAV · OGG</span>
        </div>
        <label className="w-full py-2 px-3 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#E5ECF6] hover:border-[#E5A93C] cursor-pointer flex items-center justify-center gap-2 transition-colors">
          <Upload className="w-3.5 h-3.5 text-[#E5A93C]" />
          <span>Import Music Track to A2</span>
          <input
            type="file"
            accept="audio/*"
            onChange={handleUploadMusicFile}
            className="sr-only"
          />
        </label>
        {uploadMsg && (
          <p className="text-[11px] text-[#86EFAC] leading-snug">{uploadMsg}</p>
        )}
      </div>

      <div className="pt-1 flex flex-col gap-2">
        <h3 className="text-xs font-semibold text-[#F1F3F5]">
          Local Studio Score Stems (Zero API)
        </h3>
        <div className="flex flex-col gap-2">
          {builtInStems.map((stem) => (
            <div
              key={stem.id}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(
                  'application/json',
                  JSON.stringify({ type: 'asset', assetId: stem.id })
                );
              }}
              className="flex items-center justify-between gap-2 p-2 rounded-md bg-[#12151B] border border-white/8 hover:border-[#34D399]/50 cursor-grab"
            >
              <div className="min-w-0">
                <div className="text-xs font-medium text-[#F1F3F5] truncate">
                  {stem.title}
                </div>
                <div className="text-[11px] text-[#8E96A4] font-mono">
                  {stem.duration.toFixed(1)}s · Local WAV Stem
                </div>
              </div>
              <button
                type="button"
                aria-label={`Add ${stem.title} to music track`}
                onClick={() => onAddMusicAsset(stem, true)}
                className="p-1.5 rounded bg-[#1D2129] text-[#34D399] hover:bg-[#34D399] hover:text-[#111317] transition-colors shrink-0"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </div>

      <form
        onSubmit={handleGenerateMusic}
        className="pt-3 border-t border-white/8 flex flex-col gap-3"
      >
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[#9BA1AE]">Score Presets</span>
          <div className="grid grid-cols-2 gap-1.5">
            {MOOD_PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => setPrompt(preset.prompt)}
                className="px-2 py-1.5 rounded bg-[#0E1013] border border-white/8 text-left text-[11px] text-[#C4C9D4] hover:border-[#E5A93C]/50 truncate"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <label
            htmlFor="music-prompt"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Optional Lyria Soundtrack Prompt
          </label>
          <textarea
            id="music-prompt"
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full bg-[#0E1013] border border-white/10 rounded-md px-3 py-2 text-xs leading-relaxed text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C] resize-none"
          />
        </div>

        <div className="flex flex-col gap-1">
          <label
            htmlFor="music-frame"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Optional Scene Inspiration Frame
          </label>
          <select
            id="music-frame"
            value={inspireAssetId}
            onChange={(e) => setInspireAssetId(e.target.value)}
            className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
          >
            <option value="">None (Prompt only)</option>
            {visualAssets
              .filter((a) => a.kind === 'image')
              .map((a) => (
                <option key={a.id} value={a.id}>
                  {a.title}
                </option>
              ))}
          </select>
        </div>

        <label className="flex items-center gap-2 text-xs text-[#C4C9D4] cursor-pointer">
          <input
            type="checkbox"
            checked={proMode}
            onChange={(e) => setProMode(e.target.checked)}
            className="accent-[#E5A93C]"
          />
          <span>Full-length Pro Track (vs. 30s Clip)</span>
        </label>

        {errorMsg && (
          <div
            role="alert"
            className="p-2.5 rounded-md bg-[#DC2626]/12 border border-[#DC2626]/40 text-xs text-[#FCA5A5]"
          >
            {errorMsg}
          </div>
        )}

        {generatedTrack && (
          <div className="p-2.5 rounded-md bg-[#0E1013] border border-white/10 flex flex-col gap-1.5">
            <span className="text-xs font-medium text-[#86EFAC]">
              Track added to A2
            </span>
            <audio controls src={generatedTrack.url} className="w-full h-8" />
          </div>
        )}

        <button
          type="submit"
          disabled={isGenerating}
          className="w-full py-2.5 px-4 rounded-md bg-[#1D2129] border border-white/15 text-[#F1F3F5] text-xs font-semibold hover:border-[#E5A93C] disabled:opacity-60 transition-colors flex items-center justify-center gap-2"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Composing Soundtrack...</span>
            </>
          ) : (
            <>
              <Music className="w-3.5 h-3.5" />
              <span>Compose via Lyria (Optional)</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
