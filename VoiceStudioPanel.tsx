import { useEffect, useState } from 'react';
import { Loader2, Mic, Plus } from 'lucide-react';
import { MediaAsset } from '../types/studio';

interface VoiceStudioPanelProps {
  initialLine?: string;
  onAddVoiceAsset: (asset: MediaAsset, alsoCreateSubtitles: boolean) => void;
}

const VOICES = ['Kore', 'Puck', 'Charon', 'Fenrir', 'Zephyr'];

export function VoiceStudioPanel({
  initialLine,
  onAddVoiceAsset,
}: VoiceStudioPanelProps) {
  const [mode, setMode] = useState<'single' | 'dual'>('single');
  const [voice, setVoice] = useState('Kore');
  const [styleDirection, setStyleDirection] = useState(
    'Warm cinematic narrator with clear dramatic pacing'
  );
  const [text, setText] = useState(
    'High above the alpine cloud line, <breath> the clockwork observatory awakens for one final journey.'
  );
  const [speakerA, setSpeakerA] = useState({
    name: 'Captain Fox',
    voice: 'Puck',
    style: 'Adventurous, cheerful animated hero',
    line: 'All aboard the Windrider! <laugh> Next stop, the twilight ridge!',
  });
  const [speakerB, setSpeakerB] = useState({
    name: 'Tinker Bot',
    voice: 'Charon',
    style: 'Gentle, earnest workshop companion',
    line: '|mhm| Just give me five seconds to calibrate the brass gyro.',
  });

  const [autoSubtitle, setAutoSubtitle] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [previewAudio, setPreviewAudio] = useState<{
    url: string;
    transcript: string;
    duration: number;
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (initialLine) {
      setText(initialLine);
      setMode('single');
    }
  }, [initialLine]);

  const insertTag = (tag: string) => {
    setText((prev) => `${prev.trim()} ${tag} `);
  };

  const handleGenerateVoice = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setIsGenerating(true);

    try {
      const res = await fetch('/api/media/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          text,
          voice,
          styleDirection,
          speakerA,
          speakerB,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.error || 'Voice generation failed.');
        return;
      }

      setPreviewAudio({
        url: data.audioDataUrl,
        transcript: data.transcript,
        duration: data.durationEstimate || 4,
      });

      const asset: MediaAsset = {
        id: `voice-${Date.now()}`,
        title:
          mode === 'dual'
            ? `${speakerA.name} & ${speakerB.name}`
            : `${voice} · ${text.replace(/<[^>]+>/g, '').slice(0, 20)}`,
        kind: 'voice',
        url: data.audioDataUrl,
        duration: data.durationEstimate || 4,
        styleTag: 'audio',
        transcript: data.transcript,
        createdAt: Date.now(),
      };
      onAddVoiceAsset(asset, autoSubtitle);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Network error generating voice.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <form onSubmit={handleGenerateVoice} className="flex flex-col gap-3.5 p-4 text-sm">
      <div className="grid grid-cols-2 gap-1 p-1 bg-[#0E1013] rounded-lg border border-white/8">
        <button
          type="button"
          onClick={() => setMode('single')}
          className={`py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
            mode === 'single'
              ? 'bg-[#222731] text-[#F1F3F5]'
              : 'text-[#8E96A4] hover:text-[#F1F3F5]'
          }`}
        >
          Single Actor
        </button>
        <button
          type="button"
          onClick={() => setMode('dual')}
          className={`py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
            mode === 'dual'
              ? 'bg-[#222731] text-[#F1F3F5]'
              : 'text-[#8E96A4] hover:text-[#F1F3F5]'
          }`}
        >
          Dual Screenplay
        </button>
      </div>

      {mode === 'single' ? (
        <>
          <div className="grid grid-cols-2 gap-2.5">
            <div className="flex flex-col gap-1">
              <label htmlFor="voice-persona" className="text-xs font-medium text-[#9BA1AE]">
                Voice Persona
              </label>
              <select
                id="voice-persona"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
                className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
              >
                {VOICES.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label htmlFor="voice-style" className="text-xs font-medium text-[#9BA1AE]">
                Acting Direction
              </label>
              <input
                id="voice-style"
                type="text"
                value={styleDirection}
                onChange={(e) => setStyleDirection(e.target.value)}
                className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="voice-script" className="text-xs font-medium text-[#9BA1AE]">
              Dialogue / Narration Script
            </label>
            <textarea
              id="voice-script"
              rows={3}
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full bg-[#0E1013] border border-white/10 rounded-md px-3 py-2 text-xs leading-relaxed text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C] resize-none"
            />
            <div className="flex flex-wrap items-center gap-1.5">
              {['<breath>', '<laugh>', '<gasp>', '|mhm|'].map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => insertTag(tag)}
                  className="px-2 py-1 rounded bg-[#0E1013] border border-white/10 text-[11px] font-mono text-[#C4C9D4] hover:border-[#E5A93C]"
                >
                  + {tag}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="flex flex-col gap-3">
          {[
            { label: 'Speaker A', state: speakerA, setter: setSpeakerA },
            { label: 'Speaker B', state: speakerB, setter: setSpeakerB },
          ].map(({ label, state, setter }) => (
            <div
              key={label}
              className="p-2.5 rounded-md bg-[#0E1013] border border-white/8 flex flex-col gap-2"
            >
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="text"
                  aria-label={`${label} name`}
                  value={state.name}
                  onChange={(e) => setter({ ...state, name: e.target.value })}
                  className="bg-[#15181E] border border-white/10 rounded px-2 py-1 text-xs text-[#F1F3F5]"
                />
                <select
                  aria-label={`${label} voice`}
                  value={state.voice}
                  onChange={(e) => setter({ ...state, voice: e.target.value })}
                  className="bg-[#15181E] border border-white/10 rounded px-2 py-1 text-xs text-[#F1F3F5]"
                >
                  {VOICES.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>
              <input
                type="text"
                aria-label={`${label} acting style`}
                value={state.style}
                onChange={(e) => setter({ ...state, style: e.target.value })}
                placeholder="Character acting style..."
                className="bg-[#15181E] border border-white/10 rounded px-2 py-1 text-xs text-[#C4C9D4]"
              />
              <textarea
                rows={2}
                aria-label={`${label} line`}
                value={state.line}
                onChange={(e) => setter({ ...state, line: e.target.value })}
                className="bg-[#15181E] border border-white/10 rounded px-2 py-1.5 text-xs text-[#F1F3F5] resize-none"
              />
            </div>
          ))}
        </div>
      )}

      <label className="flex items-center gap-2 text-xs text-[#C4C9D4] cursor-pointer">
        <input
          type="checkbox"
          checked={autoSubtitle}
          onChange={(e) => setAutoSubtitle(e.target.checked)}
          className="accent-[#E5A93C]"
        />
        <span>Auto-generate synced subtitles on S1</span>
      </label>

      {errorMsg && (
        <div
          role="alert"
          className="p-2.5 rounded-md bg-[#DC2626]/12 border border-[#DC2626]/40 text-xs text-[#FCA5A5]"
        >
          {errorMsg}
        </div>
      )}

      {previewAudio && (
        <div className="p-2.5 rounded-md bg-[#0E1013] border border-white/10 flex flex-col gap-2">
          <div className="text-xs font-medium text-[#86EFAC]">
            Voice performance ready ({previewAudio.duration}s)
          </div>
          <audio controls src={previewAudio.url} className="w-full h-8" />
        </div>
      )}

      <button
        type="submit"
        disabled={isGenerating}
        className="w-full py-2.5 px-4 rounded-md bg-[#E5A93C] text-[#111317] text-xs font-semibold hover:bg-[#EEB955] disabled:opacity-60 transition-colors flex items-center justify-center gap-2"
      >
        {isGenerating ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Synthesizing Voice Performance...</span>
          </>
        ) : (
          <>
            <Mic className="w-3.5 h-3.5" />
            <span>Perform & Add to A1 Voice Track</span>
          </>
        )}
      </button>
    </form>
  );
}
