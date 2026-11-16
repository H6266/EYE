import { useState } from 'react';
import { Captions, Download, Loader2, Plus, Trash2 } from 'lucide-react';
import { MediaAsset, SubtitleStyle, TimelineClip } from '../types/studio';
import {
  buildAutoSubtitleCues,
  exportSrt,
  exportVtt,
  formatSrtTimestamp,
} from '../utils/timelineEngine';

interface SubtitlePanelProps {
  clips: TimelineClip[];
  audioAssets: MediaAsset[];
  defaultSubtitleStyle: SubtitleStyle;
  onChangeDefaultStyle: (style: SubtitleStyle) => void;
  onUpdateClip: (clipId: string, patch: Partial<TimelineClip>) => void;
  onDeleteClip: (clipId: string) => void;
  onInsertSubtitleCues: (
    cues: Array<{ startTime: number; duration: number; text: string; speaker?: string }>
  ) => void;
  onSeek: (time: number) => void;
}

async function urlToBase64(url: string): Promise<{ base64: string; mimeType: string }> {
  const res = await fetch(url);
  const blob = await res.blob();
  const base64 = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const s = String(reader.result || '');
      resolve(s.includes(',') ? s.split(',')[1] : s);
    };
    reader.onerror = () => reject(new Error('Failed to encode audio'));
    reader.readAsDataURL(blob);
  });
  return { base64, mimeType: blob.type || 'audio/wav' };
}

export function SubtitlePanel({
  clips,
  audioAssets,
  defaultSubtitleStyle,
  onChangeDefaultStyle,
  onUpdateClip,
  onDeleteClip,
  onInsertSubtitleCues,
  onSeek,
}: SubtitlePanelProps) {
  const subtitleClips = clips
    .filter((c) => c.track === 'subtitle')
    .sort((a, b) => a.startTime - b.startTime);

  const [selectedAudioId, setSelectedAudioId] = useState<string>(
    audioAssets[0]?.id || ''
  );
  const [scriptDraft, setScriptDraft] = useState(
    'Welcome aboard the Windrider! Keep your goggles strapped tight as we cross the twilight canyon.'
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const handleTranscribeAudio = async () => {
    const target = audioAssets.find((a) => a.id === selectedAudioId);
    if (!target) {
      setErrorMsg('Select a voice or audio clip to transcribe.');
      return;
    }
    setErrorMsg(null);
    setStatusMsg(null);
    setIsGenerating(true);

    try {
      const { base64, mimeType } = await urlToBase64(target.url);
      const res = await fetch('/api/subtitles/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audioBase64: base64,
          audioMimeType: mimeType,
          totalDuration: target.duration || 8,
          startTimeOffset: 0,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.error || 'Transcription failed.');
        return;
      }
      if (Array.isArray(data.cues) && data.cues.length > 0) {
        onInsertSubtitleCues(data.cues);
        setStatusMsg(`Generated ${data.cues.length} timed subtitle cues from audio.`);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Subtitle transcription failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleAutoSegmentScript = async () => {
    if (!scriptDraft.trim()) {
      setErrorMsg('Enter dialogue or narration text to segment into subtitles.');
      return;
    }
    setErrorMsg(null);
    setStatusMsg(null);

    // Local fast-path fallback + AI timing alignment
    const startAt =
      subtitleClips.length > 0
        ? subtitleClips[subtitleClips.length - 1].startTime +
          subtitleClips[subtitleClips.length - 1].duration +
          0.3
        : 0;

    setIsGenerating(true);
    try {
      const res = await fetch('/api/subtitles/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scriptText: scriptDraft,
          totalDuration: Math.max(4, Math.round(scriptDraft.split(/\s+/).length * 0.45)),
          startTimeOffset: startAt,
        }),
      });
      const data = await res.json();
      if (res.ok && Array.isArray(data.cues) && data.cues.length > 0) {
        onInsertSubtitleCues(data.cues);
        setStatusMsg(`Added ${data.cues.length} AI-paced subtitle cues.`);
      } else {
        const localCues = buildAutoSubtitleCues(
          scriptDraft,
          startAt,
          Math.max(3, scriptDraft.split(/\s+/).length * 0.42)
        );
        onInsertSubtitleCues(localCues);
        setStatusMsg(`Segmented ${localCues.length} subtitle cues locally.`);
      }
    } catch {
      const localCues = buildAutoSubtitleCues(
        scriptDraft,
        startAt,
        Math.max(3, scriptDraft.split(/\s+/).length * 0.42)
      );
      onInsertSubtitleCues(localCues);
      setStatusMsg(`Segmented ${localCues.length} subtitle cues.`);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadFile = (content: string, filename: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-4 p-4 text-sm">
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-[#9BA1AE]">Subtitle Visual Style</span>
        <div className="grid grid-cols-3 gap-1 p-1 bg-[#0E1013] rounded-lg border border-white/8">
          {(
            [
              { id: 'cinematic', label: 'Cinema' },
              { id: 'cartoon', label: 'Cartoon' },
              { id: 'minimal', label: 'Minimal' },
            ] as const
          ).map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => onChangeDefaultStyle(s.id)}
              className={`py-1.5 rounded text-xs font-medium transition-colors whitespace-nowrap ${
                defaultSubtitleStyle === s.id
                  ? 'bg-[#222731] text-[#F1F3F5]'
                  : 'text-[#8E96A4] hover:text-[#F1F3F5]'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {audioAssets.length > 0 && (
        <div className="p-3 rounded-md bg-[#0E1013] border border-white/8 flex flex-col gap-2">
          <label
            htmlFor="transcribe-source"
            className="text-xs font-medium text-[#9BA1AE]"
          >
            Auto-Transcribe Voice / Audio Clip
          </label>
          <select
            id="transcribe-source"
            value={selectedAudioId}
            onChange={(e) => setSelectedAudioId(e.target.value)}
            className="bg-[#15181E] border border-white/10 rounded px-2.5 py-1.5 text-xs text-[#F1F3F5]"
          >
            {audioAssets.map((a) => (
              <option key={a.id} value={a.id}>
                {a.title} ({a.duration.toFixed(1)}s)
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={handleTranscribeAudio}
            disabled={isGenerating}
            className="py-1.5 px-3 rounded bg-[#1D2129] border border-white/10 text-xs font-medium text-[#F1F3F5] hover:border-[#E5A93C] disabled:opacity-50 flex items-center justify-center gap-1.5"
          >
            {isGenerating ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Captions className="w-3.5 h-3.5 text-[#E5A93C]" />
            )}
            <span>Transcribe Audio to Subtitles</span>
          </button>
        </div>
      )}

      <div className="flex flex-col gap-1.5">
        <label htmlFor="sub-script" className="text-xs font-medium text-[#9BA1AE]">
          Generate Cues from Script
        </label>
        <textarea
          id="sub-script"
          rows={2}
          value={scriptDraft}
          onChange={(e) => setScriptDraft(e.target.value)}
          className="w-full bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-2 text-xs text-[#F1F3F5] resize-none focus:outline-none focus:border-[#E5A93C]"
        />
        <button
          type="button"
          onClick={handleAutoSegmentScript}
          disabled={isGenerating}
          className="py-2 px-3 rounded-md bg-[#E5A93C] text-[#111317] text-xs font-semibold hover:bg-[#EEB955] disabled:opacity-50 flex items-center justify-center gap-1.5"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Auto-Time & Add Cues</span>
        </button>
      </div>

      {errorMsg && (
        <div
          role="alert"
          className="p-2 rounded bg-[#DC2626]/12 border border-[#DC2626]/40 text-xs text-[#FCA5A5]"
        >
          {errorMsg}
        </div>
      )}
      {statusMsg && (
        <div className="p-2 rounded bg-[#16A34A]/12 border border-[#16A34A]/40 text-xs text-[#86EFAC]">
          {statusMsg}
        </div>
      )}

      <div className="flex items-center justify-between pt-2 border-t border-white/8">
        <span className="text-xs font-semibold text-[#F1F3F5]">
          Subtitle Cues ({subtitleClips.length})
        </span>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() =>
              downloadFile(exportSrt(clips), 'auteur-subtitles.srt', 'text/plain')
            }
            className="flex items-center gap-1 px-2 py-1 rounded bg-[#1D2129] border border-white/10 text-[11px] font-medium text-[#D1D5DB] hover:border-[#E5A93C]"
          >
            <Download className="w-3 h-3" />
            <span>SRT</span>
          </button>
          <button
            type="button"
            onClick={() =>
              downloadFile(exportVtt(clips), 'auteur-subtitles.vtt', 'text/vtt')
            }
            className="flex items-center gap-1 px-2 py-1 rounded bg-[#1D2129] border border-white/10 text-[11px] font-medium text-[#D1D5DB] hover:border-[#E5A93C]"
          >
            <Download className="w-3 h-3" />
            <span>VTT</span>
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        {subtitleClips.map((cue) => (
          <div
            key={cue.id}
            className="p-2.5 rounded-md bg-[#12151B] border border-white/8 flex flex-col gap-1.5"
          >
            <div className="flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => onSeek(cue.startTime)}
                className="font-mono text-[11px] text-[#E5A93C] hover:underline"
              >
                {formatSrtTimestamp(cue.startTime).slice(3, 8)} →{' '}
                {formatSrtTimestamp(cue.startTime + cue.duration).slice(3, 8)}
              </button>
              <button
                type="button"
                aria-label={`Delete subtitle ${cue.title}`}
                onClick={() => onDeleteClip(cue.id)}
                className="text-[#7D8594] hover:text-[#FCA5A5]"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
            <input
              type="text"
              aria-label={`Speaker for ${cue.title}`}
              value={cue.subtitleSpeaker || ''}
              placeholder="Speaker (optional)"
              onChange={(e) =>
                onUpdateClip(cue.id, { subtitleSpeaker: e.target.value })
              }
              className="bg-[#0E1013] border border-white/8 rounded px-2 py-1 text-[11px] text-[#9BA1AE]"
            />
            <textarea
              rows={2}
              aria-label={`Caption text for ${cue.title}`}
              value={cue.subtitleText || ''}
              onChange={(e) =>
                onUpdateClip(cue.id, { subtitleText: e.target.value })
              }
              className="bg-[#0E1013] border border-white/8 rounded px-2 py-1 text-xs text-[#F1F3F5] resize-none"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
