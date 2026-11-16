import { useState } from 'react';
import { Download, Film, FileText, Loader2, X } from 'lucide-react';
import { AspectRatio, MediaAsset, TimelineClip } from '../types/studio';
import { exportSrt, exportVtt } from '../utils/timelineEngine';

interface ExportViewProps {
  clips: TimelineClip[];
  assets: MediaAsset[];
  totalDuration: number;
  aspectRatio: AspectRatio;
  canvas: HTMLCanvasElement | null;
  onStartExportPlayback: () => void;
  onStopExportPlayback: () => void;
  onClose: () => void;
}

export function ExportView({
  clips,
  assets,
  totalDuration,
  aspectRatio,
  canvas,
  onStartExportPlayback,
  onStopExportPlayback,
  onClose,
}: ExportViewProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [progress, setProgress] = useState(0);
  const [exportedVideoUrl, setExportedVideoUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const downloadTextFile = (content: string, filename: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleRenderWebm = async () => {
    setErrorMsg(null);
    if (!canvas || typeof canvas.captureStream !== 'function' || typeof MediaRecorder === 'undefined') {
      setErrorMsg('Browser video capture is unavailable in this environment.');
      return;
    }

    setIsRecording(true);
    setProgress(0);
    setExportedVideoUrl(null);

    try {
      const stream = canvas.captureStream(30);
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';
      const recorder = new MediaRecorder(stream, { mimeType });
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      const donePromise = new Promise<string>((resolve) => {
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: mimeType });
          resolve(URL.createObjectURL(blob));
        };
      });

      recorder.start(250);
      onStartExportPlayback();

      const renderMs = Math.min(20000, Math.max(3000, totalDuration * 1000));
      const startedAt = Date.now();
      const interval = window.setInterval(() => {
        const pct = Math.min(100, Math.round(((Date.now() - startedAt) / renderMs) * 100));
        setProgress(pct);
      }, 200);

      await new Promise((resolve) => setTimeout(resolve, renderMs));
      window.clearInterval(interval);
      onStopExportPlayback();
      recorder.stop();

      const url = await donePromise;
      setExportedVideoUrl(url);
      setProgress(100);
    } catch (err) {
      onStopExportPlayback();
      setErrorMsg(err instanceof Error ? err.message : 'Failed to render WebM movie.');
    } finally {
      setIsRecording(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-label="Export Movie & Project"
      className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4"
    >
      <div className="w-full max-w-lg rounded-lg bg-[#15181E] border border-white/12 p-6 flex flex-col gap-5 shadow-2xl">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-display text-lg font-bold text-[#F1F3F5]">
              Export Production
            </h2>
            <p className="text-xs text-[#8E96A4] mt-0.5">
              Render composited frames, transitions, and burned-in subtitles.
            </p>
          </div>
          <button
            type="button"
            aria-label="Close export modal"
            onClick={onClose}
            className="p-1.5 rounded-md text-[#8E96A4] hover:text-[#F1F3F5] hover:bg-white/5"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="grid grid-cols-3 gap-3 p-3 rounded-md bg-[#0E1013] border border-white/8 text-xs">
          <div>
            <div className="text-[#7D8594]">Aspect</div>
            <div className="font-mono text-[#F1F3F5] mt-0.5">{aspectRatio}</div>
          </div>
          <div>
            <div className="text-[#7D8594]">Duration</div>
            <div className="font-mono text-[#F1F3F5] mt-0.5">
              {totalDuration.toFixed(1)}s
            </div>
          </div>
          <div>
            <div className="text-[#7D8594]">Clips</div>
            <div className="font-mono text-[#F1F3F5] mt-0.5">{clips.length}</div>
          </div>
        </div>

        {errorMsg && (
          <div
            role="alert"
            className="p-3 rounded-md bg-[#DC2626]/12 border border-[#DC2626]/40 text-xs text-[#FCA5A5]"
          >
            {errorMsg}
          </div>
        )}

        <div className="flex flex-col gap-2.5">
          <button
            type="button"
            onClick={handleRenderWebm}
            disabled={isRecording}
            className="w-full py-2.5 px-4 rounded-md bg-[#E5A93C] text-[#111317] text-xs font-semibold hover:bg-[#EEB955] disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {isRecording ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Rendering Movie ({progress}%)...</span>
              </>
            ) : (
              <>
                <Film className="w-4 h-4" />
                <span>Render Master Video (.webm)</span>
              </>
            )}
          </button>

          {exportedVideoUrl && (
            <a
              href={exportedVideoUrl}
              download="auteur-studio-master.webm"
              className="w-full py-2 px-4 rounded-md bg-[#16A34A]/20 border border-[#16A34A]/50 text-[#86EFAC] text-xs font-semibold flex items-center justify-center gap-2 hover:bg-[#16A34A]/30"
            >
              <Download className="w-4 h-4" />
              <span>Download Master Video (.webm)</span>
            </a>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 pt-3 border-t border-white/8">
          <button
            type="button"
            onClick={() =>
              downloadTextFile(exportSrt(clips), 'auteur-subtitles.srt', 'text/plain')
            }
            className="py-2 px-3 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#D1D5DB] hover:border-[#E5A93C] flex items-center justify-center gap-1.5"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Export SRT</span>
          </button>

          <button
            type="button"
            onClick={() =>
              downloadTextFile(exportVtt(clips), 'auteur-subtitles.vtt', 'text/vtt')
            }
            className="py-2 px-3 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#D1D5DB] hover:border-[#E5A93C] flex items-center justify-center gap-1.5"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Export VTT</span>
          </button>

          <button
            type="button"
            onClick={() =>
              downloadTextFile(
                JSON.stringify({ clips, assets }, null, 2),
                'auteur-project.json',
                'application/json'
              )
            }
            className="py-2 px-3 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#D1D5DB] hover:border-[#E5A93C] flex items-center justify-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Project JSON</span>
          </button>
        </div>
      </div>
    </div>
  );
}
