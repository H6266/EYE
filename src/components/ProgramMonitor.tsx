import { useEffect, useRef } from 'react';
import {
  Captions,
  Pause,
  Play,
  Repeat,
  SkipBack,
  SkipForward,
  StepBack,
  StepForward,
} from 'lucide-react';
import { AspectRatio, TimelineClip } from '../types/studio';
import { formatTimecode } from '../utils/timelineEngine';
import { getCanvasDimensions, MediaCache, renderCompositeFrame } from '../utils/canvasRenderer';

interface ProgramMonitorProps {
  clips: TimelineClip[];
  currentTime: number;
  totalDuration: number;
  isPlaying: boolean;
  loop: boolean;
  showSubtitles: boolean;
  aspectRatio: AspectRatio;
  mediaCache: MediaCache;
  onTogglePlay: () => void;
  onSeek: (time: number) => void;
  onToggleLoop: () => void;
  onToggleSubtitles: () => void;
  onChangeAspectRatio: (ratio: AspectRatio) => void;
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
}

export function ProgramMonitor({
  clips,
  currentTime,
  totalDuration,
  isPlaying,
  loop,
  showSubtitles,
  aspectRatio,
  mediaCache,
  onTogglePlay,
  onSeek,
  onToggleLoop,
  onToggleSubtitles,
  onChangeAspectRatio,
  onCanvasReady,
}: ProgramMonitorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { width, height } = getCanvasDimensions(aspectRatio);

  useEffect(() => {
    if (onCanvasReady) onCanvasReady(canvasRef.current);
  }, [onCanvasReady]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    renderCompositeFrame(
      ctx,
      clips,
      currentTime,
      mediaCache,
      aspectRatio,
      isPlaying,
      showSubtitles
    );
  }, [clips, currentTime, mediaCache, aspectRatio, isPlaying, showSubtitles]);

  // Ensure initial asset images repaint canvas once decoded
  useEffect(() => {
    const timer = window.setInterval(() => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;
      renderCompositeFrame(
        ctx,
        clips,
        currentTime,
        mediaCache,
        aspectRatio,
        isPlaying,
        showSubtitles
      );
    }, 250);
    return () => window.clearInterval(timer);
  }, [clips, currentTime, mediaCache, aspectRatio, isPlaying, showSubtitles]);

  return (
    <section
      aria-label="Program Monitor"
      className="flex-1 min-w-0 flex flex-col bg-[#0B0D10] overflow-hidden"
    >
      <div className="h-10 shrink-0 px-4 border-b border-white/8 flex items-center justify-between gap-4 bg-[#12151A]">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-[#F1F3F5]">Program Monitor</span>
          <span className="font-mono text-xs text-[#E5A93C]">
            {formatTimecode(currentTime)}
          </span>
          <span className="font-mono text-xs text-[#6B7280]">
            / {formatTimecode(totalDuration)}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="monitor-aspect" className="sr-only">
            Monitor Aspect Ratio
          </label>
          <select
            id="monitor-aspect"
            value={aspectRatio}
            onChange={(e) => onChangeAspectRatio(e.target.value as AspectRatio)}
            className="bg-[#0E1013] border border-white/10 rounded px-2 py-1 text-xs text-[#C4C9D4] focus:outline-none focus:border-[#E5A93C]"
          >
            <option value="16:9">16:9 Cinema</option>
            <option value="9:16">9:16 Vertical</option>
            <option value="1:1">1:1 Square</option>
          </select>

          <button
            type="button"
            aria-label={showSubtitles ? 'Hide subtitles overlay' : 'Show subtitles overlay'}
            aria-pressed={showSubtitles}
            onClick={onToggleSubtitles}
            className={`p-1.5 rounded border text-xs transition-colors ${
              showSubtitles
                ? 'bg-[#E5A93C]/15 border-[#E5A93C]/50 text-[#FDE68A]'
                : 'bg-[#0E1013] border-white/10 text-[#8E96A4]'
            }`}
          >
            <Captions className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 p-4 flex items-center justify-center bg-[#090B0E]">
        <div className="relative max-w-full max-h-full flex items-center justify-center shadow-2xl border border-white/10 rounded overflow-hidden bg-black">
          <canvas
            ref={canvasRef}
            data-testid="program-canvas"
            width={width}
            height={height}
            className="max-w-full max-h-[44vh] w-auto h-auto object-contain block"
          />
        </div>
      </div>

      <div className="h-12 shrink-0 px-4 border-t border-white/8 bg-[#12151A] flex items-center justify-center gap-3">
        <button
          type="button"
          aria-label="Jump to start"
          onClick={() => onSeek(0)}
          className="p-2 rounded-md text-[#9BA1AE] hover:text-[#F1F3F5] hover:bg-white/5 transition-colors"
        >
          <SkipBack className="w-4 h-4" />
        </button>
        <button
          type="button"
          aria-label="Step backward 1 second"
          onClick={() => onSeek(Math.max(0, currentTime - 1))}
          className="p-2 rounded-md text-[#9BA1AE] hover:text-[#F1F3F5] hover:bg-white/5 transition-colors"
        >
          <StepBack className="w-4 h-4" />
        </button>

        <button
          type="button"
          aria-label={isPlaying ? 'Pause playback' : 'Play sequence'}
          onClick={onTogglePlay}
          className="px-4 py-2 rounded-md bg-[#E5A93C] text-[#111317] hover:bg-[#EEB955] transition-colors flex items-center gap-1.5 text-xs font-semibold"
        >
          {isPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5 fill-current" />
              <span>Pause</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Play</span>
            </>
          )}
        </button>

        <button
          type="button"
          aria-label="Step forward 1 second"
          onClick={() => onSeek(Math.min(totalDuration, currentTime + 1))}
          className="p-2 rounded-md text-[#9BA1AE] hover:text-[#F1F3F5] hover:bg-white/5 transition-colors"
        >
          <StepForward className="w-4 h-4" />
        </button>
        <button
          type="button"
          aria-label="Jump to end"
          onClick={() => onSeek(totalDuration)}
          className="p-2 rounded-md text-[#9BA1AE] hover:text-[#F1F3F5] hover:bg-white/5 transition-colors"
        >
          <SkipForward className="w-4 h-4" />
        </button>

        <button
          type="button"
          aria-label="Toggle loop playback"
          aria-pressed={loop}
          onClick={onToggleLoop}
          className={`p-2 rounded-md transition-colors ${
            loop
              ? 'text-[#E5A93C] bg-[#E5A93C]/12'
              : 'text-[#8E96A4] hover:text-[#F1F3F5]'
          }`}
        >
          <Repeat className="w-4 h-4" />
        </button>
      </div>
    </section>
  );
}
