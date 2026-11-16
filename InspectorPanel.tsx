import { Copy, Scissors, Trash2 } from 'lucide-react';
import {
  SubtitleStyle,
  TimelineClip,
  TransitionType,
  VisualStyle,
} from '../types/studio';

interface InspectorPanelProps {
  selectedClip: TimelineClip | null;
  playheadTime: number;
  totalClips: number;
  totalDuration: number;
  onUpdateClip: (clipId: string, patch: Partial<TimelineClip>) => void;
  onSplitAtPlayhead: (clipId: string) => void;
  onDuplicateClip: (clipId: string) => void;
  onDeleteClip: (clipId: string) => void;
}

export function InspectorPanel({
  selectedClip,
  playheadTime,
  totalClips,
  totalDuration,
  onUpdateClip,
  onSplitAtPlayhead,
  onDuplicateClip,
  onDeleteClip,
}: InspectorPanelProps) {
  if (!selectedClip) {
    return (
      <aside
        aria-label="Clip Inspector"
        className="w-72 shrink-0 border-l border-white/8 bg-[#12151A] p-4 flex flex-col gap-4 overflow-y-auto"
      >
        <h2 className="text-xs font-semibold text-[#F1F3F5]">Inspector</h2>
        <div className="p-3 rounded-md bg-[#0E1013] border border-white/8 flex flex-col gap-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-[#8E96A4]">Timeline Clips</span>
            <span className="font-mono text-[#F1F3F5]">{totalClips}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[#8E96A4]">Sequence Span</span>
            <span className="font-mono text-[#F1F3F5]">{totalDuration.toFixed(1)}s</span>
          </div>
        </div>
        <p className="text-xs text-[#7D8594] leading-relaxed">
          Select any video, voice, music, or subtitle clip on the timeline to adjust its visual style grade, transition, audio gain, or timing.
        </p>
      </aside>
    );
  }

  const canSplit =
    playheadTime > selectedClip.startTime + 0.35 &&
    playheadTime < selectedClip.startTime + selectedClip.duration - 0.35;

  return (
    <aside
      aria-label="Clip Inspector"
      className="w-72 shrink-0 border-l border-white/8 bg-[#12151A] p-4 flex flex-col gap-4 overflow-y-auto text-sm"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold text-[#F1F3F5] truncate">
          Clip Inspector
        </h2>
        <span className="font-mono text-[11px] text-[#E5A93C] uppercase">
          {selectedClip.track}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <label htmlFor="clip-title" className="text-xs font-medium text-[#9BA1AE]">
          Clip Name
        </label>
        <input
          id="clip-title"
          type="text"
          value={selectedClip.title}
          onChange={(e) => onUpdateClip(selectedClip.id, { title: e.target.value })}
          className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] focus:outline-none focus:border-[#E5A93C]"
        />
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        <div className="flex flex-col gap-1">
          <label htmlFor="clip-start" className="text-xs font-medium text-[#9BA1AE]">
            Start (s)
          </label>
          <input
            id="clip-start"
            type="number"
            min={0}
            step={0.1}
            value={selectedClip.startTime}
            onChange={(e) =>
              onUpdateClip(selectedClip.id, {
                startTime: Math.max(0, Number(e.target.value) || 0),
              })
            }
            className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs font-mono text-[#F1F3F5]"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="clip-dur" className="text-xs font-medium text-[#9BA1AE]">
            Duration (s)
          </label>
          <input
            id="clip-dur"
            type="number"
            min={0.5}
            step={0.1}
            value={selectedClip.duration}
            onChange={(e) =>
              onUpdateClip(selectedClip.id, {
                duration: Math.max(0.5, Number(e.target.value) || 0.5),
              })
            }
            className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs font-mono text-[#F1F3F5]"
          />
        </div>
      </div>

      {selectedClip.track === 'video' && (
        <>
          <div className="flex flex-col gap-1">
            <label htmlFor="clip-style" className="text-xs font-medium text-[#9BA1AE]">
              Visual Grade & Style
            </label>
            <select
              id="clip-style"
              value={selectedClip.visualStyle}
              onChange={(e) =>
                onUpdateClip(selectedClip.id, {
                  visualStyle: e.target.value as VisualStyle,
                })
              }
              className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5]"
            >
              <option value="original">Original Render</option>
              <option value="cinematic">Cinematic 2.39:1 Letterbox</option>
              <option value="cartoon2d">2D Cartoon Ink & Cel</option>
              <option value="anime">Anime Twilight Bloom</option>
              <option value="claymation">Claymation Warmth</option>
              <option value="pixar3d">3D Animated Cinema</option>
              <option value="watercolor">Storybook Watercolor</option>
              <option value="noir">Film Noir Silver</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <div className="flex flex-col gap-1">
              <label htmlFor="clip-trans" className="text-xs font-medium text-[#9BA1AE]">
                Intro Transition
              </label>
              <select
                id="clip-trans"
                value={selectedClip.transition}
                onChange={(e) =>
                  onUpdateClip(selectedClip.id, {
                    transition: e.target.value as TransitionType,
                    transitionDuration:
                      e.target.value === 'cut'
                        ? 0
                        : Math.max(0.5, selectedClip.transitionDuration || 0.7),
                  })
                }
                className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5]"
              >
                <option value="cut">Hard Cut</option>
                <option value="crossfade">Crossfade</option>
                <option value="wipe">Cinema Wipe</option>
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor="clip-motion" className="text-xs font-medium text-[#9BA1AE]">
                Camera Motion
              </label>
              <select
                id="clip-motion"
                value={selectedClip.kenBurns || 'zoom-in'}
                onChange={(e) =>
                  onUpdateClip(selectedClip.id, {
                    kenBurns: e.target.value as TimelineClip['kenBurns'],
                  })
                }
                className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5]"
              >
                <option value="zoom-in">Slow Push In</option>
                <option value="zoom-out">Pull Back</option>
                <option value="pan-right">Pan Right</option>
                <option value="static">Locked Tripod</option>
              </select>
            </div>
          </div>
        </>
      )}

      {selectedClip.track === 'overlay' && (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-2.5">
            <div className="flex flex-col gap-1">
              <label htmlFor="overlay-pos" className="text-xs font-medium text-[#9BA1AE]">
                Stage Position
              </label>
              <select
                id="overlay-pos"
                value={selectedClip.overlayPosition || 'right'}
                onChange={(e) =>
                  onUpdateClip(selectedClip.id, {
                    overlayPosition: e.target.value as TimelineClip['overlayPosition'],
                  })
                }
                className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5]"
              >
                <option value="left">Left Side</option>
                <option value="center">Center Stage</option>
                <option value="right">Right Side</option>
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor="overlay-frame" className="text-xs font-medium text-[#9BA1AE]">
                Frame Cutout
              </label>
              <select
                id="overlay-frame"
                value={selectedClip.overlayFrame || 'portrait'}
                onChange={(e) =>
                  onUpdateClip(selectedClip.id, {
                    overlayFrame: e.target.value as TimelineClip['overlayFrame'],
                  })
                }
                className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs text-[#F1F3F5]"
              >
                <option value="portrait">Rounded Card</option>
                <option value="badge">Circular Badge</option>
                <option value="full">Borderless</option>
              </select>
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="overlay-scale" className="font-medium text-[#9BA1AE]">
                Character Scale
              </label>
              <span className="font-mono text-[#F1F3F5]">
                {Math.round((selectedClip.overlayScale ?? 0.42) * 100)}%
              </span>
            </div>
            <input
              id="overlay-scale"
              type="range"
              min={0.2}
              max={0.85}
              step={0.05}
              value={selectedClip.overlayScale ?? 0.42}
              onChange={(e) =>
                onUpdateClip(selectedClip.id, {
                  overlayScale: Number(e.target.value),
                })
              }
              className="accent-[#E5A93C]"
            />
          </div>
        </div>
      )}

      {(selectedClip.track === 'voice' || selectedClip.track === 'music') && (
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="clip-vol" className="font-medium text-[#9BA1AE]">
                Audio Gain
              </label>
              <span className="font-mono text-[#F1F3F5]">
                {Math.round(selectedClip.volume * 100)}%
              </span>
            </div>
            <input
              id="clip-vol"
              type="range"
              min={0}
              max={1.25}
              step={0.05}
              value={selectedClip.volume}
              onChange={(e) =>
                onUpdateClip(selectedClip.id, { volume: Number(e.target.value) })
              }
              className="accent-[#E5A93C]"
            />
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <div className="flex flex-col gap-1">
              <label htmlFor="clip-fadein" className="text-xs font-medium text-[#9BA1AE]">
                Fade In (s)
              </label>
              <input
                id="clip-fadein"
                type="number"
                min={0}
                max={5}
                step={0.2}
                value={selectedClip.fadeIn}
                onChange={(e) =>
                  onUpdateClip(selectedClip.id, {
                    fadeIn: Math.max(0, Number(e.target.value) || 0),
                  })
                }
                className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs font-mono text-[#F1F3F5]"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label htmlFor="clip-fadeout" className="text-xs font-medium text-[#9BA1AE]">
                Fade Out (s)
              </label>
              <input
                id="clip-fadeout"
                type="number"
                min={0}
                max={5}
                step={0.2}
                value={selectedClip.fadeOut}
                onChange={(e) =>
                  onUpdateClip(selectedClip.id, {
                    fadeOut: Math.max(0, Number(e.target.value) || 0),
                  })
                }
                className="bg-[#0E1013] border border-white/10 rounded-md px-2 py-1.5 text-xs font-mono text-[#F1F3F5]"
              />
            </div>
          </div>
        </div>
      )}

      {selectedClip.track === 'subtitle' && (
        <div className="flex flex-col gap-2.5">
          <div className="flex flex-col gap-1">
            <label htmlFor="sub-speaker" className="text-xs font-medium text-[#9BA1AE]">
              Speaker Name
            </label>
            <input
              id="sub-speaker"
              type="text"
              value={selectedClip.subtitleSpeaker || ''}
              onChange={(e) =>
                onUpdateClip(selectedClip.id, { subtitleSpeaker: e.target.value })
              }
              className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5]"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="sub-text" className="text-xs font-medium text-[#9BA1AE]">
              Caption Text
            </label>
            <textarea
              id="sub-text"
              rows={3}
              value={selectedClip.subtitleText || ''}
              onChange={(e) =>
                onUpdateClip(selectedClip.id, { subtitleText: e.target.value })
              }
              className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] resize-none"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="sub-style" className="text-xs font-medium text-[#9BA1AE]">
              Caption Preset
            </label>
            <select
              id="sub-style"
              value={selectedClip.subtitleStyle || 'cinematic'}
              onChange={(e) =>
                onUpdateClip(selectedClip.id, {
                  subtitleStyle: e.target.value as SubtitleStyle,
                })
              }
              className="bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5]"
            >
              <option value="cinematic">Cinema Box</option>
              <option value="cartoon">Cartoon Pop</option>
              <option value="minimal">Minimal Shadow</option>
            </select>
          </div>
        </div>
      )}

      <div className="pt-3 border-t border-white/8 flex flex-col gap-2">
        <button
          type="button"
          disabled={!canSplit}
          onClick={() => onSplitAtPlayhead(selectedClip.id)}
          className="w-full py-2 px-3 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#F1F3F5] hover:border-[#E5A93C] disabled:opacity-40 flex items-center justify-center gap-2"
        >
          <Scissors className="w-3.5 h-3.5" />
          <span>Split at Playhead</span>
        </button>

        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => onDuplicateClip(selectedClip.id)}
            className="py-2 px-3 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#D1D5DB] hover:border-white/25 flex items-center justify-center gap-1.5"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>Duplicate</span>
          </button>

          <button
            type="button"
            onClick={() => onDeleteClip(selectedClip.id)}
            className="py-2 px-3 rounded-md bg-[#DC2626]/15 border border-[#DC2626]/35 text-xs font-medium text-[#FCA5A5] hover:bg-[#DC2626]/25 flex items-center justify-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Delete</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
