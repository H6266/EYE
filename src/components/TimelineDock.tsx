import { useRef, useState } from 'react';
import {
  Captions,
  Film,
  Magnet,
  Mic,
  Music,
  Scissors,
  User,
  Volume2,
  VolumeX,
  ZoomIn,
} from 'lucide-react';
import {
  MediaAsset,
  TimelineClip,
  TrackConfig,
  TrackId,
} from '../types/studio';
import { formatTimecode } from '../utils/timelineEngine';

interface TimelineDockProps {
  tracks: TrackConfig[];
  clips: TimelineClip[];
  selectedClipId: string | null;
  playheadTime: number;
  totalDuration: number;
  snapEnabled: boolean;
  onToggleSnap: () => void;
  onToggleTrackMute: (trackId: TrackId) => void;
  onSelectClip: (clipId: string | null) => void;
  onSeek: (time: number) => void;
  onMoveClip: (clipId: string, newStart: number) => void;
  onResizeClip: (clipId: string, newDuration: number) => void;
  onSplitSelected: () => void;
  onDropAsset: (assetId: string, track: TrackId, startTime: number) => void;
  assets: MediaAsset[];
}

const TRACK_COLORS: Record<TrackId, string> = {
  video: 'bg-[#263248] border-[#4B699B] text-[#E5ECF6]',
  overlay: 'bg-[#2F2438] border-[#A855F7]/60 text-[#F3E8FF]',
  voice: 'bg-[#1D3532] border-[#348F7E] text-[#D8F3EE]',
  music: 'bg-[#2D283E] border-[#7862B5] text-[#ECE8F8]',
  subtitle: 'bg-[#362C1A] border-[#B48431] text-[#FDE68A]',
};

const TRACK_ICONS: Record<TrackId, typeof Film> = {
  video: Film,
  overlay: User,
  voice: Mic,
  music: Music,
  subtitle: Captions,
};

export function TimelineDock({
  tracks,
  clips,
  selectedClipId,
  playheadTime,
  totalDuration,
  snapEnabled,
  onToggleSnap,
  onToggleTrackMute,
  onSelectClip,
  onSeek,
  onMoveClip,
  onResizeClip,
  onSplitSelected,
  onDropAsset,
}: TimelineDockProps) {
  const [pxPerSec, setPxPerSec] = useState(48);
  const [dragOverTrack, setDragOverTrack] = useState<TrackId | null>(null);
  const laneScrollRef = useRef<HTMLDivElement | null>(null);

  const timelineWidth = Math.max(960, Math.ceil(totalDuration * pxPerSec) + 120);
  const tickMarks = Array.from({ length: Math.ceil(totalDuration) + 1 }, (_, i) => i);

  const selectedClip = clips.find((c) => c.id === selectedClipId) || null;
  const canSplit =
    Boolean(selectedClip) &&
    playheadTime > (selectedClip?.startTime ?? 0) + 0.35 &&
    playheadTime < (selectedClip?.startTime ?? 0) + (selectedClip?.duration ?? 0) - 0.35;

  const clientXToTime = (clientX: number): number => {
    const el = laneScrollRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left + el.scrollLeft;
    return Math.max(0, Number((x / pxPerSec).toFixed(2)));
  };

  const startClipMove = (e: React.PointerEvent, clip: TimelineClip) => {
    e.stopPropagation();
    onSelectClip(clip.id);
    const originX = e.clientX;
    const originStart = clip.startTime;

    const handleMove = (ev: PointerEvent) => {
      const deltaSec = (ev.clientX - originX) / pxPerSec;
      onMoveClip(clip.id, Math.max(0, originStart + deltaSec));
    };

    const handleUp = () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  };

  const startClipResize = (e: React.PointerEvent, clip: TimelineClip) => {
    e.stopPropagation();
    onSelectClip(clip.id);
    const originX = e.clientX;
    const originDuration = clip.duration;

    const handleMove = (ev: PointerEvent) => {
      const deltaSec = (ev.clientX - originX) / pxPerSec;
      onResizeClip(clip.id, Math.max(0.5, originDuration + deltaSec));
    };

    const handleUp = () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  };

  const handleTrackDrop = (e: React.DragEvent, trackId: TrackId) => {
    e.preventDefault();
    setDragOverTrack(null);
    const raw = e.dataTransfer.getData('application/json');
    if (!raw) return;
    try {
      const payload = JSON.parse(raw) as { type: string; assetId?: string };
      if (payload.type === 'asset' && payload.assetId) {
        const dropTime = clientXToTime(e.clientX);
        onDropAsset(payload.assetId, trackId, dropTime);
      }
    } catch {
      // Ignore invalid drag payload
    }
  };

  return (
    <section
      aria-label="Multi-Track Timeline"
      data-testid="timeline-dock"
      className="h-72 shrink-0 border-t border-white/10 bg-[#111419] flex flex-col select-none"
    >
      {/* Timeline Toolbar */}
      <div className="h-9 shrink-0 px-4 border-b border-white/8 bg-[#15181E] flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-[#F1F3F5]">Timeline</span>
          <button
            type="button"
            disabled={!canSplit}
            onClick={onSplitSelected}
            className="flex items-center gap-1 px-2 py-1 rounded bg-[#1D2129] border border-white/8 text-[11px] font-medium text-[#D1D5DB] hover:border-[#E5A93C] disabled:opacity-40 whitespace-nowrap"
          >
            <Scissors className="w-3 h-3" />
            <span>Split (S)</span>
          </button>

          <button
            type="button"
            onClick={onToggleSnap}
            aria-pressed={snapEnabled}
            className={`flex items-center gap-1 px-2 py-1 rounded border text-[11px] font-medium transition-colors whitespace-nowrap ${
              snapEnabled
                ? 'bg-[#E5A93C]/15 border-[#E5A93C]/40 text-[#FDE68A]'
                : 'bg-[#1D2129] border-white/8 text-[#8E96A4]'
            }`}
          >
            <Magnet className="w-3 h-3" />
            <span>Snap</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          <ZoomIn className="w-3.5 h-3.5 text-[#8E96A4]" />
          <input
            type="range"
            aria-label="Timeline zoom"
            min={24}
            max={96}
            value={pxPerSec}
            onChange={(e) => setPxPerSec(Number(e.target.value))}
            className="w-24 accent-[#E5A93C]"
          />
        </div>
      </div>

      {/* Track Headers + Scrollable Lanes */}
      <div className="flex-1 min-h-0 flex overflow-hidden">
        {/* Track Labels Column */}
        <div className="w-44 shrink-0 border-r border-white/8 bg-[#14171D] flex flex-col">
          <div className="h-6 border-b border-white/8 px-3 flex items-center text-[11px] font-mono text-[#7D8594]">
            {formatTimecode(playheadTime)}
          </div>
          <div className="flex-1 flex flex-col">
            {tracks.map((track) => {
              const Icon = TRACK_ICONS[track.id];
              return (
                <div
                  key={track.id}
                  className="flex-1 px-3 border-b border-white/6 flex items-center justify-between gap-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="font-mono text-[11px] font-semibold text-[#E5A93C] w-5">
                      {track.shortCode}
                    </span>
                    <Icon className="w-3.5 h-3.5 text-[#8E96A4] shrink-0" />
                    <span className="text-xs text-[#D1D5DB] truncate">{track.label}</span>
                  </div>
                  {(track.id === 'voice' || track.id === 'music') && (
                    <button
                      type="button"
                      aria-label={
                        track.muted ? `Unmute ${track.label}` : `Mute ${track.label}`
                      }
                      onClick={() => onToggleTrackMute(track.id)}
                      className={`p-1 rounded hover:bg-white/5 ${
                        track.muted ? 'text-[#EF4444]' : 'text-[#8E96A4]'
                      }`}
                    >
                      {track.muted ? (
                        <VolumeX className="w-3.5 h-3.5" />
                      ) : (
                        <Volume2 className="w-3.5 h-3.5" />
                      )}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Scrollable Time Ruler + Tracks */}
        <div
          ref={laneScrollRef}
          className="flex-1 min-w-0 overflow-x-auto overflow-y-hidden relative bg-[#0D1014]"
        >
          <div style={{ width: timelineWidth }} className="h-full flex flex-col relative">
            {/* Clickable Time Ruler */}
            <div
              role="slider"
              aria-label="Timeline playhead scrubber"
              aria-valuemin={0}
              aria-valuemax={totalDuration}
              aria-valuenow={Number(playheadTime.toFixed(1))}
              tabIndex={0}
              onClick={(e) => onSeek(clientXToTime(e.clientX))}
              onKeyDown={(e) => {
                if (e.key === 'ArrowRight') onSeek(Math.min(totalDuration, playheadTime + 0.5));
                if (e.key === 'ArrowLeft') onSeek(Math.max(0, playheadTime - 0.5));
              }}
              className="h-6 shrink-0 border-b border-white/8 bg-[#12151B] relative cursor-pointer"
            >
              {tickMarks.map((sec) => (
                <div
                  key={sec}
                  style={{ left: sec * pxPerSec }}
                  className="absolute top-0 bottom-0 border-l border-white/10 pl-1 flex items-center"
                >
                  {sec % 2 === 0 && (
                    <span className="font-mono text-[10px] text-[#7D8594]">
                      {sec}s
                    </span>
                  )}
                </div>
              ))}
            </div>

            {/* Track Lanes */}
            <div className="flex-1 flex flex-col relative">
              {tracks.map((track) => {
                const trackClips = clips.filter((c) => c.track === track.id);
                const isDropTarget = dragOverTrack === track.id;

                return (
                  <div
                    key={track.id}
                    data-testid={`track-lane-${track.id}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOverTrack(track.id);
                    }}
                    onDragLeave={() => setDragOverTrack(null)}
                    onDrop={(e) => handleTrackDrop(e, track.id)}
                    onClick={(e) => {
                      if (e.target === e.currentTarget) {
                        onSelectClip(null);
                      }
                    }}
                    className={`flex-1 relative border-b border-white/6 transition-colors ${
                      isDropTarget ? 'bg-[#E5A93C]/10' : ''
                    }`}
                  >
                    {trackClips.length === 0 && (
                      <div className="absolute inset-0 flex items-center px-3 pointer-events-none">
                        <span className="text-[11px] text-[#525A69]">
                          Drop {track.label.toLowerCase()} clips here
                        </span>
                      </div>
                    )}
                    {trackClips.map((clip) => {
                      const isSelected = clip.id === selectedClipId;
                      const left = clip.startTime * pxPerSec;
                      const width = Math.max(28, clip.duration * pxPerSec);

                      return (
                        <div
                          key={clip.id}
                          data-testid={`timeline-clip-${clip.id}`}
                          onPointerDown={(e) => startClipMove(e, clip)}
                          className={`absolute top-1 bottom-1 rounded border flex items-center overflow-hidden cursor-grab active:cursor-grabbing ${
                            TRACK_COLORS[track.id]
                          } ${
                            isSelected
                              ? 'ring-2 ring-[#E5A93C] brightness-110 z-10'
                              : 'hover:brightness-110'
                          }`}
                          style={{ left, width }}
                        >
                          {clip.mediaKind === 'image' && clip.mediaUrl && width > 64 && (
                            <img
                              src={clip.mediaUrl}
                              alt=""
                              referrerPolicy="no-referrer"
                              className="h-full w-10 object-cover shrink-0 opacity-85 border-r border-black/30 pointer-events-none"
                            />
                          )}
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectClip(clip.id);
                            }}
                            className="flex-1 min-w-0 px-2 text-left flex items-center justify-between gap-2 focus:outline-none"
                          >
                            <span className="text-[11px] font-medium truncate">
                              {clip.track === 'subtitle'
                                ? clip.subtitleText || clip.title
                                : clip.title}
                            </span>
                            {width > 120 && (
                              <span className="font-mono text-[10px] opacity-70 shrink-0">
                                {clip.duration.toFixed(1)}s
                              </span>
                            )}
                          </button>

                          {/* Right-edge trim handle */}
                          <div
                            role="separator"
                            aria-label={`Resize ${clip.title}`}
                            onPointerDown={(e) => startClipResize(e, clip)}
                            className="w-2 h-full shrink-0 cursor-ew-resize bg-black/25 hover:bg-[#E5A93C] transition-colors"
                          />
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>

            {/* Playhead Needle */}
            <div
              aria-hidden="true"
              style={{ left: playheadTime * pxPerSec }}
              className="absolute top-0 bottom-0 w-px bg-[#E5A93C] pointer-events-none z-20"
            >
              <div className="w-2.5 h-3 -ml-[5px] bg-[#E5A93C] rounded-b-sm" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
