import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Captions, FolderOpen, Mic, Music, Sparkles, Users } from 'lucide-react';
import {
  AspectRatio,
  CharacterProfile,
  LeftTab,
  MediaAsset,
  SubtitleStyle,
  TimelineClip,
  TrackConfig,
  TrackId,
  VisualStyle,
  WorkspaceView,
} from './types/studio';
import {
  INITIAL_TIMELINE_CLIPS,
  INITIAL_TRACKS,
  SEED_MEDIA_ASSETS,
} from './data/seedProject';
import {
  buildAutoSubtitleCues,
  getProjectDuration,
  moveClip,
  resizeClip,
  splitClipAtTime,
} from './utils/timelineEngine';
import { MediaCache } from './utils/canvasRenderer';
import { synthesizeScoreStem, TimelineAudioEngine } from './utils/audioMixer';
import { TopBar } from './components/TopBar';
import { GeneratePanel } from './components/GeneratePanel';
import { CharacterCastPanel } from './components/CharacterCastPanel';
import { MediaBinPanel } from './components/MediaBinPanel';
import { VoiceStudioPanel } from './components/VoiceStudioPanel';
import { MusicPanel } from './components/MusicPanel';
import { SubtitlePanel } from './components/SubtitlePanel';
import { ProgramMonitor } from './components/ProgramMonitor';
import { InspectorPanel } from './components/InspectorPanel';
import { TimelineDock } from './components/TimelineDock';
import { ExportView } from './components/ExportView';

const LEFT_TABS: Array<{ id: LeftTab; label: string; Icon: typeof Sparkles }> = [
  { id: 'generate', label: 'Create', Icon: Sparkles },
  { id: 'characters', label: 'Cast', Icon: Users },
  { id: 'media', label: 'Media', Icon: FolderOpen },
  { id: 'voice', label: 'Voice', Icon: Mic },
  { id: 'music', label: 'Music', Icon: Music },
  { id: 'subtitles', label: 'Subs', Icon: Captions },
];

const INITIAL_CHARACTERS: CharacterProfile[] = [
  {
    id: 'char-fox',
    name: 'Captain Fox',
    imageUrl: '/src/assets/images/cartoon_fox_aviator_1794809573928.jpg',
    traits:
      '2D animated red fox aviator wearing brass goggles, brown leather flight jacket, amber scarf',
    createdAt: 1700000010,
  },
  {
    id: 'char-robot',
    name: 'Tinker Bot',
    imageUrl: '/src/assets/images/clay_robot_workshop_1794809575725.jpg',
    traits:
      'Sculpted stop-motion miniature copper-and-slate clockwork robot with glowing amber eyes',
    createdAt: 1700000011,
  },
];

export default function App() {
  const [activeView, setActiveView] = useState<WorkspaceView>('studio');
  const [leftTab, setLeftTab] = useState<LeftTab>('generate');
  const [showExportModal, setShowExportModal] = useState(false);

  const [assets, setAssets] = useState<MediaAsset[]>(SEED_MEDIA_ASSETS);
  const [characters, setCharacters] = useState<CharacterProfile[]>(INITIAL_CHARACTERS);
  const [selectedCharacterId, setSelectedCharacterId] = useState<string>('');
  const [builtInStems, setBuiltInStems] = useState<MediaAsset[]>([]);
  const [tracks, setTracks] = useState<TrackConfig[]>(INITIAL_TRACKS);
  const [clips, setClips] = useState<TimelineClip[]>(INITIAL_TIMELINE_CLIPS);

  const [selectedClipId, setSelectedClipId] = useState<string | null>('clip-v1');
  const [playheadTime, setPlayheadTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [showSubtitles, setShowSubtitles] = useState(true);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('16:9');
  const [defaultSubtitleStyle, setDefaultSubtitleStyle] =
    useState<SubtitleStyle>('cinematic');

  const [pendingVoiceLine, setPendingVoiceLine] = useState<string | undefined>();
  const [pendingMusicCue, setPendingMusicCue] = useState<string | undefined>();

  const monitorCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioEngineRef = useRef<TimelineAudioEngine | null>(null);
  const mediaCacheRef = useRef<MediaCache>({
    images: new Map(),
    videos: new Map(),
  });

  const totalDuration = useMemo(() => getProjectDuration(clips, 18), [clips]);

  // Pre-synthesize built-in Web Audio stems for immediate playback & drag-and-drop
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [orchUrl, cartoonUrl, noirUrl] = await Promise.all([
          synthesizeScoreStem('orchestral', 18),
          synthesizeScoreStem('cartoon', 16),
          synthesizeScoreStem('noir', 16),
        ]);
        if (cancelled) return;

        const stems: MediaAsset[] = [];
        if (orchUrl) {
          stems.push({
            id: 'stem-orchestral',
            title: 'Clockwork Horizon Score',
            kind: 'music',
            url: orchUrl,
            duration: 18,
            styleTag: 'audio',
            createdAt: 1700000101,
          });
        }
        if (cartoonUrl) {
          stems.push({
            id: 'stem-cartoon',
            title: 'Airship Pizzicato Capers',
            kind: 'music',
            url: cartoonUrl,
            duration: 16,
            styleTag: 'audio',
            createdAt: 1700000102,
          });
        }
        if (noirUrl) {
          stems.push({
            id: 'stem-noir',
            title: 'Midnight Rain Bassline',
            kind: 'music',
            url: noirUrl,
            duration: 16,
            styleTag: 'audio',
            createdAt: 1700000103,
          });
        }

        if (stems.length > 0) {
          setBuiltInStems(stems);
          setAssets((prev) => {
            const existingIds = new Set(prev.map((p) => p.id));
            return [...prev, ...stems.filter((s) => !existingIds.has(s.id))];
          });
          setClips((prev) => {
            if (prev.some((c) => c.track === 'music')) return prev;
            return [
              ...prev,
              {
                id: 'clip-m1',
                assetId: stems[0].id,
                track: 'music',
                title: stems[0].title,
                startTime: 0,
                duration: 18,
                mediaUrl: stems[0].url,
                mediaKind: 'music',
                visualStyle: 'original',
                transition: 'cut',
                transitionDuration: 0,
                volume: 0.45,
                fadeIn: 0.8,
                fadeOut: 1.2,
              },
            ];
          });
        }
      } catch {
        // OfflineAudioContext unavailable in minimal headless environments
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Playback loop
  useEffect(() => {
    if (!audioEngineRef.current) {
      audioEngineRef.current = new TimelineAudioEngine();
    }
    const engine = audioEngineRef.current;
    const mutedMap = Object.fromEntries(tracks.map((t) => [t.id, t.muted]));
    engine.sync(clips, playheadTime, isPlaying, mutedMap);
  }, [clips, playheadTime, isPlaying, tracks]);

  useEffect(() => {
    if (!isPlaying) return;
    let rafId = 0;
    let lastTs = performance.now();

    const tick = (now: number) => {
      const dt = (now - lastTs) / 1000;
      lastTs = now;
      setPlayheadTime((prev) => {
        const next = prev + dt;
        if (next >= totalDuration) {
          if (loop) return 0;
          setIsPlaying(false);
          return totalDuration;
        }
        return next;
      });
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, totalDuration, loop]);

  useEffect(() => {
    return () => {
      audioEngineRef.current?.dispose();
    };
  }, []);

  const handleSelectView = (view: WorkspaceView) => {
    setActiveView(view);
    if (view === 'voice') setLeftTab('voice');
    else if (view === 'subtitles') setLeftTab('subtitles');
    else if (view === 'export') setShowExportModal(true);
    else setShowExportModal(false);
  };

  const insertAssetToTimeline = useCallback(
    (asset: MediaAsset, targetTrack?: TrackId, explicitStart?: number) => {
      const resolvedTrack: TrackId =
        targetTrack ||
        (asset.kind === 'voice'
          ? 'voice'
          : asset.kind === 'music'
            ? 'music'
            : 'video');

      setClips((prev) => {
        const trackClips = prev.filter((c) => c.track === resolvedTrack);
        const endOfTrack = trackClips.reduce(
          (max, c) => Math.max(max, c.startTime + c.duration),
          0
        );
        const startTime =
          explicitStart !== undefined ? Math.max(0, explicitStart) : endOfTrack;

        const newClip: TimelineClip = {
          id: `clip-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
          assetId: asset.id,
          track: resolvedTrack,
          title: asset.title,
          startTime: Number(startTime.toFixed(2)),
          duration: asset.duration || 4.5,
          mediaUrl: asset.url,
          mediaKind: asset.kind,
          visualStyle:
            asset.styleTag === 'audio' ? 'original' : (asset.styleTag as VisualStyle),
          transition: resolvedTrack === 'video' ? 'crossfade' : 'cut',
          transitionDuration: resolvedTrack === 'video' ? 0.6 : 0,
          volume: resolvedTrack === 'music' ? 0.45 : 1,
          fadeIn: resolvedTrack === 'music' ? 0.5 : 0,
          fadeOut: resolvedTrack === 'music' ? 0.8 : 0,
          kenBurns: resolvedTrack === 'video' ? 'zoom-in' : 'static',
          ...(resolvedTrack === 'overlay'
            ? {
                overlayPosition: 'right' as const,
                overlayScale: 0.42,
                overlayFrame: 'portrait' as const,
              }
            : {}),
        };
        setSelectedClipId(newClip.id);
        return [...prev, newClip];
      });
    },
    []
  );

  const handleAddGeneratedAsset = (asset: MediaAsset, insertOnTimeline = true) => {
    setAssets((prev) => [asset, ...prev]);
    if (insertOnTimeline) {
      insertAssetToTimeline(asset);
    }
  };

  const handleAddVoiceAsset = (asset: MediaAsset, alsoCreateSubtitles: boolean) => {
    setAssets((prev) => [asset, ...prev]);
    const existingVoice = clips.filter((c) => c.track === 'voice');
    const startAt = existingVoice.reduce(
      (max, c) => Math.max(max, c.startTime + c.duration),
      0
    );
    insertAssetToTimeline(asset, 'voice', startAt);

    if (alsoCreateSubtitles && asset.transcript) {
      const cues = buildAutoSubtitleCues(
        asset.transcript,
        startAt,
        asset.duration || 4
      );
      handleInsertSubtitleCues(cues);
    }
  };

  const handleInsertSubtitleCues = (
    cues: Array<{ startTime: number; duration: number; text: string; speaker?: string }>
  ) => {
    setClips((prev) => {
      const newCues: TimelineClip[] = cues.map((cue, idx) => ({
        id: `sub-${Date.now()}-${idx}`,
        track: 'subtitle',
        title: `Cue ${idx + 1}`,
        startTime: Number(cue.startTime.toFixed(2)),
        duration: Math.max(1.2, Number(cue.duration.toFixed(2))),
        visualStyle: 'original',
        transition: 'cut',
        transitionDuration: 0,
        volume: 1,
        fadeIn: 0,
        fadeOut: 0,
        subtitleText: cue.text,
        subtitleSpeaker: cue.speaker || '',
        subtitleStyle: defaultSubtitleStyle,
      }));
      return [...prev, ...newCues];
    });
  };

  const handleUpdateClip = useCallback(
    (clipId: string, patch: Partial<TimelineClip>) => {
      setClips((prev) =>
        prev.map((c) => (c.id === clipId ? { ...c, ...patch } : c))
      );
    },
    []
  );

  const handleDeleteClip = useCallback(
    (clipId: string) => {
      setClips((prev) => prev.filter((c) => c.id !== clipId));
      if (selectedClipId === clipId) setSelectedClipId(null);
    },
    [selectedClipId]
  );

  const handleDuplicateClip = useCallback((clipId: string) => {
    setClips((prev) => {
      const target = prev.find((c) => c.id === clipId);
      if (!target) return prev;
      const clone: TimelineClip = {
        ...target,
        id: `clip-${Date.now()}-copy`,
        title: `${target.title} Copy`,
        startTime: Number((target.startTime + target.duration).toFixed(2)),
      };
      setSelectedClipId(clone.id);
      return [...prev, clone];
    });
  }, []);

  const handleSplitAtPlayhead = useCallback(
    (clipId: string) => {
      setClips((prev) =>
        splitClipAtTime(
          prev,
          clipId,
          playheadTime,
          () => `clip-${Date.now()}-${Math.floor(Math.random() * 1000)}`
        )
      );
    },
    [playheadTime]
  );

  const handleDropAsset = useCallback(
    (assetId: string, track: TrackId, startTime: number) => {
      const found = assets.find((a) => a.id === assetId);
      if (!found) return;
      insertAssetToTimeline(found, track, startTime);
    },
    [assets, insertAssetToTimeline]
  );

  // Global keyboard shortcuts (Spacebar play/pause, S split, Delete remove)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

      if (e.code === 'Space') {
        e.preventDefault();
        setIsPlaying((p) => !p);
      } else if (e.key.toLowerCase() === 's' && selectedClipId) {
        handleSplitAtPlayhead(selectedClipId);
      } else if (
        (e.key === 'Delete' || e.key === 'Backspace') &&
        selectedClipId
      ) {
        handleDeleteClip(selectedClipId);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [selectedClipId, handleSplitAtPlayhead, handleDeleteClip]);

  const selectedClip = clips.find((c) => c.id === selectedClipId) || null;

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#0E1013] text-[#F1F3F5]">
      <TopBar
        activeView={activeView}
        onSelectView={handleSelectView}
        onOpenExport={() => setShowExportModal(true)}
      />

      {/* Main Studio Workspace */}
      <main className="flex-1 min-h-0 flex overflow-hidden">
        {/* Left AI & Asset Dock */}
        <aside
          aria-label="Creation and Media Dock"
          className="w-[388px] shrink-0 border-r border-white/8 bg-[#14171D] flex flex-col overflow-hidden"
        >
          <div className="h-10 shrink-0 px-1.5 border-b border-white/8 bg-[#12151A] grid grid-cols-6 items-center gap-0.5">
            {LEFT_TABS.map(({ id, label, Icon }) => {
              const active = leftTab === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setLeftTab(id)}
                  className={`flex items-center justify-center gap-1 px-1 py-1.5 rounded text-[11px] font-medium transition-colors whitespace-nowrap ${
                    active
                      ? 'bg-[#222731] text-[#F1F3F5]'
                      : 'text-[#8E96A4] hover:text-[#F1F3F5]'
                  }`}
                >
                  <Icon className="w-3 h-3 shrink-0" />
                  <span>{label}</span>
                </button>
              );
            })}
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto">
            {leftTab === 'generate' && (
              <GeneratePanel
                visualAssets={assets.filter(
                  (a) => a.kind === 'image' || a.kind === 'video'
                )}
                characters={characters}
                selectedCharacterId={selectedCharacterId}
                onSelectCharacter={setSelectedCharacterId}
                onQuickUploadCharacter={(char) => {
                  setCharacters((prev) => [char, ...prev]);
                  setSelectedCharacterId(char.id);
                }}
                onOpenCastTab={() => setLeftTab('characters')}
                onAddAsset={handleAddGeneratedAsset}
                onQuickCreateAudioAndSub={(line, cue) => {
                  setPendingVoiceLine(line);
                  setPendingMusicCue(cue);
                  setLeftTab('voice');
                }}
              />
            )}
            {leftTab === 'characters' && (
              <CharacterCastPanel
                characters={characters}
                activeCharacterId={selectedCharacterId}
                onAddCharacter={(char) => {
                  setCharacters((prev) => [char, ...prev]);
                  setSelectedCharacterId(char.id);
                }}
                onUpdateCharacter={(id, patch) =>
                  setCharacters((prev) =>
                    prev.map((c) => (c.id === id ? { ...c, ...patch } : c))
                  )
                }
                onDeleteCharacter={(id) => {
                  setCharacters((prev) => prev.filter((c) => c.id !== id));
                  if (selectedCharacterId === id) setSelectedCharacterId('');
                }}
                onUseInGenerate={(id) => {
                  setSelectedCharacterId(id);
                  setLeftTab('generate');
                }}
                onInsertCharacterToOverlay={(char) => {
                  const charAsset: MediaAsset = {
                    id: `char-asset-${char.id}`,
                    title: char.name,
                    kind: 'image',
                    url: char.imageUrl,
                    duration: 5,
                    styleTag: 'original',
                    createdAt: Date.now(),
                  };
                  setAssets((prev) =>
                    prev.some((a) => a.id === charAsset.id)
                      ? prev
                      : [charAsset, ...prev]
                  );
                  insertAssetToTimeline(charAsset, 'overlay', playheadTime);
                }}
              />
            )}
            {leftTab === 'media' && (
              <MediaBinPanel
                assets={assets}
                onAddAssetToTimeline={(asset, track) =>
                  insertAssetToTimeline(asset, track)
                }
                onUploadLocalAsset={(asset) =>
                  handleAddGeneratedAsset(asset, false)
                }
              />
            )}
            {leftTab === 'voice' && (
              <VoiceStudioPanel
                initialLine={pendingVoiceLine}
                onAddVoiceAsset={handleAddVoiceAsset}
              />
            )}
            {leftTab === 'music' && (
              <MusicPanel
                initialCue={pendingMusicCue}
                visualAssets={assets.filter((a) => a.kind === 'image')}
                builtInStems={builtInStems}
                onAddMusicAsset={(asset, insert) =>
                  handleAddGeneratedAsset(asset, insert)
                }
              />
            )}
            {leftTab === 'subtitles' && (
              <SubtitlePanel
                clips={clips}
                audioAssets={assets.filter(
                  (a) => a.kind === 'voice' || a.kind === 'music'
                )}
                defaultSubtitleStyle={defaultSubtitleStyle}
                onChangeDefaultStyle={(style) => {
                  setDefaultSubtitleStyle(style);
                  setClips((prev) =>
                    prev.map((c) =>
                      c.track === 'subtitle' ? { ...c, subtitleStyle: style } : c
                    )
                  );
                }}
                onUpdateClip={handleUpdateClip}
                onDeleteClip={handleDeleteClip}
                onInsertSubtitleCues={handleInsertSubtitleCues}
                onSeek={(t) => {
                  setIsPlaying(false);
                  setPlayheadTime(t);
                }}
              />
            )}
          </div>
        </aside>

        {/* Center Program Monitor */}
        <ProgramMonitor
          clips={clips}
          currentTime={playheadTime}
          totalDuration={totalDuration}
          isPlaying={isPlaying}
          loop={loop}
          showSubtitles={showSubtitles}
          aspectRatio={aspectRatio}
          mediaCache={mediaCacheRef.current}
          onTogglePlay={() => {
            if (!isPlaying && playheadTime >= totalDuration) {
              setPlayheadTime(0);
            }
            setIsPlaying((p) => !p);
          }}
          onSeek={(t) => setPlayheadTime(Math.max(0, Math.min(totalDuration, t)))}
          onToggleLoop={() => setLoop((l) => !l)}
          onToggleSubtitles={() => setShowSubtitles((s) => !s)}
          onChangeAspectRatio={setAspectRatio}
          onCanvasReady={(canvas) => {
            monitorCanvasRef.current = canvas;
          }}
        />

        {/* Right Clip Inspector */}
        <InspectorPanel
          selectedClip={selectedClip}
          playheadTime={playheadTime}
          totalClips={clips.length}
          totalDuration={totalDuration}
          onUpdateClip={handleUpdateClip}
          onSplitAtPlayhead={handleSplitAtPlayhead}
          onDuplicateClip={handleDuplicateClip}
          onDeleteClip={handleDeleteClip}
        />
      </main>

      {/* Bottom Multi-Track Timeline */}
      <TimelineDock
        tracks={tracks}
        clips={clips}
        selectedClipId={selectedClipId}
        playheadTime={playheadTime}
        totalDuration={totalDuration}
        snapEnabled={snapEnabled}
        onToggleSnap={() => setSnapEnabled((s) => !s)}
        onToggleTrackMute={(trackId) =>
          setTracks((prev) =>
            prev.map((t) => (t.id === trackId ? { ...t, muted: !t.muted } : t))
          )
        }
        onSelectClip={setSelectedClipId}
        onSeek={(t) => setPlayheadTime(t)}
        onMoveClip={(clipId, newStart) =>
          setClips((prev) =>
            moveClip(prev, clipId, newStart, playheadTime, snapEnabled)
          )
        }
        onResizeClip={(clipId, newDur) =>
          setClips((prev) => resizeClip(prev, clipId, newDur))
        }
        onSplitSelected={() => {
          if (selectedClipId) handleSplitAtPlayhead(selectedClipId);
        }}
        onDropAsset={handleDropAsset}
        assets={assets}
      />

      {showExportModal && (
        <ExportView
          clips={clips}
          assets={assets}
          totalDuration={totalDuration}
          aspectRatio={aspectRatio}
          canvas={monitorCanvasRef.current}
          onStartExportPlayback={() => {
            setPlayheadTime(0);
            setIsPlaying(true);
          }}
          onStopExportPlayback={() => setIsPlaying(false)}
          onClose={() => {
            setShowExportModal(false);
            if (activeView === 'export') setActiveView('studio');
          }}
        />
      )}
    </div>
  );
}
