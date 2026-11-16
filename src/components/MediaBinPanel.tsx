import { useState } from 'react';
import { Film, Image as ImageIcon, Mic, Music, Plus, Upload } from 'lucide-react';
import { MediaAsset, MediaKind, TrackId } from '../types/studio';

interface MediaBinPanelProps {
  assets: MediaAsset[];
  onAddAssetToTimeline: (asset: MediaAsset, track?: TrackId) => void;
  onUploadLocalAsset: (asset: MediaAsset) => void;
}

type FilterKind = 'all' | 'visual' | 'voice' | 'music';

const STYLE_LABELS: Record<string, string> = {
  cinematic: 'Cinematic',
  cartoon2d: '2D Cartoon',
  anime: 'Anime',
  claymation: 'Claymation',
  pixar3d: '3D Animated',
  watercolor: 'Watercolor',
  noir: 'Film Noir',
  original: 'Natural',
};

export function MediaBinPanel({
  assets,
  onAddAssetToTimeline,
  onUploadLocalAsset,
}: MediaBinPanelProps) {
  const [filter, setFilter] = useState<FilterKind>('all');
  const [query, setQuery] = useState('');

  const filtered = assets.filter((a) => {
    const matchesKind =
      filter === 'all' ||
      (filter === 'visual' && (a.kind === 'image' || a.kind === 'video')) ||
      a.kind === filter;
    const matchesQuery =
      !query.trim() ||
      a.title.toLowerCase().includes(query.toLowerCase()) ||
      String(a.styleTag).toLowerCase().includes(query.toLowerCase());
    return matchesKind && matchesQuery;
  });

  const handleDragStart = (e: React.DragEvent, asset: MediaAsset) => {
    e.dataTransfer.setData(
      'application/json',
      JSON.stringify({ type: 'asset', assetId: asset.id })
    );
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    let kind: MediaKind = 'image';
    if (file.type.startsWith('video/')) kind = 'video';
    else if (file.type.startsWith('audio/')) kind = 'music';

    const cleanTitle = file.name.replace(/\.[^/.]+$/, '').slice(0, 28);
    if (kind === 'music') {
      const audio = new Audio(url);
      audio.onloadedmetadata = () => {
        onUploadLocalAsset({
          id: `local-${Date.now()}`,
          title: cleanTitle,
          kind,
          url,
          duration:
            Number.isFinite(audio.duration) && audio.duration > 0
              ? Number(audio.duration.toFixed(1))
              : 12,
          styleTag: 'audio',
          createdAt: Date.now(),
        });
      };
      audio.onerror = () => {
        onUploadLocalAsset({
          id: `local-${Date.now()}`,
          title: cleanTitle,
          kind,
          url,
          duration: 12,
          styleTag: 'audio',
          createdAt: Date.now(),
        });
      };
      return;
    }

    onUploadLocalAsset({
      id: `local-${Date.now()}`,
      title: cleanTitle,
      kind,
      url,
      duration: kind === 'image' ? 4.5 : 8,
      styleTag: 'original',
      createdAt: Date.now(),
    });
  };

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between gap-2">
        <input
          type="search"
          aria-label="Search media bin"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter clips & audio..."
          className="flex-1 bg-[#0E1013] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] placeholder:text-[#5F6674] focus:outline-none focus:border-[#E5A93C]"
        />
        <label className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-[#1D2129] border border-white/10 text-xs font-medium text-[#D1D5DB] hover:bg-[#262B35] cursor-pointer whitespace-nowrap shrink-0">
          <Upload className="w-3.5 h-3.5" />
          <span>Import</span>
          <input
            type="file"
            accept="image/*,video/*,audio/*"
            onChange={handleFileUpload}
            className="sr-only"
          />
        </label>
      </div>

      <div className="grid grid-cols-4 gap-1 p-1 bg-[#0E1013] rounded-lg border border-white/8">
        {(
          [
            { id: 'all', label: 'All' },
            { id: 'visual', label: 'Visuals' },
            { id: 'voice', label: 'Voice' },
            { id: 'music', label: 'Music' },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setFilter(tab.id)}
            className={`py-1 rounded text-xs font-medium transition-colors whitespace-nowrap ${
              filter === tab.id
                ? 'bg-[#222731] text-[#F1F3F5]'
                : 'text-[#8E96A4] hover:text-[#F1F3F5]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <p className="text-xs text-[#7D8594]">
        Drag any item onto a timeline track, or click Add.
      </p>

      {filtered.length === 0 ? (
        <div className="py-8 text-center text-xs text-[#7D8594] border border-dashed border-white/10 rounded-md">
          No assets match your filter.
        </div>
      ) : (
        <div className="flex flex-col gap-2.5">
          {filtered.map((asset) => {
            const targetTrack: TrackId =
              asset.kind === 'voice'
                ? 'voice'
                : asset.kind === 'music'
                  ? 'music'
                  : 'video';

            return (
              <div
                key={asset.id}
                draggable
                onDragStart={(e) => handleDragStart(e, asset)}
                className="group flex items-center gap-3 p-2 rounded-lg bg-[#12151B] border border-white/8 hover:border-[#E5A93C]/50 cursor-grab active:cursor-grabbing transition-colors"
              >
                <div className="w-16 h-10 rounded overflow-hidden bg-[#0A0C0F] shrink-0 flex items-center justify-center border border-white/6">
                  {asset.kind === 'image' ? (
                    <img
                      src={asset.url}
                      alt={asset.title}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                  ) : asset.kind === 'video' ? (
                    <Film className="w-4 h-4 text-[#E5A93C]" />
                  ) : asset.kind === 'voice' ? (
                    <Mic className="w-4 h-4 text-[#60A5FA]" />
                  ) : (
                    <Music className="w-4 h-4 text-[#34D399]" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-[#F1F3F5] truncate">
                    {asset.title}
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-[#8E96A4] mt-0.5">
                    <span className="capitalize">{asset.kind}</span>
                    <span aria-hidden="true">·</span>
                    <span className="font-mono">{asset.duration.toFixed(1)}s</span>
                    {asset.styleTag !== 'audio' && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="truncate">
                          {STYLE_LABELS[asset.styleTag] || asset.styleTag}
                        </span>
                      </>
                    )}
                  </div>
                </div>

                <button
                  type="button"
                  aria-label={`Add ${asset.title} to timeline`}
                  onClick={() => onAddAssetToTimeline(asset, targetTrack)}
                  className="p-1.5 rounded-md bg-[#1D2129] text-[#E5A93C] hover:bg-[#E5A93C] hover:text-[#111317] transition-colors shrink-0"
                >
                  <Plus className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
