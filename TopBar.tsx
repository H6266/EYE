import { WorkspaceView } from '../types/studio';

interface TopBarProps {
  activeView: WorkspaceView;
  onSelectView: (view: WorkspaceView) => void;
  onOpenExport: () => void;
}

const NAV_ITEMS: Array<{ id: WorkspaceView; label: string }> = [
  { id: 'studio', label: 'Studio' },
  { id: 'voice', label: 'Voice Studio' },
  { id: 'subtitles', label: 'Subtitles' },
  { id: 'export', label: 'Export' },
];

export function TopBar({ activeView, onSelectView, onOpenExport }: TopBarProps) {
  return (
    <header className="h-14 shrink-0 border-b border-white/8 bg-[#12151A] px-6 flex items-center justify-between gap-8">
      <button
        type="button"
        onClick={() => onSelectView('studio')}
        className="font-display text-lg font-bold tracking-tight text-[#F1F3F5] whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#E5A93C] rounded"
      >
        Auteur Studio
      </button>

      <nav aria-label="Workspace Navigation" className="hidden md:flex items-center gap-6">
        {NAV_ITEMS.map((item) => {
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectView(item.id)}
              className={`text-sm font-medium py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus-visible:outline-2 focus-visible:outline-[#E5A93C] ${
                isActive
                  ? 'text-[#F1F3F5] border-[#E5A93C]'
                  : 'text-[#9BA1AE] border-transparent hover:text-[#F1F3F5]'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="flex items-center shrink-0">
        <button
          type="button"
          onClick={onOpenExport}
          className="px-4 py-2 text-xs font-semibold bg-[#E5A93C] text-[#111317] rounded-md hover:bg-[#EEB955] transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-white"
        >
          Export Movie
        </button>
      </div>
    </header>
  );
}
