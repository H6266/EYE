import { useState } from 'react';
import { Loader2, Layers, Sparkles, Trash2, Upload, UserPlus } from 'lucide-react';
import { CharacterProfile } from '../types/studio';

interface CharacterCastPanelProps {
  characters: CharacterProfile[];
  activeCharacterId: string;
  onAddCharacter: (character: CharacterProfile) => void;
  onUpdateCharacter: (id: string, patch: Partial<CharacterProfile>) => void;
  onDeleteCharacter: (id: string) => void;
  onUseInGenerate: (characterId: string) => void;
  onInsertCharacterToOverlay: (character: CharacterProfile) => void;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('Failed to read character image file'));
    reader.readAsDataURL(file);
  });
}

export function CharacterCastPanel({
  characters,
  activeCharacterId,
  onAddCharacter,
  onUpdateCharacter,
  onDeleteCharacter,
  onUseInGenerate,
  onInsertCharacterToOverlay,
}: CharacterCastPanelProps) {
  const [name, setName] = useState('');
  const [traits, setTraits] = useState('');
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleUploadCharacter = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    setStatusMsg(null);

    try {
      const dataUrl = await fileToDataUrl(file);
      const cleanName =
        name.trim() || file.name.replace(/\.[^/.]+$/, '').slice(0, 24) || 'Hero Character';
      const newChar: CharacterProfile = {
        id: `char-${Date.now()}`,
        name: cleanName,
        imageUrl: dataUrl,
        traits:
          traits.trim() ||
          'Expressive protagonist portrait ready for IP-Adapter & Meta Chameleon scene consistency',
        createdAt: Date.now(),
      };
      onAddCharacter(newChar);
      setName('');
      setTraits('');
      setStatusMsg(`Added "${newChar.name}" to Character Cast.`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Could not load image file.');
    } finally {
      e.target.value = '';
    }
  };

  const handleAnalyzeTraits = async (char: CharacterProfile) => {
    setErrorMsg(null);
    setStatusMsg(null);
    setAnalyzingId(char.id);
    try {
      let base64 = '';
      let mimeType = 'image/jpeg';
      if (char.imageUrl.startsWith('data:')) {
        const [header, payload] = char.imageUrl.split(',');
        mimeType = header.match(/data:([^;]+)/)?.[1] || 'image/jpeg';
        base64 = payload || '';
      } else {
        const res = await fetch(char.imageUrl);
        const blob = await res.blob();
        mimeType = blob.type || 'image/jpeg';
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const r = new FileReader();
          r.onloadend = () => resolve(String(r.result || ''));
          r.onerror = () => reject(new Error('Failed to encode character image'));
          r.readAsDataURL(blob);
        });
        base64 = dataUrl.split(',')[1] || '';
      }

      const apiRes = await fetch('/api/characters/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: base64,
          mimeType,
          name: char.name,
        }),
      });
      const data = await apiRes.json();
      if (!apiRes.ok) {
        setErrorMsg(data.error || 'Vision analysis unavailable.');
        return;
      }
      if (data.traits) {
        onUpdateCharacter(char.id, { traits: data.traits });
        setStatusMsg(`Extracted visual traits for ${char.name}.`);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to analyze photo.');
    } finally {
      setAnalyzingId(null);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4 text-sm">
      <div className="flex flex-col gap-2.5 p-3 rounded-lg bg-[#0E1013] border border-white/8">
        <div className="flex items-center gap-2 text-xs font-semibold text-[#F1F3F5]">
          <UserPlus className="w-3.5 h-3.5 text-[#E5A93C]" />
          <span>Upload Character Portrait</span>
        </div>

        <input
          type="text"
          aria-label="Character name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Character name (e.g. Captain Vega)"
          className="bg-[#14171D] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] placeholder:text-[#5F6674] focus:outline-none focus:border-[#E5A93C]"
        />

        <input
          type="text"
          aria-label="Character visual traits"
          value={traits}
          onChange={(e) => setTraits(e.target.value)}
          placeholder="Optional look notes: silver jacket, brass goggles..."
          className="bg-[#14171D] border border-white/10 rounded-md px-2.5 py-1.5 text-xs text-[#F1F3F5] placeholder:text-[#5F6674] focus:outline-none focus:border-[#E5A93C]"
        />

        <label className="w-full py-2 px-3 rounded-md bg-[#E5A93C] text-[#111317] text-xs font-semibold hover:bg-[#EEB955] transition-colors flex items-center justify-center gap-2 cursor-pointer">
          <Upload className="w-3.5 h-3.5" />
          <span>Upload Character Picture</span>
          <input
            type="file"
            accept="image/*"
            onChange={handleUploadCharacter}
            className="sr-only"
          />
        </label>
        <p className="text-[11px] text-[#7D8594] leading-relaxed">
          Uploaded character photos condition open-source Kwai-Kolors IP-Adapter scenes and can be placed directly onto the V2 Character Layer.
        </p>
      </div>

      {errorMsg && (
        <div
          role="alert"
          className="p-2.5 rounded-md bg-[#DC2626]/12 border border-[#DC2626]/40 text-xs text-[#FCA5A5]"
        >
          {errorMsg}
        </div>
      )}

      {statusMsg && (
        <div className="p-2.5 rounded-md bg-[#16A34A]/12 border border-[#16A34A]/40 text-xs text-[#86EFAC]">
          {statusMsg}
        </div>
      )}

      <div className="flex flex-col gap-2.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-[#F1F3F5]">Cast Roster</span>
          <span className="font-mono text-[#8E96A4]">{characters.length} cast</span>
        </div>

        {characters.map((char) => {
          const isActive = char.id === activeCharacterId;
          return (
            <div
              key={char.id}
              className={`p-2.5 rounded-lg border flex flex-col gap-2.5 transition-colors ${
                isActive
                  ? 'bg-[#171B24] border-[#E5A93C]/70'
                  : 'bg-[#11141A] border-white/8'
              }`}
            >
              <div className="flex items-start gap-2.5">
                <img
                  src={char.imageUrl}
                  alt={char.name}
                  referrerPolicy="no-referrer"
                  className="w-12 h-12 rounded-md object-cover border border-white/10 shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-[#F1F3F5] truncate">
                      {char.name}
                    </span>
                    <button
                      type="button"
                      aria-label={`Remove ${char.name}`}
                      onClick={() => onDeleteCharacter(char.id)}
                      className="text-[#6B7280] hover:text-[#FCA5A5] shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <textarea
                    rows={2}
                    aria-label={`${char.name} traits`}
                    value={char.traits}
                    onChange={(e) =>
                      onUpdateCharacter(char.id, { traits: e.target.value })
                    }
                    className="mt-1 w-full bg-[#0E1013] border border-white/8 rounded px-2 py-1 text-[11px] text-[#C4C9D4] focus:outline-none focus:border-[#E5A93C] resize-none"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-1.5">
                <button
                  type="button"
                  onClick={() => onUseInGenerate(char.id)}
                  className="py-1.5 px-2 rounded bg-[#E5A93C]/15 border border-[#E5A93C]/40 text-[11px] font-medium text-[#FDE68A] hover:bg-[#E5A93C]/25 flex items-center justify-center gap-1 whitespace-nowrap"
                >
                  <Sparkles className="w-3 h-3 shrink-0" />
                  <span>Use in Scene</span>
                </button>

                <button
                  type="button"
                  onClick={() => onInsertCharacterToOverlay(char)}
                  className="py-1.5 px-2 rounded bg-[#1D2129] border border-white/10 text-[11px] font-medium text-[#E5ECF6] hover:border-[#A855F7]/60 flex items-center justify-center gap-1 whitespace-nowrap"
                >
                  <Layers className="w-3 h-3 shrink-0" />
                  <span>+ V2 Layer</span>
                </button>

                <button
                  type="button"
                  disabled={analyzingId === char.id}
                  onClick={() => handleAnalyzeTraits(char)}
                  className="py-1.5 px-2 rounded bg-[#1D2129] border border-white/10 text-[11px] font-medium text-[#9BA1AE] hover:text-[#F1F3F5] disabled:opacity-50 flex items-center justify-center gap-1 whitespace-nowrap"
                >
                  {analyzingId === char.id ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <span>Extract Look</span>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
