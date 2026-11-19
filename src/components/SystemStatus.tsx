import { useCallback, useEffect, useState } from 'react';
import { Activity, CircleAlert, LoaderCircle } from 'lucide-react';

interface SystemHealth {
  ok: boolean;
  hasApiKey: boolean;
  localDiffusion: {
    ready: boolean;
    state: 'idle' | 'loading' | 'ready' | 'error';
    weightsPresent: boolean;
  };
  pythonService: {
    available: boolean;
    gpuAvailable: boolean;
    gpuCount: number;
    models: string[];
    loadedModels: string[];
  };
}

export function SystemStatus() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch('/api/health', {
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) {
        throw new Error(`Health check failed (${response.status}).`);
      }

      const result = (await response.json()) as SystemHealth;
      if (!result.ok) {
        throw new Error('The application server reported an unhealthy state.');
      }
      setHealth(result);
      setError(null);
    } catch (reason) {
      setHealth(null);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const apiOnline = health?.ok === true;
  const systemReady =
    apiOnline &&
    health.pythonService.available &&
    health.localDiffusion.weightsPresent;
  const summary = error ? 'API offline' : systemReady ? 'System ready' : apiOnline ? 'Partially ready' : 'Checking';
  const summaryColor = error
    ? 'text-rose-300'
    : systemReady
      ? 'text-emerald-300'
      : 'text-amber-300';

  const localModelStatus = health
    ? health.localDiffusion.state === 'ready'
      ? 'Ready'
      : health.localDiffusion.state === 'loading'
        ? health.localDiffusion.weightsPresent
          ? 'Loading model'
          : 'Downloading weights'
        : health.localDiffusion.state === 'error'
          ? 'Initialization failed'
          : health.localDiffusion.weightsPresent
            ? 'Weights cached · idle'
            : 'Not downloaded'
    : 'Unknown';

  return (
    <details className="relative">
      <summary
        className={`flex cursor-pointer list-none items-center gap-1.5 rounded-md border border-white/10 px-2.5 py-2 text-xs font-medium ${summaryColor} hover:bg-white/5 focus-visible:outline-2 focus-visible:outline-[#E5A93C]`}
        aria-label={`System status: ${summary}`}
      >
        {error ? (
          <CircleAlert className="h-3.5 w-3.5" />
        ) : health ? (
          <Activity className="h-3.5 w-3.5" />
        ) : (
          <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
        )}
        <span>{summary}</span>
      </summary>
      <section
        aria-label="System status details"
        className="absolute right-0 top-full z-50 mt-2 w-72 rounded-lg border border-white/10 bg-[#181C23] p-3 text-xs shadow-xl"
      >
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-semibold text-[#F1F3F5]">System status</h2>
          <button
            type="button"
            onClick={() => void refresh()}
            className="text-[#E5A93C] hover:text-[#F1C46B] focus-visible:outline-2 focus-visible:outline-[#E5A93C]"
          >
            Refresh
          </button>
        </div>
        <ul className="space-y-2 text-[#C4C9D4]">
          <StatusRow label="Studio UI" value="Running" ready />
          <StatusRow label="Application API" value={apiOnline ? 'Online' : 'Offline'} ready={apiOnline} />
          <StatusRow
            label="Gemini"
            value={health ? (health.hasApiKey ? 'Configured' : 'No API key') : 'Unknown'}
            ready={health?.hasApiKey === true}
          />
          <StatusRow
            label="Local SDXS-512"
            value={localModelStatus}
            ready={health?.localDiffusion.ready === true}
          />
          <StatusRow
            label="Python inference"
            value={
              health?.pythonService.available
                ? health.pythonService.gpuAvailable
                  ? `Online · ${health.pythonService.gpuCount} GPU(s)`
                  : 'Online · CPU'
                : health
                  ? 'Offline'
                  : 'Unknown'
            }
            ready={health?.pythonService.available === true}
          />
        </ul>
        {error && <p className="mt-2 break-words text-rose-300">{error}</p>}
      </section>
    </details>
  );
}

function StatusRow({
  label,
  value,
  ready,
}: {
  label: string;
  value: string;
  ready: boolean;
}) {
  return (
    <li className="flex items-center justify-between gap-3">
      <span>{label}</span>
      <span className={ready ? 'text-emerald-300' : 'text-amber-300'}>{value}</span>
    </li>
  );
}
