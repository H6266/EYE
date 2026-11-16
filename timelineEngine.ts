import { TimelineClip } from '../types/studio';

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function formatTimecode(seconds: number, fps = 24): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const frames = Math.floor((safe - Math.floor(safe)) * fps);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}:${String(frames).padStart(2, '0')}`;
}

export function formatSrtTimestamp(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const hrs = Math.floor(safe / 3600);
  const mins = Math.floor((safe % 3600) / 60);
  const secs = Math.floor(safe % 60);
  const ms = Math.round((safe - Math.floor(safe)) * 1000);
  const clampedMs = ms >= 1000 ? 999 : ms;
  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(clampedMs).padStart(3, '0')}`;
}

export function formatVttTimestamp(seconds: number): string {
  return formatSrtTimestamp(seconds).replace(',', '.');
}

export function getProjectDuration(clips: TimelineClip[], minDuration = 18): number {
  if (clips.length === 0) return minDuration;
  const maxEnd = clips.reduce((max, c) => Math.max(max, c.startTime + c.duration), 0);
  return Math.max(minDuration, Number(maxEnd.toFixed(2)));
}

export interface ActiveVideoState {
  primary: TimelineClip | null;
  outgoing: TimelineClip | null;
  blend: number; // 0 = outgoing only, 1 = primary only
}

export function getActiveVideoLayer(clips: TimelineClip[], time: number): ActiveVideoState {
  const videoClips = clips
    .filter((c) => c.track === 'video')
    .sort((a, b) => a.startTime - b.startTime);

  const active = videoClips.filter(
    (c) => time >= c.startTime && time < c.startTime + c.duration
  );

  if (active.length === 0) {
    return { primary: null, outgoing: null, blend: 1 };
  }

  const primary = active[active.length - 1];
  const elapsed = time - primary.startTime;

  if (
    primary.transition !== 'cut' &&
    primary.transitionDuration > 0 &&
    elapsed < primary.transitionDuration
  ) {
    // Find immediately preceding clip or overlapping clip
    const idx = videoClips.findIndex((c) => c.id === primary.id);
    const prev = idx > 0 ? videoClips[idx - 1] : null;
    if (prev && time <= prev.startTime + prev.duration + primary.transitionDuration) {
      const blend = clamp(elapsed / primary.transitionDuration, 0, 1);
      return { primary, outgoing: prev, blend };
    }
  }

  return { primary, outgoing: null, blend: 1 };
}

export function getActiveOverlayClips(clips: TimelineClip[], time: number): TimelineClip[] {
  return clips
    .filter(
      (c) =>
        c.track === 'overlay' &&
        time >= c.startTime &&
        time < c.startTime + c.duration &&
        Boolean(c.mediaUrl)
    )
    .sort((a, b) => a.startTime - b.startTime);
}

export function getActiveSubtitles(clips: TimelineClip[], time: number): TimelineClip[] {
  return clips
    .filter(
      (c) =>
        c.track === 'subtitle' &&
        time >= c.startTime &&
        time < c.startTime + c.duration &&
        Boolean(c.subtitleText?.trim())
    )
    .sort((a, b) => a.startTime - b.startTime);
}

export function getActiveAudioClips(clips: TimelineClip[], time: number): TimelineClip[] {
  return clips.filter(
    (c) =>
      (c.track === 'voice' || c.track === 'music') &&
      time >= c.startTime &&
      time < c.startTime + c.duration &&
      Boolean(c.mediaUrl)
  );
}

export function computeClipGain(clip: TimelineClip, time: number): number {
  const base = clamp(clip.volume, 0, 1.5);
  const elapsed = time - clip.startTime;
  const remaining = clip.startTime + clip.duration - time;
  if (elapsed < 0 || remaining < 0) return 0;

  let env = 1;
  if (clip.fadeIn > 0 && elapsed < clip.fadeIn) {
    env = Math.min(env, clamp(elapsed / clip.fadeIn, 0, 1));
  }
  if (clip.fadeOut > 0 && remaining < clip.fadeOut) {
    env = Math.min(env, clamp(remaining / clip.fadeOut, 0, 1));
  }
  return base * env;
}

export function snapStartTime(
  clips: TimelineClip[],
  clipId: string,
  proposedStart: number,
  playheadTime: number,
  snapThreshold = 0.25
): number {
  const safeProposed = Math.max(0, Number(proposedStart.toFixed(2)));
  const movingClip = clips.find((c) => c.id === clipId);
  if (!movingClip) return safeProposed;

  const candidates: number[] = [0, playheadTime];
  for (const c of clips) {
    if (c.id === clipId || c.track !== movingClip.track) continue;
    candidates.push(c.startTime);
    candidates.push(c.startTime + c.duration);
  }

  let best = safeProposed;
  let bestDist = snapThreshold;

  for (const target of candidates) {
    const dist = Math.abs(safeProposed - target);
    if (dist < bestDist) {
      bestDist = dist;
      best = target;
    }
    const endDist = Math.abs(safeProposed + movingClip.duration - target);
    if (endDist < bestDist) {
      bestDist = endDist;
      best = Math.max(0, target - movingClip.duration);
    }
  }

  return Math.max(0, Number(best.toFixed(2)));
}

export function moveClip(
  clips: TimelineClip[],
  clipId: string,
  newStartTime: number,
  playheadTime = 0,
  enableSnap = true
): TimelineClip[] {
  const snapped = enableSnap
    ? snapStartTime(clips, clipId, newStartTime, playheadTime)
    : Math.max(0, Number(newStartTime.toFixed(2)));

  return clips.map((c) => (c.id === clipId ? { ...c, startTime: snapped } : c));
}

export function resizeClip(
  clips: TimelineClip[],
  clipId: string,
  newDuration: number,
  minDuration = 0.5
): TimelineClip[] {
  const safeDuration = Math.max(minDuration, Number(newDuration.toFixed(2)));
  return clips.map((c) => (c.id === clipId ? { ...c, duration: safeDuration } : c));
}

export function splitClipAtTime(
  clips: TimelineClip[],
  clipId: string,
  splitTime: number,
  newIdFactory: () => string
): TimelineClip[] {
  const target = clips.find((c) => c.id === clipId);
  if (!target) return clips;

  const offset = splitTime - target.startTime;
  if (offset <= 0.3 || target.duration - offset <= 0.3) {
    return clips;
  }

  const firstHalf: TimelineClip = {
    ...target,
    duration: Number(offset.toFixed(2)),
  };

  const secondHalf: TimelineClip = {
    ...target,
    id: newIdFactory(),
    title: `${target.title} (Part 2)`,
    startTime: Number(splitTime.toFixed(2)),
    duration: Number((target.duration - offset).toFixed(2)),
  };

  return clips.flatMap((c) => (c.id === clipId ? [firstHalf, secondHalf] : [c]));
}

export function exportSrt(clips: TimelineClip[]): string {
  const subs = clips
    .filter((c) => c.track === 'subtitle' && c.subtitleText?.trim())
    .sort((a, b) => a.startTime - b.startTime);

  return subs
    .map((cue, index) => {
      const start = formatSrtTimestamp(cue.startTime);
      const end = formatSrtTimestamp(cue.startTime + cue.duration);
      const speakerPrefix = cue.subtitleSpeaker?.trim()
        ? `${cue.subtitleSpeaker.trim()}: `
        : '';
      return `${index + 1}\n${start} --> ${end}\n${speakerPrefix}${cue.subtitleText?.trim()}`;
    })
    .join('\n\n');
}

export function exportVtt(clips: TimelineClip[]): string {
  const subs = clips
    .filter((c) => c.track === 'subtitle' && c.subtitleText?.trim())
    .sort((a, b) => a.startTime - b.startTime);

  const body = subs
    .map((cue) => {
      const start = formatVttTimestamp(cue.startTime);
      const end = formatVttTimestamp(cue.startTime + cue.duration);
      const speakerPrefix = cue.subtitleSpeaker?.trim()
        ? `<v ${cue.subtitleSpeaker.trim()}>`
        : '';
      return `${start} --> ${end}\n${speakerPrefix}${cue.subtitleText?.trim()}`;
    })
    .join('\n\n');

  return `WEBVTT\n\n${body}`;
}

export interface GeneratedCue {
  startTime: number;
  duration: number;
  text: string;
  speaker?: string;
}

export function buildAutoSubtitleCues(
  text: string,
  startTime: number,
  totalDuration: number,
  speaker?: string
): GeneratedCue[] {
  const cleaned = text.replace(/<[^>]+>/g, '').replace(/\|[^|]+\|/g, '').trim();
  if (!cleaned) return [];

  const sentences = cleaned
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter(Boolean);

  if (sentences.length === 0) return [];

  const totalChars = sentences.reduce((acc, s) => acc + s.length, 0) || 1;
  const safeTotalDur = Math.max(1.5, totalDuration);

  let cursor = Math.max(0, startTime);
  return sentences.map((sentence) => {
    const proportion = sentence.length / totalChars;
    const dur = Math.max(1.2, Number((safeTotalDur * proportion).toFixed(2)));
    const cue: GeneratedCue = {
      startTime: Number(cursor.toFixed(2)),
      duration: dur,
      text: sentence,
      speaker,
    };
    cursor += dur;
    return cue;
  });
}
