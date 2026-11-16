import { TimelineClip } from '../types/studio';
import { computeClipGain } from './timelineEngine';

/**
 * Encodes an AudioBuffer into a standard 16-bit PCM WAV Blob URL.
 */
function audioBufferToWavUrl(buffer: AudioBuffer): string {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const samples = buffer.getChannelData(0);
  const byteLength = samples.length * 2;
  const arrayBuffer = new ArrayBuffer(44 + byteLength);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + byteLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, byteLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
}

export type ScorePreset = 'orchestral' | 'cartoon' | 'noir';

/**
 * Synthesizes clean procedural studio score stems via OfflineAudioContext
 * so the built-in Media Bin and demo sequence have real, playable music tracks.
 */
export async function synthesizeScoreStem(
  preset: ScorePreset,
  durationSeconds = 18
): Promise<string> {
  if (typeof window === 'undefined' || !window.OfflineAudioContext) {
    return '';
  }

  const sampleRate = 22050;
  const length = Math.floor(sampleRate * durationSeconds);
  const ctx = new window.OfflineAudioContext(1, length, sampleRate);
  const master = ctx.createGain();
  master.gain.value = 0.28;
  master.connect(ctx.destination);

  if (preset === 'orchestral') {
    const chords = [
      [130.81, 196.0, 261.63, 329.63], // Cmaj
      [110.0, 164.81, 220.0, 261.63], // Am
      [87.31, 174.61, 220.0, 261.63], // Fmaj
      [98.0, 196.0, 246.94, 293.66], // Gmaj
    ];
    const step = 4.5;
    chords.forEach((chord, idx) => {
      const start = idx * step;
      if (start >= durationSeconds) return;
      chord.forEach((freq, voiceIdx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = voiceIdx === 0 ? 'triangle' : 'sine';
        osc.frequency.setValueAtTime(freq, start);
        gain.gain.setValueAtTime(0.001, start);
        gain.gain.linearRampToValueAtTime(voiceIdx === 0 ? 0.35 : 0.16, start + 0.8);
        gain.gain.exponentialRampToValueAtTime(0.01, Math.min(durationSeconds, start + step));
        osc.connect(gain);
        gain.connect(master);
        osc.start(start);
        osc.stop(Math.min(durationSeconds, start + step));
      });
    });
  } else if (preset === 'cartoon') {
    const melody = [392, 440, 493.88, 523.25, 659.25, 587.33, 523.25, 392];
    const beat = 0.35;
    for (let t = 0, i = 0; t < durationSeconds - 0.3; t += beat, i++) {
      const freq = melody[i % melody.length];
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(freq, t);
      osc.frequency.exponentialRampToValueAtTime(freq * 1.03, t + 0.08);
      gain.gain.setValueAtTime(0.28, t);
      gain.gain.exponentialRampToValueAtTime(0.005, t + 0.24);
      osc.connect(gain);
      gain.connect(master);
      osc.start(t);
      osc.stop(t + 0.25);
    }
  } else {
    // noir
    const bassNotes = [73.42, 82.41, 65.41, 73.42];
    for (let t = 0, i = 0; t < durationSeconds - 0.5; t += 2.0, i++) {
      const freq = bassNotes[i % bassNotes.length];
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.linearRampToValueAtTime(0.32, t + 0.2);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 1.9);
      osc.connect(gain);
      gain.connect(master);
      osc.start(t);
      osc.stop(t + 1.95);
    }
  }

  const rendered = await ctx.startRendering();
  return audioBufferToWavUrl(rendered);
}

/**
 * Manages live synchronized playback of timeline audio clips during NLE preview.
 */
export class TimelineAudioEngine {
  private elements = new Map<string, HTMLAudioElement>();

  sync(
    clips: TimelineClip[],
    currentTime: number,
    isPlaying: boolean,
    mutedTracks: Record<string, boolean>
  ) {
    const activeIds = new Set<string>();

    for (const clip of clips) {
      if ((clip.track !== 'voice' && clip.track !== 'music') || !clip.mediaUrl) {
        continue;
      }

      const isTrackMuted = Boolean(mutedTracks[clip.track]);
      const inRange =
        currentTime >= clip.startTime && currentTime < clip.startTime + clip.duration;

      let el = this.elements.get(clip.id);
      if (!el) {
        el = new Audio(clip.mediaUrl);
        el.preload = 'auto';
        this.elements.set(clip.id, el);
      } else if (el.src !== clip.mediaUrl && !el.src.endsWith(clip.mediaUrl)) {
        el.src = clip.mediaUrl;
      }

      activeIds.add(clip.id);

      if (!isPlaying || !inRange || isTrackMuted) {
        if (!el.paused) el.pause();
        continue;
      }

      const targetOffset = Math.max(0, currentTime - clip.startTime);
      if (Math.abs(el.currentTime - targetOffset) > 0.35) {
        try {
          el.currentTime = targetOffset;
        } catch {
          // Ignore seek errors before metadata loads
        }
      }

      const gain = Math.max(0, Math.min(1, computeClipGain(clip, currentTime)));
      el.volume = gain;

      if (el.paused) {
        el.play().catch(() => {
          // Browser autoplay policies may block uninitiated audio
        });
      }
    }

    for (const [id, el] of this.elements.entries()) {
      if (!activeIds.has(id)) {
        el.pause();
        this.elements.delete(id);
      }
    }
  }

  pauseAll() {
    for (const el of this.elements.values()) {
      el.pause();
    }
  }

  dispose() {
    this.pauseAll();
    this.elements.clear();
  }
}
