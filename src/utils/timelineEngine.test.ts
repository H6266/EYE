import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildAutoSubtitleCues,
  computeClipGain,
  exportSrt,
  exportVtt,
  formatSrtTimestamp,
  formatTimecode,
  getActiveVideoLayer,
  getProjectDuration,
  moveClip,
  resizeClip,
  splitClipAtTime,
} from './timelineEngine';
import { TimelineClip } from '../types/studio';

const baseClips: TimelineClip[] = [
  {
    id: 'v1',
    track: 'video',
    title: 'Opening Observatory',
    startTime: 0,
    duration: 5,
    visualStyle: 'cinematic',
    transition: 'cut',
    transitionDuration: 0,
    volume: 1,
    fadeIn: 0,
    fadeOut: 0,
  },
  {
    id: 'v2',
    track: 'video',
    title: 'Cartoon Airship',
    startTime: 5,
    duration: 6,
    visualStyle: 'cartoon2d',
    transition: 'crossfade',
    transitionDuration: 1,
    volume: 1,
    fadeIn: 0,
    fadeOut: 0,
  },
  {
    id: 's1',
    track: 'subtitle',
    title: 'Line 1',
    startTime: 0.5,
    duration: 3.5,
    visualStyle: 'original',
    transition: 'cut',
    transitionDuration: 0,
    volume: 1,
    fadeIn: 0,
    fadeOut: 0,
    subtitleText: 'Above the clouds, the observatory wakes.',
    subtitleSpeaker: 'Narrator',
  },
];

test('formatTimecode formats MM:SS:FF accurately at 24fps', () => {
  assert.equal(formatTimecode(0, 24), '00:00:00');
  assert.equal(formatTimecode(65.5, 24), '01:05:12');
  assert.equal(formatTimecode(-3, 24), '00:00:00');
});

test('formatSrtTimestamp formats HH:MM:SS,mmm accurately', () => {
  assert.equal(formatSrtTimestamp(0), '00:00:00,000');
  assert.equal(formatSrtTimestamp(3661.25), '01:01:01,250');
});

test('getProjectDuration computes exact max end or minimum duration', () => {
  assert.equal(getProjectDuration(baseClips, 10), 11);
  assert.equal(getProjectDuration([], 20), 20);
});

test('getActiveVideoLayer resolves crossfade blend between adjacent clips', () => {
  const midFade = getActiveVideoLayer(baseClips, 5.5);
  assert.equal(midFade.primary?.id, 'v2');
  assert.equal(midFade.outgoing?.id, 'v1');
  assert.equal(midFade.blend, 0.5);

  const steady = getActiveVideoLayer(baseClips, 7.5);
  assert.equal(steady.primary?.id, 'v2');
  assert.equal(steady.outgoing, null);
  assert.equal(steady.blend, 1);
});

test('moveClip snaps to adjacent clip edge within threshold', () => {
  const moved = moveClip(baseClips, 'v2', 5.12, 0, true);
  const v2 = moved.find((c) => c.id === 'v2');
  assert.equal(v2?.startTime, 5);
});

test('resizeClip enforces minimum clip duration', () => {
  const resized = resizeClip(baseClips, 'v1', 0.1, 0.5);
  assert.equal(resized[0].duration, 0.5);
});

test('splitClipAtTime splits a clip into two contiguous parts', () => {
  let seq = 1;
  const split = splitClipAtTime(baseClips, 'v1', 2.5, () => `split-${seq++}`);
  const parts = split.filter((c) => c.track === 'video' && c.startTime < 5);
  assert.equal(parts.length, 2);
  assert.equal(parts[0].duration, 2.5);
  assert.equal(parts[1].startTime, 2.5);
  assert.equal(parts[1].duration, 2.5);
});

test('computeClipGain applies linear fadeIn and fadeOut envelopes', () => {
  const audioClip: TimelineClip = {
    ...baseClips[0],
    track: 'music',
    startTime: 0,
    duration: 10,
    volume: 0.8,
    fadeIn: 2,
    fadeOut: 2,
  };
  assert.equal(Number(computeClipGain(audioClip, 1).toFixed(2)), 0.4);
  assert.equal(Number(computeClipGain(audioClip, 5).toFixed(2)), 0.8);
  assert.equal(Number(computeClipGain(audioClip, 9).toFixed(2)), 0.4);
});

test('exportSrt and exportVtt serialize timed subtitle tracks', () => {
  const srt = exportSrt(baseClips);
  assert.match(srt, /^1\n00:00:00,500 --> 00:00:04,000\nNarrator: Above the clouds/);
  const vtt = exportVtt(baseClips);
  assert.match(vtt, /^WEBVTT\n\n00:00:00\.500 --> 00:00:04\.000\n<v Narrator>Above the clouds/);
});

test('buildAutoSubtitleCues splits narration into proportional cues and strips markup tags', () => {
  const cues = buildAutoSubtitleCues(
    'Hello <breath> world! Time to fly |yeah| across the sky.',
    2,
    6,
    'Fox'
  );
  assert.equal(cues.length, 2);
  assert.equal(cues[0].startTime, 2);
  assert.equal(cues[0].speaker, 'Fox');
  assert.ok(!cues[0].text.includes('<breath>'));
  assert.ok(!cues[1].text.includes('|yeah|'));
});
