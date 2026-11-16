import { AspectRatio, SubtitleStyle, TimelineClip, VisualStyle } from '../types/studio';
import {
  getActiveOverlayClips,
  getActiveSubtitles,
  getActiveVideoLayer,
} from './timelineEngine';

export interface MediaCache {
  images: Map<string, HTMLImageElement>;
  videos: Map<string, HTMLVideoElement>;
}

export function getCanvasDimensions(aspect: AspectRatio): { width: number; height: number } {
  if (aspect === '9:16') return { width: 540, height: 960 };
  if (aspect === '1:1') return { width: 720, height: 720 };
  return { width: 960, height: 540 };
}

function getFilterForStyle(style: VisualStyle): string {
  switch (style) {
    case 'cinematic':
      return 'contrast(1.12) saturate(1.08) brightness(0.96)';
    case 'cartoon2d':
      return 'contrast(1.18) saturate(1.38) brightness(1.04)';
    case 'anime':
      return 'contrast(1.08) saturate(1.32) brightness(1.03)';
    case 'claymation':
      return 'contrast(1.1) saturate(1.14) sepia(0.08)';
    case 'noir':
      return 'grayscale(1) contrast(1.42) brightness(0.92)';
    case 'pixar3d':
      return 'contrast(1.1) saturate(1.26) brightness(1.05)';
    case 'watercolor':
      return 'contrast(0.96) saturate(1.18) brightness(1.04)';
    default:
      return 'none';
  }
}

function drawCoverMedia(
  ctx: CanvasRenderingContext2D,
  source: HTMLImageElement | HTMLVideoElement,
  srcW: number,
  srcH: number,
  canvasW: number,
  canvasH: number,
  clip: TimelineClip,
  time: number
) {
  if (!srcW || !srcH) return;

  const progress = Math.max(0, Math.min(1, (time - clip.startTime) / Math.max(0.1, clip.duration)));
  let scaleBoost = 1;
  let panX = 0;

  if (clip.kenBurns === 'zoom-in') {
    scaleBoost = 1 + progress * 0.08;
  } else if (clip.kenBurns === 'zoom-out') {
    scaleBoost = 1.08 - progress * 0.08;
  } else if (clip.kenBurns === 'pan-right') {
    scaleBoost = 1.06;
    panX = (progress - 0.5) * 28;
  }

  const baseScale = Math.max(canvasW / srcW, canvasH / srcH) * scaleBoost;
  const drawW = srcW * baseScale;
  const drawH = srcH * baseScale;
  const dx = (canvasW - drawW) / 2 + panX;
  const dy = (canvasH - drawH) / 2;

  ctx.save();
  ctx.filter = getFilterForStyle(clip.visualStyle);
  ctx.drawImage(source, dx, dy, drawW, drawH);
  ctx.restore();

  // Subtle style finishing overlay
  if (clip.visualStyle === 'cinematic') {
    const barH = Math.round(canvasH * 0.085);
    ctx.fillStyle = '#07080A';
    ctx.fillRect(0, 0, canvasW, barH);
    ctx.fillRect(0, canvasH - barH, canvasW, barH);
  } else if (clip.visualStyle === 'cartoon2d') {
    ctx.save();
    ctx.strokeStyle = 'rgba(20, 20, 24, 0.75)';
    ctx.lineWidth = 6;
    ctx.strokeRect(3, 3, canvasW - 6, canvasH - 6);
    ctx.restore();
  } else if (clip.visualStyle === 'noir') {
    const grad = ctx.createRadialGradient(
      canvasW / 2,
      canvasH / 2,
      canvasW * 0.25,
      canvasW / 2,
      canvasH / 2,
      canvasW * 0.7
    );
    grad.addColorStop(0, 'rgba(0,0,0,0)');
    grad.addColorStop(1, 'rgba(0,0,0,0.48)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvasW, canvasH);
  }
}

function resolveDrawable(
  clip: TimelineClip,
  cache: MediaCache,
  time: number,
  isPlaying: boolean
): { el: HTMLImageElement | HTMLVideoElement; w: number; h: number } | null {
  if (!clip.mediaUrl) return null;

  if (clip.mediaKind === 'video') {
    let video = cache.videos.get(clip.mediaUrl);
    if (!video) {
      video = document.createElement('video');
      video.src = clip.mediaUrl;
      video.muted = true;
      video.playsInline = true;
      video.loop = true;
      cache.videos.set(clip.mediaUrl, video);
    }
    const targetOffset = Math.max(0, time - clip.startTime);
    if (video.readyState >= 2) {
      if (Math.abs(video.currentTime - targetOffset) > 0.3) {
        video.currentTime = targetOffset % Math.max(1, video.duration || 5);
      }
      if (isPlaying && video.paused) {
        video.play().catch(() => {});
      } else if (!isPlaying && !video.paused) {
        video.pause();
      }
      return { el: video, w: video.videoWidth, h: video.videoHeight };
    }
    return null;
  }

  let img = cache.images.get(clip.mediaUrl);
  if (!img) {
    img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = clip.mediaUrl;
    cache.images.set(clip.mediaUrl, img);
  }
  if (img.complete && img.naturalWidth > 0) {
    return { el: img, w: img.naturalWidth, h: img.naturalHeight };
  }
  return null;
}

function drawSubtitles(
  ctx: CanvasRenderingContext2D,
  cues: TimelineClip[],
  canvasW: number,
  canvasH: number
) {
  if (cues.length === 0) return;

  const cue = cues[0];
  const text = cue.subtitleText?.trim() || '';
  if (!text) return;

  const style: SubtitleStyle = cue.subtitleStyle || 'cinematic';
  const speakerPrefix = cue.subtitleSpeaker?.trim() ? `${cue.subtitleSpeaker.trim()}: ` : '';
  const fullLine = `${speakerPrefix}${text}`;

  ctx.save();
  const fontSize = Math.round(canvasH * 0.044);
  ctx.font =
    style === 'cartoon'
      ? `700 ${fontSize}px "Outfit", sans-serif`
      : `500 ${fontSize}px "DM Sans", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const maxLineWidth = canvasW * 0.82;
  const words = fullLine.split(' ');
  const lines: string[] = [];
  let current = '';
  for (const w of words) {
    const next = current ? `${current} ${w}` : w;
    if (ctx.measureText(next).width > maxLineWidth && current) {
      lines.push(current);
      current = w;
    } else {
      current = next;
    }
  }
  if (current) lines.push(current);

  const lineHeight = fontSize * 1.35;
  const blockH = lines.length * lineHeight + 18;
  const yBottom = canvasH - Math.round(canvasH * 0.11);
  const blockY = yBottom - blockH;

  const widest = Math.max(...lines.map((l) => ctx.measureText(l).width));
  const boxW = Math.min(canvasW - 40, widest + 36);
  const boxX = (canvasW - boxW) / 2;

  if (style === 'cartoon') {
    ctx.fillStyle = '#FACC15';
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(boxX, blockY, boxW, blockH, 10);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#111827';
  } else if (style === 'minimal') {
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 8;
    ctx.fillStyle = '#FFFFFF';
  } else {
    ctx.fillStyle = 'rgba(12, 14, 18, 0.78)';
    ctx.beginPath();
    ctx.roundRect(boxX, blockY, boxW, blockH, 6);
    ctx.fill();
    ctx.fillStyle = '#F8FAFC';
  }

  lines.forEach((line, idx) => {
    const y = blockY + 12 + lineHeight * (idx + 0.5);
    ctx.fillText(line, canvasW / 2, y);
  });

  ctx.restore();
}

function drawOverlayClip(
  ctx: CanvasRenderingContext2D,
  clip: TimelineClip,
  cache: MediaCache,
  canvasW: number,
  canvasH: number,
  time: number,
  isPlaying: boolean
) {
  const drawable = resolveDrawable(clip, cache, time, isPlaying);
  if (!drawable || !drawable.w || !drawable.h) return;

  const scale = Math.max(0.2, Math.min(0.85, clip.overlayScale ?? 0.42));
  const frameMode = clip.overlayFrame || 'portrait';
  const position = clip.overlayPosition || 'right';

  const maxH = canvasH * scale;
  const maxW = canvasW * (frameMode === 'badge' ? scale * 0.65 : scale * 0.85);
  const ratio = drawable.w / drawable.h;

  let drawW = maxW;
  let drawH = drawW / ratio;
  if (drawH > maxH) {
    drawH = maxH;
    drawW = drawH * ratio;
  }

  if (frameMode === 'badge') {
    const size = Math.min(maxW, maxH);
    drawW = size;
    drawH = size;
  }

  const marginX = Math.round(canvasW * 0.05);
  const marginY = Math.round(canvasH * 0.08);
  let x = canvasW - drawW - marginX;
  if (position === 'left') x = marginX;
  else if (position === 'center') x = (canvasW - drawW) / 2;
  const y = canvasH - drawH - marginY;

  ctx.save();
  ctx.filter = getFilterForStyle(clip.visualStyle);
  ctx.shadowColor = 'rgba(0, 0, 0, 0.55)';
  ctx.shadowBlur = 18;
  ctx.shadowOffsetY = 6;

  if (frameMode === 'badge') {
    const cx = x + drawW / 2;
    const cy = y + drawH / 2;
    const r = drawW / 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.closePath();
    ctx.clip();
    const coverScale = Math.max(drawW / drawable.w, drawH / drawable.h);
    const sw = drawable.w * coverScale;
    const sh = drawable.h * coverScale;
    ctx.drawImage(drawable.el, cx - sw / 2, cy - sh / 2, sw, sh);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = 'rgba(229, 169, 60, 0.85)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (frameMode === 'portrait') {
    ctx.beginPath();
    ctx.roundRect(x, y, drawW, drawH, 14);
    ctx.clip();
    ctx.drawImage(drawable.el, x, y, drawW, drawH);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, drawW, drawH, 14);
    ctx.stroke();
    ctx.restore();
    return;
  }

  ctx.drawImage(drawable.el, x, y, drawW, drawH);
  ctx.restore();
}

export function renderCompositeFrame(
  ctx: CanvasRenderingContext2D,
  clips: TimelineClip[],
  time: number,
  cache: MediaCache,
  aspectRatio: AspectRatio,
  isPlaying: boolean,
  showSubtitles = true
) {
  const { width, height } = getCanvasDimensions(aspectRatio);
  ctx.fillStyle = '#090B0E';
  ctx.fillRect(0, 0, width, height);

  const layer = getActiveVideoLayer(clips, time);

  if (!layer.primary) {
    // Clean unpopulated slate
    ctx.fillStyle = '#15181E';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#6B7280';
    ctx.font = '500 15px "DM Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No active clip at playhead — drag media onto V1', width / 2, height / 2);
  } else {
    if (layer.outgoing && layer.blend < 1) {
      const outDrawable = resolveDrawable(layer.outgoing, cache, time, isPlaying);
      if (outDrawable) {
        ctx.save();
        ctx.globalAlpha = 1 - layer.blend;
        drawCoverMedia(
          ctx,
          outDrawable.el,
          outDrawable.w,
          outDrawable.h,
          width,
          height,
          layer.outgoing,
          time
        );
        ctx.restore();
      }
    }

    const primaryDrawable = resolveDrawable(layer.primary, cache, time, isPlaying);
    if (primaryDrawable) {
      ctx.save();
      if (layer.outgoing && layer.blend < 1) {
        if (layer.primary.transition === 'wipe') {
          ctx.beginPath();
          ctx.rect(0, 0, width * layer.blend, height);
          ctx.clip();
        } else {
          ctx.globalAlpha = layer.blend;
        }
      }
      drawCoverMedia(
        ctx,
        primaryDrawable.el,
        primaryDrawable.w,
        primaryDrawable.h,
        width,
        height,
        layer.primary,
        time
      );
      ctx.restore();
    } else {
      // Loading frame placeholder
      ctx.fillStyle = '#181C24';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '500 14px "DM Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Loading frame: ${layer.primary.title}`, width / 2, height / 2);
    }
  }

  const activeOverlays = getActiveOverlayClips(clips, time);
  for (const overlayClip of activeOverlays) {
    drawOverlayClip(ctx, overlayClip, cache, width, height, time, isPlaying);
  }

  if (showSubtitles) {
    const activeSubs = getActiveSubtitles(clips, time);
    drawSubtitles(ctx, activeSubs, width, height);
  }
}
