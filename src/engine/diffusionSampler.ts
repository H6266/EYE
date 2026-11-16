export const LATENT_CHANNELS = 4;
export const LATENT_HEIGHT = 64;
export const LATENT_WIDTH = 64;
export const LATENT_SIZE = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH; // 16384
export const SDXS_SIGMA_MAX = 14.614646911621094;
export const VAE_LATENT_SCALE = 0.18215;

/**
 * Deterministic Mulberry32 PRNG for reproducible diffusion seeds.
 */
export function createSeededPrng(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller standard normal N(0, 1) latent noise tensor scaled by initSigma.
 */
export function generateSeededLatents(
  seed: number,
  initSigma = SDXS_SIGMA_MAX,
  size = LATENT_SIZE
): Float32Array {
  const rand = createSeededPrng(seed);
  const out = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.max(1e-7, rand());
    const u2 = rand();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const angle = 2.0 * Math.PI * u2;
    out[i] = mag * Math.cos(angle) * initSigma;
    if (i + 1 < size) {
      out[i + 1] = mag * Math.sin(angle) * initSigma;
    }
  }
  return out;
}

export interface ScheduleStep {
  timestep: number;
  sigma: number;
  nextSigma: number;
}

function computeAllSigmas(trainTimesteps = 1000): Float64Array {
  const betaStart = 0.00085;
  const betaEnd = 0.012;
  const sqrtStart = Math.sqrt(betaStart);
  const sqrtEnd = Math.sqrt(betaEnd);
  const allSigmas = new Float64Array(trainTimesteps);

  let prod = 1.0;
  for (let i = 0; i < trainTimesteps; i++) {
    const t = i / (trainTimesteps - 1);
    const sqrtBeta = sqrtStart + t * (sqrtEnd - sqrtStart);
    const beta = sqrtBeta * sqrtBeta;
    prod *= 1.0 - beta;
    allSigmas[i] = Math.sqrt((1.0 - prod) / prod);
  }
  return allSigmas;
}

/**
 * Trailing EulerDiscreteScheduler matching Stable Diffusion 1.5 / SDXS-512.
 */
export function buildEulerTrailingSchedule(numSteps: number): ScheduleStep[] {
  const steps = Math.max(1, Math.min(8, Math.round(numSteps)));
  const trainTimesteps = 1000;
  const allSigmas = computeAllSigmas(trainTimesteps);
  const stepRatio = trainTimesteps / steps;
  const schedule: ScheduleStep[] = [];

  for (let s = 0; s < steps; s++) {
    const timestep = Math.max(
      0,
      Math.min(trainTimesteps - 1, Math.round(trainTimesteps - 1 - s * stepRatio))
    );
    const sigma = s === 0 && steps === 1 ? SDXS_SIGMA_MAX : allSigmas[timestep];

    let nextSigma = 0;
    if (s + 1 < steps) {
      const nextT = Math.max(
        0,
        Math.min(trainTimesteps - 1, Math.round(trainTimesteps - 1 - (s + 1) * stepRatio))
      );
      nextSigma = allSigmas[nextT];
    }

    schedule.push({ timestep, sigma, nextSigma });
  }

  return schedule;
}

/**
 * Computes noisy Img2Img starting latent at effective strength in [0.35, 1.0].
 */
export function addLatentNoiseForImg2Img(
  cleanLatent: Float32Array,
  seed: number,
  strength: number
): { noisyLatent: Float32Array; sigma: number; timestep: number } {
  const clampedStrength = Math.max(0.35, Math.min(1.0, strength));
  const allSigmas = computeAllSigmas(1000);
  const timestep = Math.max(550, Math.min(999, Math.round(550 + clampedStrength * 449)));
  const sigma = timestep === 999 ? SDXS_SIGMA_MAX : allSigmas[timestep];
  const unitNoise = generateSeededLatents(seed, 1.0, cleanLatent.length);
  const noisyLatent = new Float32Array(cleanLatent.length);

  for (let i = 0; i < cleanLatent.length; i++) {
    noisyLatent[i] = cleanLatent[i] + sigma * unitNoise[i];
  }

  return { noisyLatent, sigma, timestep };
}

/**
 * Scales noisy latent sample by 1 / sqrt(sigma^2 + 1) before feeding into the UNet Brain.
 */
export function scaleModelInput(sample: Float32Array, sigma: number): Float32Array {
  const factor = 1.0 / Math.sqrt(sigma * sigma + 1.0);
  const scaled = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) {
    scaled[i] = sample[i] * factor;
  }
  return scaled;
}

/**
 * Classifier-Free Guidance (CFG): eps = uncond + guidanceScale * (cond - uncond)
 */
export function applyClassifierFreeGuidance(
  condPred: Float32Array,
  uncondPred: Float32Array,
  guidanceScale: number
): Float32Array {
  const out = new Float32Array(condPred.length);
  for (let i = 0; i < condPred.length; i++) {
    out[i] = uncondPred[i] + guidanceScale * (condPred[i] - uncondPred[i]);
  }
  return out;
}

/**
 * Single Euler step update: x_{next} = x + (sigma_next - sigma) * eps
 */
export function eulerStep(
  sample: Float32Array,
  noisePred: Float32Array,
  sigma: number,
  nextSigma: number
): Float32Array {
  const dt = nextSigma - sigma;
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) {
    out[i] = sample[i] + dt * noisePred[i];
  }
  return out;
}
