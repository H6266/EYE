import { ImageEngineId, PYTHON_ENGINE_IDS } from '../types/studio';
import { localDiffusionEngine, LocalDiffusionResult } from './localDiffusionPipeline';
import { PythonDiffusionClient, PythonGenerationInput, PythonGenerationResult } from './pythonDiffusionClient';

export interface LocalDiffusionInput {
  prompt: string;
  negativePrompt?: string;
  guidanceScale?: number;
  seed?: number;
  aspectRatio?: '16:9' | '9:16' | '1:1';
  referenceImagePngBase64?: string;
  img2imgStrength?: number;
}

export type GenerationResult = LocalDiffusionResult | PythonGenerationResult;

export async function generateImage(
  input: LocalDiffusionInput | PythonGenerationInput,
  engineId: ImageEngineId,
): Promise<GenerationResult> {
  if (PYTHON_ENGINE_IDS.includes(engineId)) {
    if (!('model' in input)) {
      throw new Error(`The '${engineId}' engine requires Python generation options.`);
    }
    const client = new PythonDiffusionClient(
      process.env.PYTHON_SERVICE_URL ?? 'http://localhost:8100',
      Number(process.env.PYTHON_SERVICE_TIMEOUT_MS ?? 300000),
    );
    return client.generate({ ...input, model: engineId });
  }

  if ('model' in input) {
    throw new Error(`The '${engineId}' engine does not accept Python generation options.`);
  }
  return localDiffusionEngine.generate(input);
}

export async function checkPythonService(): Promise<boolean> {
  const client = new PythonDiffusionClient(
    process.env.PYTHON_SERVICE_URL ?? 'http://localhost:8100',
  );
  return client.checkHealth();
}

export function getPythonEngineIds(): ImageEngineId[] {
  return [...PYTHON_ENGINE_IDS];
}
