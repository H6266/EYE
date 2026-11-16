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
    const client = new PythonDiffusionClient(
      process.env.PYTHON_SERVICE_URL ?? 'http://localhost:8100',
      Number(process.env.PYTHON_SERVICE_TIMEOUT_MS ?? 300000),
    );

    const available = await client.checkHealth();
    if (!available) {
      // Fall back to local SDXS-512
      return localDiffusionEngine.generate({
        prompt: (input as LocalDiffusionInput).prompt,
        negativePrompt: (input as PythonGenerationInput).negativePrompt ?? '',
        guidanceScale: 0,
        seed: (input as PythonGenerationInput).seed ?? 0,
        aspectRatio: (input as LocalDiffusionInput).aspectRatio ?? '16:9',
        referenceImagePngBase64: (input as LocalDiffusionInput).referenceImagePngBase64,
        img2imgStrength: (input as LocalDiffusionInput).img2imgStrength,
      });
    }

    const pythonInput: PythonGenerationInput = {
      model: engineId,
      prompt: (input as PythonGenerationInput).prompt,
      negativePrompt: (input as PythonGenerationInput).negativePrompt,
      width: (input as PythonGenerationInput).width,
      height: (input as PythonGenerationInput).height,
      numInferenceSteps: (input as PythonGenerationInput).numInferenceSteps,
      guidanceScale: (input as PythonGenerationInput).guidanceScale,
      seed: (input as PythonGenerationInput).seed,
      returnBase64: true,
    };
    return client.generate(pythonInput);
  }

  return localDiffusionEngine.generate(input as LocalDiffusionInput);
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
