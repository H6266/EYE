export type AiTask =
  | 'image.generate'
  | 'image.edit'
  | 'image.analyze'
  | 'image.segment';

export type AiProvider = 'mock' | 'hf-inference' | 'hf-endpoint' | 'runpod';

export interface AiModelDefinition {
  id: string;
  repository: string;
  task: AiTask;
  providers: AiProvider[];
  endpointEnv?: string;
  requiredInputs: string[];
  supportedParameters: string[];
  expectedOutputs: string[];
  dependencies?: string[];
  deployment: 'hosted-api' | 'custom-worker';
}

export const AI_MODEL_REGISTRY: AiModelDefinition[] = [
  {
    id: 'z-image-turbo',
    repository: 'Tongyi-MAI/Z-Image-Turbo',
    task: 'image.generate',
    providers: ['mock', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_GENERATION_ENDPOINT_URL',
    requiredInputs: ['prompt'],
    supportedParameters: ['width', 'height', 'steps', 'guidanceScale', 'seed'],
    expectedOutputs: ['image/png'],
    dependencies: ['PyTorch CUDA worker', 'Diffusers ZImagePipeline'],
    deployment: 'custom-worker',
  },
  {
    id: 'flux-2-klein-4b',
    repository: 'black-forest-labs/FLUX.2-klein-4B',
    task: 'image.generate',
    providers: ['mock', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_GENERATION_ENDPOINT_URL',
    requiredInputs: ['prompt'],
    supportedParameters: ['width', 'height', 'steps', 'guidanceScale', 'seed'],
    expectedOutputs: ['image/png'],
    dependencies: ['PyTorch CUDA worker', 'Diffusers Flux2KleinPipeline'],
    deployment: 'custom-worker',
  },
  {
    id: 'flux-2-klein-4b-edit',
    repository: 'black-forest-labs/FLUX.2-klein-4B',
    task: 'image.edit',
    providers: ['mock', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_GENERATION_ENDPOINT_URL',
    requiredInputs: ['prompt', 'referenceImage'],
    supportedParameters: ['width', 'height', 'steps', 'guidanceScale', 'seed'],
    expectedOutputs: ['image/png'],
    dependencies: ['PyTorch CUDA worker', 'Diffusers Flux2KleinPipeline'],
    deployment: 'custom-worker',
  },
  {
    id: 'qwen-image-2512',
    repository: 'Qwen/Qwen-Image-2512',
    task: 'image.generate',
    providers: ['mock', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_GENERATION_ENDPOINT_URL',
    requiredInputs: ['prompt'],
    supportedParameters: ['width', 'height', 'steps', 'guidanceScale', 'seed'],
    expectedOutputs: ['image/png'],
    dependencies: ['PyTorch CUDA worker', 'Diffusers QwenImagePipeline'],
    deployment: 'custom-worker',
  },
  {
    id: 'qwen-image-2512-lightning',
    repository: 'lightx2v/Qwen-Image-2512-Lightning',
    task: 'image.generate',
    providers: ['mock', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_GENERATION_ENDPOINT_URL',
    requiredInputs: ['prompt'],
    supportedParameters: ['width', 'height', 'steps', 'guidanceScale', 'seed'],
    expectedOutputs: ['image/png'],
    dependencies: [
      'Qwen/Qwen-Image-2512 base model required on the worker',
      'Diffusers QwenImagePipeline',
      'Lightning LoRA adapter',
    ],
    deployment: 'custom-worker',
  },
  {
    id: 'florence-2-base-ft',
    repository: 'microsoft/Florence-2-base-ft',
    task: 'image.analyze',
    providers: ['mock', 'hf-inference', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_FLORENCE_ENDPOINT_URL',
    requiredInputs: ['referenceImage', 'instructions'],
    supportedParameters: ['taskPrompt'],
    expectedOutputs: ['application/json'],
    dependencies: ['Transformers Florence-2 custom-code support'],
    deployment: 'hosted-api',
  },
  {
    id: 'sam2-hiera-base-plus',
    repository: 'facebook/sam2-hiera-base-plus',
    task: 'image.segment',
    providers: ['mock', 'hf-inference', 'hf-endpoint', 'runpod'],
    endpointEnv: 'HF_SAM2_ENDPOINT_URL',
    requiredInputs: ['referenceImage'],
    supportedParameters: ['points', 'labels', 'box'],
    expectedOutputs: ['application/json'],
    dependencies: ['SAM2 custom worker or compatible Transformers mask-generation pipeline'],
    deployment: 'custom-worker',
  },
];

export function getAiModel(modelId: string): AiModelDefinition | undefined {
  return AI_MODEL_REGISTRY.find((model) => model.id === modelId);
}
