import type {
  CompactLightGbmModel,
  CompactTreeNode,
  ModelManifest,
  ModelPrediction,
} from "./types";

let classifierModel: CompactLightGbmModel | null = null;
let regressorModel: CompactLightGbmModel | null = null;
let loadedVersion: string | null = null;

export async function loadManifest(): Promise<ModelManifest> {
  const response = await fetch("/models/feature_manifest.json", {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Model manifest is unavailable.");
  return response.json() as Promise<ModelManifest>;
}

async function loadJsonModel(path: string): Promise<CompactLightGbmModel> {
  const response = await fetch(path, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Browser model is unavailable: ${path}`);
  return response.json() as Promise<CompactLightGbmModel>;
}

async function ensureModels(manifest: ModelManifest) {
  if (!manifest.modelsReady || !manifest.classifier || !manifest.regressor) {
    throw new Error("Browser models have not been exported yet.");
  }
  if (
    classifierModel &&
    regressorModel &&
    loadedVersion === manifest.modelVersion
  ) {
    return { classifierModel, regressorModel };
  }

  [classifierModel, regressorModel] = await Promise.all([
    loadJsonModel(manifest.classifier.path),
    loadJsonModel(manifest.regressor.path),
  ]);
  loadedVersion = manifest.modelVersion;
  return { classifierModel, regressorModel };
}

function treeValue(node: CompactTreeNode, vector: number[]): number {
  let current = node;
  while (!("v" in current)) {
    const value = vector[current.f];
    const missing =
      Number.isNaN(value) || (current.m === "Zero" && value === 0);
    const goLeft = missing ? current.d : value <= current.t;
    current = goLeft ? current.l : current.r;
  }
  return current.v;
}

function rawScores(
  model: CompactLightGbmModel,
  vector: number[],
  outputs: number,
): number[] {
  const result = Array.from({ length: outputs }, () => 0);
  model.trees.forEach((tree, treeIndex) => {
    result[treeIndex % outputs] += treeValue(tree, vector);
  });
  return result;
}

function softmax(values: number[]): number[] {
  const maximum = Math.max(...values);
  const exponentiated = values.map((value) => Math.exp(value - maximum));
  const total = exponentiated.reduce((sum, value) => sum + value, 0);
  return exponentiated.map((value) => value / total);
}

export async function predictWithFrozenModels(
  manifest: ModelManifest,
  vector: number[],
): Promise<ModelPrediction> {
  if (vector.length !== manifest.featureCount) {
    throw new Error(
      `Expected ${manifest.featureCount} features, received ${vector.length}.`,
    );
  }

  const models = await ensureModels(manifest);
  const labels =
    models.classifierModel.classLabels ?? manifest.classLabels ?? [-1, 0, 1];
  const probabilities = softmax(
    rawScores(models.classifierModel, vector, labels.length),
  );
  const predictedReturnBps = rawScores(
    models.regressorModel,
    vector,
    1,
  )[0];

  const maximum = probabilities.reduce(
    (best, value, index) =>
      value > probabilities[best] ? index : best,
    0,
  );

  return {
    labels,
    probabilities,
    predictedLabel: labels[maximum],
    predictedReturnBps,
  };
}
