export type FeatureRange = {
  p01: number;
  median: number;
  p99: number;
  minimum: number;
  maximum: number;
};

export type ModelManifest = {
  modelsReady: boolean;
  modelVersion: string;
  sourceConfigurationHash?: string | null;
  sourceGitCommit?: string | null;
  ticker: string;
  horizonEvents: number;
  featureSet: string;
  featureCount: number;
  featureNames: string[];
  featureRanges: Record<string, FeatureRange>;
  inputType?: string;
  classLabels?: number[];
  classifier?: {
    path: string;
    sha256: string;
    bytes: number;
  };
  regressor?: {
    path: string;
    sha256: string;
    bytes: number;
  };
  parity?: {
    rows: number;
    rtol: number;
    atol: number;
    classifierMaximumAbsoluteDifference: number;
    regressorMaximumAbsoluteDifference: number;
    classLabelsExact: boolean;
  };
  scope: string;
};

export type CompactTreeNode =
  | { v: number }
  | {
      f: number;
      t: number;
      d: boolean;
      m: string;
      l: CompactTreeNode;
      r: CompactTreeNode;
    };

export type CompactLightGbmModel = {
  format: "lightgbm-compact-v1";
  kind: "multiclass" | "regression";
  featureCount: number;
  treeCount: number;
  numClass?: number;
  classLabels?: number[];
  trees: CompactTreeNode[];
};

export type ModelPrediction = {
  labels: number[];
  probabilities: number[];
  predictedLabel: number;
  predictedReturnBps: number;
};
