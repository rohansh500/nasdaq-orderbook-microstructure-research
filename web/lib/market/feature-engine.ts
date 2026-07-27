import type { SyntheticSnapshot } from "./scenario-generator";

export const FEATURE_NAMES = [
  "spread_bps", "queue_imbalance_l1", "microprice_deviation_bps",
  "depth_imbalance_1", "depth_imbalance_5", "depth_imbalance_10",
  "ofi_l1", "event_interarrival_us", "is_submission", "is_partial_cancel",
  "is_deletion", "is_visible_execution", "is_hidden_execution",
  "event_intensity_20", "ofi_l1_sum_20", "add_pressure_20",
  "cancel_pressure_20", "trade_pressure_20", "rolling_volatility_20",
  "event_intensity_50", "ofi_l1_sum_50", "add_pressure_50",
  "cancel_pressure_50", "trade_pressure_50", "rolling_volatility_50",
  "event_intensity_100", "ofi_l1_sum_100", "add_pressure_100",
  "cancel_pressure_100", "trade_pressure_100", "rolling_volatility_100",
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];
export type FeatureVector = Record<FeatureName, number>;

const epsilon = 1e-12;
function safeDivide(numerator: number, denominator: number) {
  return numerator / (Math.abs(denominator) + epsilon);
}
function sum(values: number[]) { return values.reduce((total, value) => total + value, 0); }
function sampleStd(values: number[]) {
  if (values.length < 2) return 0;
  const mean = sum(values) / values.length;
  return Math.sqrt(sum(values.map((value) => (value - mean) ** 2)) / (values.length - 1));
}

export function calculateFeatures(rows: SyntheticSnapshot[]): FeatureVector {
  if (rows.length < 101) throw new Error("A 100-event warm-up history is required.");
  const enriched = rows.map((row, index) => {
    const bid = row.bidPrices[0];
    const ask = row.askPrices[0];
    const bidSize = row.bidSizes[0];
    const askSize = row.askSizes[0];
    const mid = (ask + bid) / 2;
    const spread = ask - bid;
    const topDepth = bidSize + askSize;
    const microprice = safeDivide(ask * bidSize + bid * askSize, topDepth);
    const previous = index > 0 ? rows[index - 1] : null;
    let ofi = 0;
    if (previous) {
      const previousBid = previous.bidPrices[0];
      const previousAsk = previous.askPrices[0];
      const previousBidSize = previous.bidSizes[0];
      const previousAskSize = previous.askSizes[0];
      const bidContribution = (bid >= previousBid ? bidSize : 0) - (bid <= previousBid ? previousBidSize : 0);
      const askContribution = -(ask <= previousAsk ? askSize : 0) + (ask >= previousAsk ? previousAskSize : 0);
      ofi = bidContribution + askContribution;
    }
    const signedAdd = row.eventType === 1 ? row.direction * row.size : 0;
    const signedCancel = row.eventType === 2 || row.eventType === 3 ? -row.direction * row.size : 0;
    const signedTrade = row.eventType === 4 || row.eventType === 5 ? -row.direction * row.size : 0;
    const logReturn = previous ? Math.log(mid) - Math.log((previous.askPrices[0] + previous.bidPrices[0]) / 2) : 0;
    return { row, mid, spread, topDepth, microprice, ofi, signedAdd, signedCancel, signedTrade, logReturn };
  });

  const lastIndex = enriched.length - 1;
  const current = enriched[lastIndex];
  const depthImbalance = (depth: number) => {
    const bidDepth = sum(current.row.bidSizes.slice(0, depth));
    const askDepth = sum(current.row.askSizes.slice(0, depth));
    return safeDivide(bidDepth - askDepth, bidDepth + askDepth);
  };

  const result = {} as FeatureVector;
  result.spread_bps = 10_000 * safeDivide(current.spread, current.mid);
  result.queue_imbalance_l1 = safeDivide(current.row.bidSizes[0] - current.row.askSizes[0], current.topDepth);
  result.microprice_deviation_bps = 10_000 * safeDivide(current.microprice - current.mid, current.mid);
  result.depth_imbalance_1 = depthImbalance(1);
  result.depth_imbalance_5 = depthImbalance(5);
  result.depth_imbalance_10 = depthImbalance(10);
  result.ofi_l1 = current.ofi;
  result.event_interarrival_us = Math.max(0, current.row.timeSeconds - enriched[lastIndex - 1].row.timeSeconds) * 1_000_000;
  result.is_submission = current.row.eventType === 1 ? 1 : 0;
  result.is_partial_cancel = current.row.eventType === 2 ? 1 : 0;
  result.is_deletion = current.row.eventType === 3 ? 1 : 0;
  result.is_visible_execution = current.row.eventType === 4 ? 1 : 0;
  result.is_hidden_execution = current.row.eventType === 5 ? 1 : 0;

  for (const window of [20, 50, 100] as const) {
    const windowRows = enriched.slice(-window);
    const previousTime = enriched[lastIndex - window].row.timeSeconds;
    result[`event_intensity_${window}`] = safeDivide(window, current.row.timeSeconds - previousTime);
    result[`ofi_l1_sum_${window}`] = sum(windowRows.map((value) => value.ofi));
    result[`add_pressure_${window}`] = safeDivide(sum(windowRows.map((value) => value.signedAdd)), sum(current.row.bidSizes) + sum(current.row.askSizes));
    result[`cancel_pressure_${window}`] = safeDivide(sum(windowRows.map((value) => value.signedCancel)), sum(current.row.bidSizes) + sum(current.row.askSizes));
    result[`trade_pressure_${window}`] = safeDivide(sum(windowRows.map((value) => value.signedTrade)), sum(current.row.bidSizes) + sum(current.row.askSizes));
    result[`rolling_volatility_${window}`] = sampleStd(windowRows.map((value) => value.logReturn));
  }
  return result;
}

export function vectorInOrder(features: FeatureVector, names: readonly string[]) {
  return names.map((name) => {
    if (!(name in features)) throw new Error(`Missing generated feature: ${name}`);
    const value = features[name as FeatureName];
    if (!Number.isFinite(value)) throw new Error(`Non-finite generated feature: ${name}`);
    return value;
  });
}
