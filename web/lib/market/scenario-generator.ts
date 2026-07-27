export type ScenarioControls = {
  spreadBps: number;
  liquidity: number;
  queueSkew: number;
  tradePressure: number;
  cancelPressure: number;
  volatility: number;
  activity: number;
  seed: number;
};

export type SyntheticSnapshot = {
  timeSeconds: number;
  eventType: 1 | 2 | 3 | 4 | 5;
  direction: -1 | 1;
  size: number;
  bidPrices: number[];
  askPrices: number[];
  bidSizes: number[];
  askSizes: number[];
};

export const presets: Record<string, ScenarioControls> = {
  "Balanced market": { spreadBps: 1.8, liquidity: 0.62, queueSkew: 0, tradePressure: 0, cancelPressure: 0, volatility: 0.35, activity: 0.58, seed: 11 },
  "Buy-pressure buildup": { spreadBps: 1.5, liquidity: 0.55, queueSkew: 0.34, tradePressure: 0.62, cancelPressure: 0.25, volatility: 0.46, activity: 0.72, seed: 23 },
  "Sell-pressure buildup": { spreadBps: 1.7, liquidity: 0.58, queueSkew: -0.38, tradePressure: -0.66, cancelPressure: -0.18, volatility: 0.48, activity: 0.73, seed: 31 },
  "Thin and volatile": { spreadBps: 3.8, liquidity: 0.18, queueSkew: 0.08, tradePressure: 0.2, cancelPressure: 0.4, volatility: 0.9, activity: 0.82, seed: 47 },
  "Deep and quiet": { spreadBps: 1.1, liquidity: 0.92, queueSkew: 0.02, tradePressure: 0.05, cancelPressure: -0.05, volatility: 0.12, activity: 0.32, seed: 59 },
  "Cancellation shock": { spreadBps: 2.6, liquidity: 0.34, queueSkew: -0.12, tradePressure: -0.15, cancelPressure: 0.88, volatility: 0.72, activity: 0.86, seed: 71 },
};

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normal(random: () => number) {
  const u = Math.max(random(), 1e-12);
  const v = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function chooseEventType(random: () => number, controls: ScenarioControls): 1 | 2 | 3 | 4 | 5 {
  const cancellation = 0.13 + 0.09 * Math.abs(controls.cancelPressure);
  const deletion = 0.07 + 0.05 * Math.abs(controls.cancelPressure);
  const execution = 0.23 + 0.1 * Math.abs(controls.tradePressure);
  const hidden = 0.025;
  const draw = random();
  if (draw < cancellation) return 2;
  if (draw < cancellation + deletion) return 3;
  if (draw < cancellation + deletion + execution) return 4;
  if (draw < cancellation + deletion + execution + hidden) return 5;
  return 1;
}

function chooseDirection(
  random: () => number,
  eventType: number,
  controls: ScenarioControls,
): -1 | 1 {
  let bidProbability = 0.5;
  if (eventType === 1) bidProbability += controls.queueSkew * 0.25;
  if (eventType === 2 || eventType === 3) bidProbability -= controls.cancelPressure * 0.28;
  if (eventType === 4 || eventType === 5) bidProbability -= controls.tradePressure * 0.3;
  return random() < clamp(bidProbability, 0.08, 0.92) ? 1 : -1;
}

export function generateScenario(
  controls: ScenarioControls,
  eventCount = 150,
): SyntheticSnapshot[] {
  if (eventCount < 120) throw new Error("At least 120 events are required for 100-event features.");
  const random = mulberry32(controls.seed);
  const tick = 0.01;
  let mid = 100;
  let time = 34_200;
  const baseDepth = 160 + controls.liquidity * 1_840;
  let bidSizes = Array.from({ length: 10 }, (_, index) => Math.round(baseDepth * (1 + index * 0.12) * (1 + controls.queueSkew * 0.32)));
  let askSizes = Array.from({ length: 10 }, (_, index) => Math.round(baseDepth * (1 + index * 0.12) * (1 - controls.queueSkew * 0.32)));
  const rows: SyntheticSnapshot[] = [];

  for (let index = 0; index < eventCount; index += 1) {
    const eventType = chooseEventType(random, controls);
    const direction = chooseDirection(random, eventType, controls);
    const size = Math.max(1, Math.round((18 + random() * 145) * (0.5 + controls.liquidity)));
    const level = Math.min(9, Math.floor(Math.pow(random(), 2.3) * 10));
    const side = direction === 1 ? bidSizes : askSizes;

    if (eventType === 1) side[level] += size;
    if (eventType === 2) side[level] = Math.max(1, side[level] - Math.max(1, Math.round(size * 0.55)));
    if (eventType === 3) side[level] = Math.max(1, side[level] - size);
    if (eventType === 4) side[Math.min(level, 2)] = Math.max(1, side[Math.min(level, 2)] - size);

    const pressureDrift = controls.tradePressure * 0.00042 + controls.cancelPressure * 0.00018;
    const imbalance = (bidSizes[0] - askSizes[0]) / Math.max(1, bidSizes[0] + askSizes[0]);
    const randomMove = normal(random) * (0.00008 + controls.volatility * 0.00072);
    const moveTrigger = Math.abs(randomMove + pressureDrift + imbalance * 0.00012);
    if (moveTrigger > 0.00035 || random() < controls.volatility * 0.08) {
      const sign = randomMove + pressureDrift + imbalance * 0.00012 >= 0 ? 1 : -1;
      mid = Math.max(10, mid + sign * tick * (random() < controls.volatility * 0.18 ? 2 : 1));
    }

    const spreadPrice = Math.max(tick, Math.round((mid * controls.spreadBps / 10_000) / tick) * tick);
    const halfSpread = spreadPrice / 2;
    const bestBid = Math.floor((mid - halfSpread) / tick) * tick;
    const bestAsk = Math.ceil((mid + halfSpread) / tick) * tick;
    const bidPrices = Array.from({ length: 10 }, (_, depth) => Number((bestBid - depth * tick).toFixed(2)));
    const askPrices = Array.from({ length: 10 }, (_, depth) => Number((bestAsk + depth * tick).toFixed(2)));

    const meanReversion = 0.025;
    bidSizes = bidSizes.map((value, depth) => Math.max(1, Math.round(value + (baseDepth * (1 + depth * 0.12) * (1 + controls.queueSkew * 0.32) - value) * meanReversion)));
    askSizes = askSizes.map((value, depth) => Math.max(1, Math.round(value + (baseDepth * (1 + depth * 0.12) * (1 - controls.queueSkew * 0.32) - value) * meanReversion)));

    const averageGap = 0.9 - controls.activity * 0.82;
    time += Math.max(0.00005, averageGap * (0.35 + random() * 1.3));
    rows.push({
      timeSeconds: time,
      eventType,
      direction,
      size,
      bidPrices,
      askPrices,
      bidSizes: [...bidSizes],
      askSizes: [...askSizes],
    });
  }
  return rows;
}
