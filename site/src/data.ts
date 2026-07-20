// Single source of truth for the page. Every number here is copied from the repo's real
// eval output (eval/results_table_real.md, results_multi.md, backbone_ablation.md) or the
// router's cost math (edge/router.py). No invented figures.

export const COSTS = {
  C_FN: 100.0, // false negative (defect shipped) — large
  C_FP: 5.0, // false positive (good part rejected) — small
  C_cloud: 2.0, // one escalation
  epsilon: 0.3, // residual_cloud_error
} as const

// Escalation band, derived exactly as edge/router.py does:
//   T   = C_cloud + epsilon
//   p_lo = T / C_FN         (rising branch)
//   p_hi = 1 - T / C_FP     (falling branch)
// A part with p in [p_lo, p_hi] is uncertain enough that a cloud call is worth its cost.
export function escalationBand(c: {
  C_FN: number
  C_FP: number
  C_cloud: number
  epsilon: number
}): { pLo: number; pHi: number; pStar: number; valid: boolean } {
  const T = c.C_cloud + c.epsilon
  const pLo = T / c.C_FN
  const pHi = 1 - T / c.C_FP
  const pStar = c.C_FP / (c.C_FP + c.C_FN) // cost-optimal local boundary
  return { pLo, pHi, pStar, valid: pHi > pLo }
}

export type Row = {
  condition: string
  recall: number | null
  recallLo?: number
  recallHi?: number
  p50: number | null
  p99: number | null
  bytes: number | null
  costPer1k: number | null
  pii: number
  us?: boolean
}

// results_table_real.md — bottle, real local model + fitted calibration, modeled cloud.
export const HEADLINE_ROWS: Row[] = [
  { condition: 'Cloud-everything', recall: 0.998, recallLo: 0.996, recallHi: 0.999, p50: 0.2, p99: 0.3, bytes: 54, costPer1k: 2000.0, pii: 0 },
  { condition: 'Hybrid (ours)', recall: 0.988, recallLo: 0.987, recallHi: 0.99, p50: 0.2, p99: 0.3, bytes: 31, costPer1k: 1147.54, pii: 0, us: true },
  { condition: 'Local-only', recall: 0.908, recallLo: 0.908, recallHi: 0.908, p50: 0.0, p99: 0.0, bytes: 0, costPer1k: 0.0, pii: 0 },
  { condition: 'Hybrid (degraded)', recall: 0.908, p50: 0.4, p99: 0.6, bytes: 0, costPer1k: 0.0, pii: 0 },
  { condition: 'Hybrid (offline)', recall: 0.908, p50: 0.4, p99: 0.6, bytes: 0, costPer1k: 0.0, pii: 0 },
]

// Live cloud, measured on real Qwen-VL calls (results_table_real.md note).
export const LIVE_CLOUD = {
  calls: 12,
  accuracy: 1.0,
  p50ms: 3676,
  p99ms: 12443,
} as const

export type CategoryRow = {
  category: string
  dataset: 'AD' | 'AD2'
  n: number
  escalation: number
  local: number
  hybrid: number
}

// Per-category results on real MVTec AD (15) and MVTec AD 2 (8) with the DINOv2 backbone.
// Unsupervised anomaly score (fit the normal distribution, calibrate to p), same cost router
// everywhere. local = local-only recall, hybrid = MEASURED with real qwen3-vl-plus verdicts on
// the escalated frames, escalation = fraction routed to the cloud. From eval/results_ad{,2}.md
// (job 54345293, GPU DINOv2, parallel cloud, 0 failures).
export const CATEGORIES: CategoryRow[] = [
  { category: 'bottle', dataset: 'AD', n: 42, escalation: 0.07, local: 1.0, hybrid: 1.0 },
  { category: 'cable', dataset: 'AD', n: 75, escalation: 0.39, local: 0.89, hybrid: 0.96 },
  { category: 'capsule', dataset: 'AD', n: 67, escalation: 0.45, local: 0.67, hybrid: 0.76 },
  { category: 'carpet', dataset: 'AD', n: 59, escalation: 0.29, local: 0.89, hybrid: 0.89 },
  { category: 'grid', dataset: 'AD', n: 40, escalation: 0.15, local: 0.9, hybrid: 0.97 },
  { category: 'hazelnut', dataset: 'AD', n: 55, escalation: 0.33, local: 0.89, hybrid: 0.97 },
  { category: 'leather', dataset: 'AD', n: 62, escalation: 0.02, local: 1.0, hybrid: 1.0 },
  { category: 'metal_nut', dataset: 'AD', n: 58, escalation: 0.22, local: 0.89, hybrid: 0.91 },
  { category: 'pill', dataset: 'AD', n: 84, escalation: 0.25, local: 0.85, hybrid: 0.97 },
  { category: 'screw', dataset: 'AD', n: 81, escalation: 0.51, local: 0.62, hybrid: 0.77 },
  { category: 'tile', dataset: 'AD', n: 59, escalation: 0.08, local: 0.98, hybrid: 1.0 },
  { category: 'toothbrush', dataset: 'AD', n: 21, escalation: 0.24, local: 0.93, hybrid: 0.93 },
  { category: 'transistor', dataset: 'AD', n: 50, escalation: 0.64, local: 0.8, hybrid: 0.95 },
  { category: 'wood', dataset: 'AD', n: 40, escalation: 0.28, local: 0.9, hybrid: 1.0 },
  { category: 'zipper', dataset: 'AD', n: 76, escalation: 0.16, local: 0.92, hybrid: 0.93 },
  { category: 'can', dataset: 'AD2', n: 81, escalation: 0.8, local: 0.18, hybrid: 0.18 },
  { category: 'fabric', dataset: 'AD2', n: 78, escalation: 0.88, local: 0.2, hybrid: 0.2 },
  { category: 'fruit_jelly', dataset: 'AD2', n: 40, escalation: 0.25, local: 0.87, hybrid: 0.87 },
  { category: 'rice', dataset: 'AD2', n: 66, escalation: 0.65, local: 0.33, hybrid: 0.33 },
  { category: 'sheet_metal', dataset: 'AD2', n: 57, escalation: 0.58, local: 0.38, hybrid: 0.44 },
  { category: 'vial', dataset: 'AD2', n: 71, escalation: 0.38, local: 0.75, hybrid: 0.79 },
  { category: 'wallplugs', dataset: 'AD2', n: 75, escalation: 0.47, local: 0.53, hybrid: 0.53 },
  { category: 'walnuts', dataset: 'AD2', n: 75, escalation: 0.67, local: 0.53, hybrid: 0.76 },
]

// Split and pooled aggregates. r = Pearson correlation of local recall vs. escalation rate:
// strongly negative means the router escalates in proportion to local uncertainty.
export const AGGREGATE = {
  ad: { local: 0.875, hybrid: 0.934, escalation: 0.272, r: -0.8, n: 15 },
  ad2: { local: 0.471, hybrid: 0.512, escalation: 0.585, r: -0.94, n: 8 },
  all: { local: 0.735, hybrid: 0.787, escalation: 0.381, r: -0.91, n: 23 },
} as const

export type AblationRow = {
  category: string
  localHand: number
  localMobile: number
  localDino: number
  hybridHand: number
  hybridMobile: number
  hybridDino: number
}

// backbone_ablation.md — same head, only the frozen backbone changes across a weak
// (handcrafted), medium (MobileNetV2), and SOTA (DINOv2) extractor. 6 categories.
export const ABLATION: AblationRow[] = [
  { category: 'bottle', localHand: 0.908, localMobile: 0.982, localDino: 0.988, hybridHand: 0.988, hybridMobile: 1.0, hybridDino: 1.0 },
  { category: 'grid', localHand: 0.811, localMobile: 0.808, localDino: 0.993, hybridHand: 0.939, hybridMobile: 0.916, hybridDino: 1.0 },
  { category: 'metal_nut', localHand: 0.951, localMobile: 0.92, localDino: 0.976, hybridHand: 0.974, hybridMobile: 0.982, hybridDino: 0.998 },
  { category: 'screw', localHand: 0.873, localMobile: 0.888, localDino: 0.941, hybridHand: 0.97, hybridMobile: 0.993, hybridDino: 0.993 },
  { category: 'cable', localHand: 0.883, localMobile: 0.874, localDino: 0.913, hybridHand: 0.963, hybridMobile: 0.945, hybridDino: 0.995 },
  { category: 'capsule', localHand: 0.921, localMobile: 0.955, localDino: 0.906, hybridHand: 0.98, hybridMobile: 0.961, hybridDino: 0.994 },
]

// Max spread of each metric across the three backbones, within a category.
export const ABLATION_DELTAS = {
  localSpread: 0.185,
  hybridSpread: 0.084,
} as const

// MVTec LOCO — logical vs. structural anomalies. Unsupervised anomaly score (train on good
// only), hybrid recall MEASURED with real qwen3-vl-plus verdicts. From eval/results_loco_final.md.
export type LocoRow = {
  category: string
  kind: 'logical' | 'structural'
  n: number
  escalation: number
  localRecall: number
  hybridRecall: number
}
export const LOCO: LocoRow[] = [
  { category: 'breakfast_box', kind: 'logical', n: 42, escalation: 0.29, localRecall: 0.71, hybridRecall: 1.0 },
  { category: 'breakfast_box', kind: 'structural', n: 45, escalation: 0.78, localRecall: 0.22, hybridRecall: 1.0 },
  { category: 'juice_bottle', kind: 'logical', n: 70, escalation: 0.31, localRecall: 0.69, hybridRecall: 0.69 },
  { category: 'juice_bottle', kind: 'structural', n: 48, escalation: 0.73, localRecall: 0.27, hybridRecall: 0.4 },
  { category: 'pushpins', kind: 'logical', n: 45, escalation: 0.49, localRecall: 0.49, hybridRecall: 0.98 },
  { category: 'pushpins', kind: 'structural', n: 41, escalation: 0.02, localRecall: 0.98, hybridRecall: 1.0 },
  { category: 'screw_bag', kind: 'logical', n: 66, escalation: 0.56, localRecall: 0.44, hybridRecall: 1.0 },
  { category: 'screw_bag', kind: 'structural', n: 44, escalation: 0.45, localRecall: 0.55, hybridRecall: 0.98 },
  { category: 'splicing_connectors', kind: 'logical', n: 52, escalation: 0.62, localRecall: 0.38, hybridRecall: 0.46 },
  { category: 'splicing_connectors', kind: 'structural', n: 45, escalation: 0.69, localRecall: 0.31, hybridRecall: 0.4 },
]
export const LOCO_AGG = {
  logical: { escalation: 0.45, localRecall: 0.54, hybridRecall: 0.83 },
  structural: { escalation: 0.53, localRecall: 0.47, hybridRecall: 0.75 },
} as const

export const PIPELINE = [
  { key: 'perceive', name: 'Perceive', detail: 'ONNX classifier gives a calibrated defect probability p in milliseconds.' },
  { key: 'route', name: 'Route', detail: 'The cost inequality decides: in-band escalates, out-of-band acts locally.' },
  { key: 'privacy', name: 'Filter', detail: 'On escalation, only the cropped ROI leaves the device. Skin-tone regions are blurred; raw frames and biometrics never cross.' },
  { key: 'cloud', name: 'Reason', detail: 'Qwen-VL diagnoses the ROI and returns a structured verdict over HTTP or MCP.' },
  { key: 'act', name: 'Act', detail: 'The relay fires. Actuation never blocks on the cloud, so the line never stalls.' },
  { key: 'log', name: 'Log', detail: 'Every boundary crossing is recorded, making the zero-PII claim measurable.' },
] as const

export const LINKS = {
  repo: 'https://github.com/SuryanshSS1011/edge-inspection-agent',
  // Fill these in before submission:
  video: '#',
  cloudBase: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
} as const

// The deployed reasoning server the "try it" box calls. Must be HTTPS for a browser on an
// HTTPS page (see docs/SAS_HTTPS.md for the Caddy + DuckDNS setup). Override at build time
// with VITE_DIAGNOSE_URL; empty string disables the live box gracefully.
export const DIAGNOSE_URL = (import.meta.env.VITE_DIAGNOSE_URL ?? '').replace(/\/$/, '')

// Sample ROIs shown in the live-diagnose box. Files live in site/public/samples/ and are
// real MVTec test images (CC BY-NC-SA, shown for the hackathon demo). Add/rename to match
// whatever you drop into that folder.
export type Sample = { file: string; label: string; category: string; expect: string }
export const SAMPLES: Sample[] = [
  { file: 'bottle-good.png', label: 'Clean bottle', category: 'bottle', expect: 'defect-free' },
  { file: 'bottle-broken.png', label: 'Broken bottle', category: 'bottle', expect: 'broken glass' },
  { file: 'capsule-defect.png', label: 'Contaminated capsule', category: 'capsule', expect: 'contamination' },
]
