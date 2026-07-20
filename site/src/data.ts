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
  n: number
  eceBefore: number
  eceAfter: number
  local: number
  cloud: number
  hybrid: number
  hybridLo: number
  hybridHi: number
}

// results_multi.md — per-category, independently trained.
export const CATEGORIES: CategoryRow[] = [
  { category: 'bottle', n: 61, eceBefore: 0.059, eceAfter: 0.05, local: 0.908, cloud: 0.997, hybrid: 0.988, hybridLo: 0.985, hybridHi: 0.991 },
  { category: 'grid', n: 69, eceBefore: 0.222, eceAfter: 0.221, local: 0.811, cloud: 0.997, hybrid: 0.939, hybridLo: 0.936, hybridHi: 0.943 },
  { category: 'metal_nut', n: 69, eceBefore: 0.167, eceAfter: 0.096, local: 0.951, cloud: 0.998, hybrid: 0.974, hybridLo: 0.944, hybridHi: 0.989 },
  { category: 'screw', n: 98, eceBefore: 0.224, eceAfter: 0.214, local: 0.873, cloud: 0.963, hybrid: 0.97, hybridLo: 0.969, hybridHi: 0.972 },
  { category: 'cable', n: 76, eceBefore: 0.163, eceAfter: 0.155, local: 0.883, cloud: 0.998, hybrid: 0.963, hybridLo: 0.961, hybridHi: 0.966 },
  { category: 'capsule', n: 72, eceBefore: 0.147, eceAfter: 0.138, local: 0.921, cloud: 0.998, hybrid: 0.98, hybridLo: 0.978, hybridHi: 0.982 },
]

export const AGGREGATE = {
  hybridMean: 0.969,
  hybridStd: 0.015,
  localMean: 0.891,
  lift: 0.078,
  nCategories: 6,
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
  { file: 'capsule-crack.png', label: 'Cracked capsule', category: 'capsule', expect: 'crack' },
]
