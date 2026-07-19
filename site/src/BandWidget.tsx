import { useMemo, useState } from 'react'
import { escalationBand } from './data'
import './BandWidget.css'

type Costs = { C_FN: number; C_FP: number; C_cloud: number; epsilon: number }

const SLIDERS: {
  key: keyof Costs
  label: string
  hint: string
  min: number
  max: number
  step: number
}[] = [
  { key: 'C_FN', label: 'C_FN', hint: 'missed defect', min: 10, max: 300, step: 1 },
  { key: 'C_FP', label: 'C_FP', hint: 'false alarm', min: 1, max: 50, step: 0.5 },
  { key: 'C_cloud', label: 'C_cloud', hint: 'one escalation', min: 0.5, max: 20, step: 0.5 },
  { key: 'epsilon', label: 'ε', hint: 'residual cloud error', min: 0, max: 5, step: 0.1 },
]

// A few illustrative parts placed along the 0->1 probability axis.
const PARTS = [0.01, 0.12, 0.3, 0.48, 0.92]

function pct(x: number) {
  return `${Math.max(0, Math.min(1, x)) * 100}%`
}

export default function BandWidget() {
  const [costs, setCosts] = useState<Costs>({ C_FN: 100, C_FP: 5, C_cloud: 2, epsilon: 0.3 })
  const { pLo, pHi, pStar, valid } = useMemo(() => escalationBand(costs), [costs])

  const T = costs.C_cloud + costs.epsilon
  const loClamped = Math.max(0, pLo)
  const hiClamped = Math.min(1, pHi)

  function classify(p: number): 'pass' | 'escalate' | 'reject' {
    if (!valid) return p < pStar ? 'pass' : 'reject'
    if (p < loClamped) return 'pass'
    if (p > hiClamped) return 'reject'
    return 'escalate'
  }

  return (
    <div className="band">
      <div className="band-track" role="img"
        aria-label={`Escalation band from p=${loClamped.toFixed(3)} to p=${hiClamped.toFixed(3)}`}>
        {valid && (
          <div className="band-zone" style={{ left: pct(loClamped), width: pct(hiClamped - loClamped) }}>
            <span className="band-zone-label">escalate</span>
          </div>
        )}
        <div className="band-side band-pass" style={{ width: pct(loClamped) }}>
          {loClamped > 0.12 && <span>pass locally</span>}
        </div>
        <div className="band-side band-reject" style={{ left: pct(hiClamped), width: pct(1 - hiClamped) }}>
          {1 - hiClamped > 0.12 && <span>reject locally</span>}
        </div>

        {valid && (
          <>
            <Thresh x={loClamped} label="p_lo" value={pLo} />
            <Thresh x={hiClamped} label="p_hi" value={pHi} />
          </>
        )}

        {PARTS.map((p) => (
          <div key={p} className={`part part-${classify(p)}`} style={{ left: pct(p) }} title={`p=${p}`}>
            <span className="part-p">{p}</span>
          </div>
        ))}

        <div className="band-axis">
          <span>0.0</span>
          <span className="band-axis-mid">defect probability p</span>
          <span>1.0</span>
        </div>
      </div>

      <div className="band-readout mono">
        <span>T = C_cloud + ε = <b>{T.toFixed(2)}</b></span>
        <span>p_lo = T/C_FN = <b>{pLo.toFixed(3)}</b></span>
        <span>p_hi = 1 − T/C_FP = <b>{valid ? pHi.toFixed(3) : '—'}</b></span>
        {!valid && <span className="band-warn">band empty: no item is worth escalating</span>}
      </div>

      <div className="band-sliders">
        {SLIDERS.map((s) => (
          <label key={s.key} className="slider">
            <span className="slider-top">
              <span className="slider-label mono">{s.label}</span>
              <span className="slider-val mono">{costs[s.key]}</span>
            </span>
            <input
              type="range"
              min={s.min}
              max={s.max}
              step={s.step}
              value={costs[s.key]}
              onChange={(e) => setCosts((c) => ({ ...c, [s.key]: Number(e.target.value) }))}
              aria-label={`${s.label}, ${s.hint}`}
            />
            <span className="slider-hint">{s.hint}</span>
          </label>
        ))}
      </div>
      <p className="band-caption">
        Drag the costs. The gate is <em>derived</em>, never hand-tuned: raise the price of a
        missed defect and the band widens to catch more; raise the cloud toll and it narrows to
        escalate only the genuinely uncertain.
      </p>
    </div>
  )
}

function Thresh({ x, label, value }: { x: number; label: string; value: number }) {
  return (
    <div className="thresh" style={{ left: pct(x) }}>
      <span className="thresh-tag mono">
        {label} <b>{value.toFixed(3)}</b>
      </span>
    </div>
  )
}
