import { useMemo, useRef, useState } from 'react'
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
  { key: 'C_FN', label: 'C_FN', hint: 'cost of a missed defect', min: 10, max: 300, step: 1 },
  { key: 'C_FP', label: 'C_FP', hint: 'cost of a false alarm', min: 1, max: 50, step: 0.5 },
  { key: 'C_cloud', label: 'C_cloud', hint: 'cost of one escalation', min: 0.5, max: 20, step: 0.5 },
  { key: 'epsilon', label: 'ε', hint: 'residual cloud error', min: 0, max: 5, step: 0.1 },
]

// Named example parts, so each marker on the axis reads as a concrete case.
const PARTS = [
  { p: 0.02, label: 'clean part' },
  { p: 0.3, label: 'ambiguous part' },
  { p: 0.92, label: 'clear defect' },
]

const pct = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`

type Zone = 'pass' | 'escalate' | 'reject'
const ZONE_LABEL: Record<Zone, string> = {
  pass: 'Pass locally',
  escalate: 'Escalate to cloud',
  reject: 'Reject locally',
}

export default function BandWidget() {
  const [costs, setCosts] = useState<Costs>({ C_FN: 100, C_FP: 5, C_cloud: 2, epsilon: 0.3 })
  const [probe, setProbe] = useState(0.3)
  const trackRef = useRef<HTMLDivElement>(null)
  const { pLo, pHi, pStar, valid } = useMemo(() => escalationBand(costs), [costs])

  const T = costs.C_cloud + costs.epsilon
  const lo = Math.max(0, pLo)
  const hi = Math.min(1, pHi)

  function classify(p: number): Zone {
    if (!valid) return p < pStar ? 'pass' : 'reject'
    if (p < lo) return 'pass'
    if (p > hi) return 'reject'
    return 'escalate'
  }

  const probeZone = classify(probe)

  // Drag the probe along the track.
  function moveProbe(clientX: number) {
    const el = trackRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setProbe(Math.max(0, Math.min(1, (clientX - r.left) / r.width)))
  }
  function onPointerDown(e: React.PointerEvent) {
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    moveProbe(e.clientX)
  }
  function onPointerMove(e: React.PointerEvent) {
    if (e.buttons === 1) moveProbe(e.clientX)
  }
  function onKey(e: React.KeyboardEvent) {
    if (e.key === 'ArrowLeft') setProbe((p) => Math.max(0, p - 0.01))
    if (e.key === 'ArrowRight') setProbe((p) => Math.min(1, p + 0.01))
  }

  return (
    <div className="band">
      {/* legend */}
      <div className="band-legend">
        <span className="lg-item"><i className="sw sw-pass" /> Pass locally</span>
        <span className="lg-item"><i className="sw sw-escalate" /> Escalate to cloud</span>
        <span className="lg-item"><i className="sw sw-reject" /> Reject locally</span>
      </div>

      {/* the axis */}
      <div
        className="band-track"
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(probe.toFixed(2))}
        aria-label="defect probability probe"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onKeyDown={onKey}
      >
        <div className="zone zone-pass" style={{ width: pct(lo) }} />
        {valid && <div className="zone zone-escalate" style={{ left: pct(lo), width: pct(hi - lo) }} />}
        <div className="zone zone-reject" style={{ left: pct(valid ? hi : pStar), right: 0 }} />

        {/* thresholds */}
        {valid && <Threshold x={lo} label="p_lo" />}
        {valid && <Threshold x={hi} label="p_hi" />}

        {/* named example parts as marker lines with a label at the top edge */}
        {PARTS.map((part, i) => {
          const edge = i === 0 ? 'mark-first' : i === PARTS.length - 1 ? 'mark-last' : ''
          return (
            <div key={part.label} className={`pmark pmark-${classify(part.p)} ${edge}`} style={{ left: pct(part.p) }}>
              <span className="pmark-tag">
                <span className="pmark-name">{part.label}</span>
                <span className="pmark-p mono">p = {part.p}</span>
              </span>
            </div>
          )
        })}

        {/* draggable probe */}
        <div className={`probe probe-${probeZone}`} style={{ left: pct(probe) }}>
          <span className="probe-handle" />
        </div>
      </div>

      {/* axis ticks */}
      <div className="band-axis">
        <span>0.0</span>
        <span className="axis-title">defect probability p</span>
        <span>1.0</span>
      </div>

      {/* live probe readout */}
      <div className="probe-readout">
        <span>Drag the marker:</span>
        <span className="mono probe-p">p = {probe.toFixed(2)}</span>
        <span className="probe-arrow">→</span>
        <span className={`probe-decision d-${probeZone}`}>{ZONE_LABEL[probeZone]}</span>
        {!valid && <span className="band-warn">band empty, no item is worth escalating</span>}
      </div>

      {/* the derived thresholds */}
      <div className="band-derived mono">
        <span>T = C_cloud + ε = <b>{T.toFixed(2)}</b></span>
        <span>p_lo = T / C_FN = <b>{pLo.toFixed(3)}</b></span>
        <span>p_hi = 1 − T / C_FP = <b>{valid ? pHi.toFixed(3) : '—'}</b></span>
      </div>

      {/* cost sliders */}
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
    </div>
  )
}

function Threshold({ x, label }: { x: number; label: string }) {
  return (
    <div className="thresh" style={{ left: pct(x) }}>
      <span className="thresh-tag mono">{label}</span>
    </div>
  )
}
