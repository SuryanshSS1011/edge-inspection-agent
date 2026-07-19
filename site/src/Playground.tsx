import { useEffect, useMemo, useRef, useState } from 'react'
import { DIAGNOSE_URL } from './data'
import './Playground.css'

// Shape of site/public/playground/runs.json, produced by eval/capture_playground.py.
type Cloud = {
  defect_present: boolean
  defect_type: string
  confidence: number
  root_cause: string
  recommended_action: string
} | null

type ModeRun = {
  network: string
  p: number
  in_band: boolean
  decision: string
  action: string
  bytes_to_cloud?: number
  pii_bytes?: number
  cloud?: Cloud
  outbox_state?: string
  reconciled?: Cloud
}

type Run = {
  key: string
  caption: string
  image: string
  uncertainty: number
  full: ModeRun
  offline: ModeRun
}

type Data = {
  costs: { C_FN: number; C_FP: number; C_cloud: number; epsilon: number }
  band: { lo: number; hi: number } | null
  pstar: number
  runs: Run[]
}

type Net = 'full' | 'offline'
type Stage = 'idle' | 'perceive' | 'route' | 'privacy' | 'cloud' | 'act' | 'done'

const STAGES: { key: Stage; label: string; num: string }[] = [
  { key: 'perceive', label: 'Perceive', num: '01' },
  { key: 'route', label: 'Route', num: '02' },
  { key: 'privacy', label: 'Filter', num: '03' },
  { key: 'cloud', label: 'Reason', num: '04' },
  { key: 'act', label: 'Act', num: '05' },
]

const base = (p: string) => `${import.meta.env.BASE_URL}${p.replace(/^\//, '')}`

export default function Playground() {
  const [data, setData] = useState<Data | null>(null)
  const [failed, setFailed] = useState(false)
  const [net, setNet] = useState<Net>('full')
  const [selected, setSelected] = useState<Run | null>(null)
  const [stage, setStage] = useState<Stage>('idle')
  const [liveCloud, setLiveCloud] = useState<Cloud | 'pending' | null>(null)
  const timers = useRef<number[]>([])

  useEffect(() => {
    fetch(base('playground/runs.json'))
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setFailed(true))
    return () => timers.current.forEach(clearTimeout)
  }, [])

  const modeRun = useMemo<ModeRun | null>(
    () => (selected ? selected[net] : null),
    [selected, net],
  )

  function run(part: Run) {
    timers.current.forEach(clearTimeout)
    timers.current = []
    setSelected(part)
    setLiveCloud(null)
    setStage('idle')
    const mr = part[net]
    // Advance the pipeline stages with small delays so the flow is legible.
    const seq: Stage[] = ['perceive', 'route']
    if (mr.in_band) seq.push('privacy', 'cloud')
    seq.push('act', 'done')
    seq.forEach((s, i) => {
      const t = window.setTimeout(() => {
        setStage(s)
        if (s === 'cloud' && net === 'full') triggerLiveCloud(part)
      }, 600 * (i + 1))
      timers.current.push(t)
    })
  }

  async function triggerLiveCloud(part: Run) {
    // On escalation in FULL mode, actually call the deployed reasoner. Fall back to the
    // captured verdict if the endpoint is not configured or unreachable.
    const captured = part.full.cloud ?? null
    if (!DIAGNOSE_URL) {
      setLiveCloud(captured)
      return
    }
    setLiveCloud('pending')
    try {
      const img = await fetch(base(part.image))
        .then((r) => r.blob())
        .then(blobToB64)
      const res = await fetch(`${DIAGNOSE_URL}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi_png_b64: img, context: { category: part.key, note: 'inspect count/arrangement too' } }),
      })
      setLiveCloud(res.ok ? await res.json() : captured)
    } catch {
      setLiveCloud(captured) // endpoint down: show the real captured verdict, labeled
    }
  }

  if (failed) {
    return (
      <section id="playground">
        <div className="wrap">
          <p className="eyebrow">Playground</p>
          <h2 className="section-title">Interactive demo</h2>
          <p className="pg-note">Captured runs not found. Run <code>eval.capture_playground</code> to generate them.</p>
        </div>
      </section>
    )
  }
  if (!data) {
    return <section id="playground"><div className="wrap"><p className="pg-note">Loading playground…</p></div></section>
  }

  const cloudShown = liveCloud === 'pending' ? null : (liveCloud ?? modeRun?.cloud ?? null)
  const usedLive = DIAGNOSE_URL && liveCloud && liveCloud !== 'pending'

  return (
    <section id="playground">
      <div className="wrap">
        <p className="eyebrow">Playground · drive the pipeline</p>
        <h2 className="section-title">Run a part through the gate</h2>
        <p className="section-lead">
          Pick a part and watch it flow through the real pipeline. The edge decision (local{' '}
          <code>p</code>, routing, byte and PII counts) is <strong>replayed from a real
          captured run</strong>; on escalation the cloud reasoning is a <strong>live call</strong>{' '}
          to the deployed qwen3-vl-plus server. Flip the network to see it keep working offline.
        </p>

        <div className="pg-controls">
          <div className="pg-parts">
            {data.runs.map((r) => (
              <button
                key={r.key}
                className={`pg-part ${selected?.key === r.key ? 'is-active' : ''}`}
                onClick={() => run(r)}
              >
                <img src={base(r.image)} alt={r.caption} onError={(e) => ((e.target as HTMLImageElement).style.visibility = 'hidden')} />
                <span>{r.caption}</span>
              </button>
            ))}
          </div>
          <div className="pg-net">
            <span className="pg-net-label mono">network</span>
            {(['full', 'offline'] as Net[]).map((n) => (
              <button
                key={n}
                className={`pg-net-btn ${net === n ? 'is-on' : ''} pg-net-${n}`}
                onClick={() => { setNet(n); if (selected) run({ ...selected }) }}
              >
                {n === 'full' ? 'FULL' : 'OFFLINE'}
              </button>
            ))}
          </div>
        </div>

        {!selected && <p className="pg-hint">Pick a part above to run it.</p>}

        {selected && modeRun && (
          <div className="pg-stage-area">
            <Band data={data} p={modeRun.p} inBand={modeRun.in_band} active={stageIdx(stage) >= 1} />
            <ol className="pg-pipeline">
              {STAGES.map((s) => {
                const shown = isStageShown(s.key, modeRun)
                const active = stage === s.key
                const done = stageIdx(stage) > stageIdx(s.key) || stage === 'done'
                return (
                  <li key={s.key} className={`pg-step ${active ? 'active' : ''} ${done && shown ? 'done' : ''} ${!shown ? 'skipped' : ''}`}>
                    <span className="pg-step-num mono">{s.num}</span>
                    <span className="pg-step-label">{s.label}</span>
                    <span className="pg-step-detail mono">
                      {stageDetail(s.key, modeRun, cloudShown, liveCloud === 'pending')}
                    </span>
                  </li>
                )
              })}
            </ol>
            {(stage === 'done' || stage === 'act') && (
              <Verdict modeRun={modeRun} net={net} cloud={cloudShown} usedLive={!!usedLive} />
            )}
          </div>
        )}
      </div>
    </section>
  )
}

function stageIdx(s: Stage): number {
  const order: Stage[] = ['idle', 'perceive', 'route', 'privacy', 'cloud', 'act', 'done']
  return order.indexOf(s)
}

function isStageShown(key: Stage, mr: ModeRun): boolean {
  if (key === 'privacy' || key === 'cloud') return mr.in_band && mr.network === 'full'
  return true
}

function stageDetail(key: Stage, mr: ModeRun, cloud: Cloud, pending: boolean): string {
  switch (key) {
    case 'perceive':
      return `p = ${mr.p.toFixed(3)}`
    case 'route':
      return mr.in_band ? `in-band → ${mr.decision}` : `out-of-band → ${mr.decision}`
    case 'privacy':
      return `ROI · ${mr.bytes_to_cloud ?? 0} bytes · PII ${mr.pii_bytes ?? 0}`
    case 'cloud':
      if (mr.network === 'offline') return 'unreachable → deferred'
      if (pending) return 'calling qwen3-vl-plus…'
      return cloud ? (cloud.defect_present ? `defect: ${cloud.defect_type}` : 'clean') : '—'
    case 'act':
      return mr.action
    default:
      return ''
  }
}

function Band({ data, p, inBand, active }: { data: Data; p: number; inBand: boolean; active: boolean }) {
  if (!data.band) return null
  const { lo, hi } = data.band
  const pct = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`
  const cls = inBand ? 'escalate' : p < lo ? 'pass' : 'reject'
  return (
    <div className={`pg-band ${active ? 'live' : ''}`}>
      <div className="pg-band-zone" style={{ left: pct(lo), width: pct(hi - lo) }}>escalate</div>
      <div className={`pg-band-marker pg-mark-${cls}`} style={{ left: pct(p) }}>
        <span className="mono">p={p.toFixed(2)}</span>
      </div>
      <div className="pg-band-axis"><span>0</span><span className="mono">defect probability</span><span>1</span></div>
    </div>
  )
}

function Verdict({ modeRun, net, cloud, usedLive }: { modeRun: ModeRun; net: Net; cloud: Cloud; usedLive: boolean }) {
  if (net === 'offline' && modeRun.in_band) {
    return (
      <div className="pg-verdict pg-offline">
        <div className="pg-verdict-head"><span className="pg-flag reject">DEFERRED</span></div>
        <p>Offline and uncertain: the line rejects conservatively and queues the escalation to the outbox
          (<span className="mono">{modeRun.outbox_state}</span>). On reconnect it drains and the cloud verdict
          back-fills the record{modeRun.reconciled ? `: ${modeRun.reconciled.defect_type}` : ''}.</p>
      </div>
    )
  }
  const defect = cloud?.defect_present ?? modeRun.action === 'REJECT'
  return (
    <div className={`pg-verdict ${defect ? 'pg-defect' : 'pg-clean'}`}>
      <div className="pg-verdict-head">
        <span className={`pg-flag ${defect ? 'reject' : 'pass'}`}>{modeRun.action}</span>
        {modeRun.in_band && (
          <span className="pg-src mono">{usedLive ? 'live qwen3-vl-plus' : 'captured verdict'}</span>
        )}
      </div>
      {cloud && (
        <dl className="pg-verdict-rows">
          <div><dt className="mono">type</dt><dd>{cloud.defect_type}</dd></div>
          <div><dt className="mono">reasoning</dt><dd>{cloud.root_cause}</dd></div>
        </dl>
      )}
      {!modeRun.in_band && <p className="pg-local">Decided locally — no cloud call, no bytes off-device.</p>}
    </div>
  )
}

function blobToB64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onloadend = () => resolve((r.result as string).split(',')[1] ?? '')
    r.onerror = () => reject(new Error('read failed'))
    r.readAsDataURL(blob)
  })
}
