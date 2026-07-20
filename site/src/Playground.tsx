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
// The escalation branch is now a live state machine, not a pre-baked record. An in-band part
// reaches 'reason' and then follows the CURRENT network: online resolves to a verdict; offline
// parks in the outbox; flipping back online mid-park drains it and back-fills the verdict.
type CloudState = 'idle' | 'pending' | 'resolved' | 'queued' | 'draining' | 'reconciled'

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
  const [cloudState, setCloudState] = useState<CloudState>('idle')
  const [cloud, setCloud] = useState<Cloud>(null)
  const timers = useRef<number[]>([])
  const netRef = useRef<Net>(net)
  netRef.current = net

  useEffect(() => {
    fetch(base('playground/runs.json'))
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setFailed(true))
    return () => timers.current.forEach(clearTimeout)
  }, [])

  // The edge decision (p, routing, bytes/PII) is identical regardless of the network, so it is
  // replayed from the captured FULL record. The network only changes the escalation branch.
  const modeRun = useMemo<ModeRun | null>(() => (selected ? selected.full : null), [selected])

  function run(part: Run) {
    timers.current.forEach(clearTimeout)
    timers.current = []
    setSelected(part)
    setCloud(null)
    setCloudState('idle')
    setStage('idle')
    const mr = part.full
    const seq: Stage[] = ['perceive', 'route']
    if (mr.in_band) seq.push('privacy', 'cloud')
    seq.push('act', 'done')
    seq.forEach((s, i) => {
      const t = window.setTimeout(() => {
        setStage(s)
        if (s === 'cloud') enterReason(part) // resolve against the live network at this moment
      }, 650 * (i + 1))
      timers.current.push(t)
    })
  }

  // Reached the escalation point. Branch on the CURRENT network, read live so a cut that
  // happened during the earlier stages is honored.
  function enterReason(part: Run) {
    if (netRef.current === 'offline') {
      setCloudState('queued') // park in the outbox; the run stays here until reconnect
    } else {
      resolveCloud(part)
    }
  }

  async function resolveCloud(part: Run) {
    const captured = part.full.cloud ?? null
    if (!DIAGNOSE_URL) {
      setCloud(captured)
      setCloudState('resolved')
      return
    }
    setCloudState('pending')
    try {
      const img = await fetch(base(part.image)).then((r) => r.blob()).then(blobToB64)
      const res = await fetch(`${DIAGNOSE_URL}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi_png_b64: img, context: { category: part.key } }),
      })
      setCloud(res.ok ? await res.json() : captured)
    } catch {
      setCloud(captured) // endpoint down: fall back to the real captured verdict
    }
    setCloudState('resolved')
  }

  // Live network toggle. This is the demonstrable cut: it does not restart the run, it acts on
  // the escalation branch in flight. Cutting the line while a call is queued keeps it queued;
  // restoring the line drains the outbox and reconciles the deferred verdict.
  function toggleNet(n: Net) {
    setNet(n)
    netRef.current = n
    if (!selected) return
    if (n === 'offline') {
      // A cut during a live call reverts that escalation to the queued (deferred) state.
      if (cloudState === 'pending' || cloudState === 'resolved') {
        setCloud(null)
        setCloudState('queued')
      }
    } else if (n === 'full' && cloudState === 'queued') {
      // Reconnect: drain the outbox and back-fill the reconciled verdict.
      setCloudState('draining')
      const t = window.setTimeout(() => {
        setCloud(selected.offline.reconciled ?? selected.full.cloud ?? null)
        setCloudState('reconciled')
      }, 900)
      timers.current.push(t)
    }
  }

  if (failed) {
    return (
      <section id="playground" className="col">
        <h2 className="sec-head"><span className="sec-num">3</span> Interactive pipeline demonstration</h2>
        <p className="pg-note">Captured runs not found. Run <code>eval.capture_playground</code> to generate them.</p>
      </section>
    )
  }
  if (!data) {
    return <section id="playground" className="col"><p className="pg-note">Loading playground…</p></section>
  }

  const usedLive = !!(DIAGNOSE_URL && cloudState === 'resolved' && cloud)
  const queued = cloudState === 'queued' || cloudState === 'draining'

  return (
    <section id="playground" className="col-wide">
      <div className="col-inner">
        <h2 className="sec-head"><span className="sec-num">3</span> Interactive pipeline demonstration</h2>
        <p>
          Pick a part and watch it flow through the real pipeline. The edge decision (local{' '}
          <code>p</code>, routing, byte and PII counts) is <strong>replayed from a real
          captured run</strong>; on escalation the cloud reasoning is a <strong>live call</strong>{' '}
          to the deployed Qwen3-VL server. Cut the network mid-run to watch an escalation defer
          to the outbox, then restore it to watch the queue drain and reconcile.
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
                onClick={() => toggleNet(n)}
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
                const active = stage === s.key || (s.key === 'cloud' && queued)
                const done = (stageIdx(stage) > stageIdx(s.key) || stage === 'done') && !(s.key === 'cloud' && queued)
                return (
                  <li key={s.key} className={`pg-step ${active ? 'active' : ''} ${done && shown ? 'done' : ''} ${!shown ? 'skipped' : ''} ${s.key === 'cloud' && queued ? 'queued' : ''}`}>
                    <span className="pg-step-num mono">{s.num}</span>
                    <span className="pg-step-label">{s.label}</span>
                    <span className="pg-step-detail mono">
                      {stageDetail(s.key, modeRun, cloud, cloudState)}
                    </span>
                  </li>
                )
              })}
            </ol>
            {(stage === 'done' || stage === 'act') && (
              <Verdict modeRun={modeRun} cloud={cloud} cloudState={cloudState} usedLive={usedLive} />
            )}
          </div>
        )}
      </div>
      <p className="caption">
        <strong>Figure 4 (interactive).</strong> One part driven through a single continuous
        pipeline run. The local decision and byte/PII counts are replayed from a real captured
        run; on escalation the reasoning is a live Qwen3-VL call. The network toggle is a live
        cut on the same run: drop it while an item is uncertain and the escalation parks in the
        outbox, restore it and the queue drains and the deferred verdict reconciles.
      </p>
    </section>
  )
}

function stageIdx(s: Stage): number {
  const order: Stage[] = ['idle', 'perceive', 'route', 'privacy', 'cloud', 'act', 'done']
  return order.indexOf(s)
}

function isStageShown(key: Stage, mr: ModeRun): boolean {
  // The privacy filter and cloud reasoning belong to the escalation branch. Whether the cloud
  // is actually reached is now a live, network-dependent thing, so these stages are shown for
  // any in-band part; the cloud step's detail reports what happened (called / queued / drained).
  if (key === 'privacy' || key === 'cloud') return mr.in_band
  return true
}

function stageDetail(key: Stage, mr: ModeRun, cloud: Cloud, cs: CloudState): string {
  switch (key) {
    case 'perceive':
      return `p = ${mr.p.toFixed(3)}`
    case 'route':
      return mr.in_band ? `in-band → ${mr.decision}` : `out-of-band → ${mr.decision}`
    case 'privacy':
      return `ROI · ${mr.bytes_to_cloud ?? 0} bytes · PII ${mr.pii_bytes ?? 0}`
    case 'cloud':
      if (cs === 'queued') return 'unreachable → queued to outbox'
      if (cs === 'draining') return 'reconnected → draining outbox…'
      if (cs === 'pending') return 'calling qwen3-vl-plus…'
      if (cs === 'reconciled')
        return cloud ? `${cloud.defect_present ? `defect: ${cloud.defect_type}` : 'clean'} ✓ reconciled` : '✓ reconciled'
      if (cs === 'resolved')
        return cloud ? (cloud.defect_present ? `defect: ${cloud.defect_type}` : 'clean') : '—'
      return '—'
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

function Verdict({ modeRun, cloud, cloudState, usedLive }: { modeRun: ModeRun; cloud: Cloud; cloudState: CloudState; usedLive: boolean }) {
  // Escalation is parked in the outbox: the line has acted conservatively but the cloud verdict
  // is still owed. This is the offline beat, live rather than pre-baked.
  if (modeRun.in_band && (cloudState === 'queued' || cloudState === 'draining')) {
    return (
      <div className="pg-verdict pg-offline">
        <div className="pg-verdict-head">
          <span className="pg-flag reject">{cloudState === 'draining' ? 'DRAINING' : 'DEFERRED'}</span>
          <span className="pg-src mono">{cloudState === 'draining' ? 'reconnecting' : 'outbox: queued'}</span>
        </div>
        <p>
          {cloudState === 'draining'
            ? 'Network restored. The outbox is draining and the deferred escalation is being reconciled with the cloud.'
            : 'Offline and uncertain: the line rejects conservatively and queues the escalation to the outbox. Flip the network back to FULL to drain it and back-fill the cloud verdict.'}
        </p>
      </div>
    )
  }
  const reconciled = cloudState === 'reconciled'
  const defect = cloud?.defect_present ?? modeRun.action === 'REJECT'
  return (
    <div className={`pg-verdict ${defect ? 'pg-defect' : 'pg-clean'}`}>
      <div className="pg-verdict-head">
        <span className={`pg-flag ${defect ? 'reject' : 'pass'}`}>{modeRun.action}</span>
        {modeRun.in_band && (
          <span className="pg-src mono">
            {reconciled ? 'reconciled from outbox' : usedLive ? 'live qwen3-vl-plus' : 'captured verdict'}
          </span>
        )}
      </div>
      {cloud && (
        <dl className="pg-verdict-rows">
          <div><dt className="mono">type</dt><dd>{cloud.defect_type}</dd></div>
          <div><dt className="mono">reasoning</dt><dd>{cloud.root_cause}</dd></div>
        </dl>
      )}
      {!modeRun.in_band && <p className="pg-local">Decided locally, with no cloud call and no bytes off-device.</p>}
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
