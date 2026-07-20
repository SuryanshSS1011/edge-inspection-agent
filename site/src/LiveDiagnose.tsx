import { useState } from 'react'
import { DIAGNOSE_URL, SAMPLES, type Sample } from './data'
import './LiveDiagnose.css'

type Verdict = {
  defect_present: boolean
  defect_type: string
  confidence: number
  root_cause: string
  recommended_action: string
}

type State =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ok'; verdict: Verdict; ms: number }
  | { status: 'error'; message: string }

// Fetch a bundled sample image and return its base64 PNG payload (no data: prefix), which
// is exactly what POST /diagnose expects for roi_png_b64.
async function toBase64(url: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`sample image not found (${res.status})`)
  const blob = await res.blob()
  const dataUrl: string = await new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onloadend = () => resolve(r.result as string)
    r.onerror = () => reject(new Error('could not read sample image'))
    r.readAsDataURL(blob)
  })
  return dataUrl.split(',')[1] ?? ''
}

export default function LiveDiagnose() {
  const [selected, setSelected] = useState<Sample | null>(null)
  const [state, setState] = useState<State>({ status: 'idle' })

  const enabled = DIAGNOSE_URL.length > 0

  async function run(sample: Sample) {
    setSelected(sample)
    if (!enabled) {
      setState({ status: 'error', message: 'Live endpoint not configured for this build.' })
      return
    }
    setState({ status: 'loading' })
    const t0 = performance.now()
    try {
      const roi = await toBase64(`${import.meta.env.BASE_URL}samples/${sample.file}`)
      const res = await fetch(`${DIAGNOSE_URL}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi_png_b64: roi, context: { category: sample.category } }),
      })
      if (!res.ok) throw new Error(`server returned ${res.status}`)
      const verdict = (await res.json()) as Verdict
      setState({ status: 'ok', verdict, ms: Math.round(performance.now() - t0) })
    } catch (err) {
      const message =
        err instanceof TypeError
          ? 'Could not reach the reasoning server. It may be offline, or blocked by mixed-content if it is not HTTPS.'
          : err instanceof Error
            ? err.message
            : 'Unknown error'
      setState({ status: 'error', message })
    }
  }

  return (
    <section id="try" className="col-wide">
      <div className="col-inner">
        <h3 className="sub-head">Live cloud: send a real ROI to the deployed reasoner</h3>
        <p>
          These calls hit the actual Qwen-VL reasoning server running on Alibaba Cloud, the
          same <code>POST /diagnose</code> the edge escalates to. Pick a part and read the
          verdict it returns.
        </p>

        {!enabled && (
          <div className="ld-note">
            The live endpoint is not wired into this build. Set{' '}
            <code>VITE_DIAGNOSE_URL</code> to the deployed HTTPS server to enable it.
          </div>
        )}

        <div className="ld-grid">
          <div className="ld-samples">
            {SAMPLES.map((s) => (
              <button
                key={s.file}
                className={`ld-sample ${selected?.file === s.file ? 'is-active' : ''}`}
                onClick={() => run(s)}
                disabled={state.status === 'loading'}
              >
                <img
                  src={`${import.meta.env.BASE_URL}samples/${s.file}`}
                  alt={s.label}
                  onError={(e) => ((e.target as HTMLImageElement).style.visibility = 'hidden')}
                />
                <span className="ld-sample-label">{s.label}</span>
                <span className="ld-sample-hint mono">{s.category}</span>
              </button>
            ))}
          </div>

          <div className="ld-result">
            {state.status === 'idle' && (
              <p className="ld-placeholder">Pick a sample to run a live diagnosis.</p>
            )}
            {state.status === 'loading' && (
              <p className="ld-placeholder ld-loading">Calling Qwen-VL…</p>
            )}
            {state.status === 'error' && (
              <div className="ld-error">
                <span className="mono">diagnosis unavailable</span>
                <p>{state.message}</p>
              </div>
            )}
            {state.status === 'ok' && <VerdictCard v={state.verdict} ms={state.ms} />}
          </div>
        </div>
        <p className="ld-credit mono">Sample images: MVTec AD (CC BY-NC-SA, shown for demo).</p>
      </div>
    </section>
  )
}

function VerdictCard({ v, ms }: { v: Verdict; ms: number }) {
  return (
    <div className={`verdict ${v.defect_present ? 'v-defect' : 'v-clean'}`}>
      <div className="verdict-head">
        <span className="verdict-flag">
          {v.defect_present ? 'DEFECT' : 'CLEAN'}
        </span>
        <span className="verdict-ms mono">{ms} ms round-trip</span>
      </div>
      <dl className="verdict-rows">
        <Row k="type" val={v.defect_type} />
        <Row k="confidence" val={v.confidence.toFixed(2)} />
        <Row k="root cause" val={v.root_cause} />
        <Row k="action" val={v.recommended_action} />
      </dl>
    </div>
  )
}

function Row({ k, val }: { k: string; val: string }) {
  return (
    <div className="verdict-row">
      <dt className="mono">{k}</dt>
      <dd>{val}</dd>
    </div>
  )
}
