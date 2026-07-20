import BandWidget from './BandWidget'
import LiveDiagnose from './LiveDiagnose'
import Modalities from './Modalities'
import Playground from './Playground'
import {
  ABLATION_DELTAS,
  HEADLINE_ROWS,
  LINKS,
  LIVE_CLOUD,
  PIPELINE,
} from './data'
import { CostChart, RobustnessChart } from './Charts'
import './App.css'

export default function App() {
  return (
    <>
      <Nav />
      <Hero />
      <Idea />
      <HowItWorks />
      <Interactive />
      <Playground />
      <Results />
      <Modalities />
      <LiveDiagnose />
      <Ablation />
      <Offline />
      <Privacy />
      <GetStarted />
      <Footer />
    </>
  )
}

function Nav() {
  return (
    <nav className="nav">
      <div className="wrap nav-inner">
        <a href="#top" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          Tollgate
        </a>
        <div className="nav-links">
          <a href="#idea">Idea</a>
          <a href="#how">Pipeline</a>
          <a href="#band">Band</a>
          <a href="#playground">Playground</a>
          <a href="#results">Results</a>
          <a href="#modalities">Modalities</a>
          <a href="#try">Try it</a>
          <a href="#start">Get started</a>
          <a className="nav-cta" href={LINKS.repo}>GitHub</a>
        </div>
      </div>
    </nav>
  )
}

function Hero() {
  return (
    <header className="hero" id="top">
      <div className="wrap">
        <p className="eyebrow">Qwen Cloud Hackathon · EdgeAgent track</p>
        <h1 className="hero-title">
          Inspect at the edge.<br />
          Escalate by <span className="hl-toll">cost</span>, not confidence.
        </h1>
        <p className="hero-lead">
          Tollgate is an edge inspection agent that decides at the gate. It passes a frame
          locally when confident, pays the cloud toll only when it is worth it, and keeps
          working when the cloud is gone. Privacy, offline resilience, and cloud
          orchestration all fall out of one inequality.
        </p>
        <div className="hero-cta">
          <a className="btn btn-primary" href="#band">See the gate move</a>
          <a className="btn" href={LINKS.repo}>Read the code</a>
        </div>
        <div className="hero-stats">
          <Stat value="57%" label="of cloud cost" sub="hybrid vs. cloud-everything" />
          <Stat value="0.988" label="recall on bottle" sub="vs. 0.998 cloud-everything" />
          <Stat value="0" label="PII bytes egress" sub="measured at the boundary" />
          <Stat value="0.969" label="±0.015 across 6 cats" sub="the router generalizes" />
        </div>
      </div>
    </header>
  )
}

function Stat({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="stat">
      <div className="stat-value mono">{value}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}

function Idea() {
  return (
    <section id="idea">
      <div className="wrap">
        <p className="eyebrow">The idea</p>
        <h2 className="section-title">One decision generates three requirements</h2>
        <p className="section-lead">
          Most inspection systems bolt on privacy, offline support, and cloud orchestration as
          separate features. Tollgate derives all three from a single choice: route a frame to
          the cloud based on the <em>cost</em> of being wrong, not a fixed confidence cutoff.
        </p>

        <div className="formula">
          <code>
            band = [ T / C_FN , 1 − T / C_FP ]
            <span className="formula-where">where T = C_cloud + ε</span>
          </code>
        </div>

        <div className="idea-grid">
          <IdeaCard tone="toll" head="In-band (uncertain)"
            body="Escalate only the cropped ROI to Qwen-VL. Zero raw frames, zero PII." />
          <IdeaCard tone="pass" head="Out-of-band (confident)"
            body="Decide locally in milliseconds. No cloud call, no bandwidth, no wait." />
          <IdeaCard tone="reject" head="Offline + in-band"
            body="Conservatively reject locally and queue to the outbox. The line never stops." />
          <IdeaCard tone="accent" head="Reconnect"
            body="The outbox drains concurrently and the cloud verdict back-fills the record." />
        </div>
        <p className="idea-foot mono">One inequality. Three requirements.</p>
      </div>
    </section>
  )
}

function IdeaCard({ tone, head, body }: { tone: string; head: string; body: string }) {
  return (
    <div className={`idea-card tone-${tone}`}>
      <h3>{head}</h3>
      <p>{body}</p>
    </div>
  )
}

function HowItWorks() {
  return (
    <section id="how">
      <div className="wrap">
        <p className="eyebrow">How it works</p>
        <h2 className="section-title">A thin loop, all policy in one place</h2>
        <p className="section-lead">
          Every frame runs the same path. The router holds the entire decision, so perception,
          privacy, and actuation stay simple and swappable.
        </p>
        <ol className="pipeline">
          {PIPELINE.map((s, i) => (
            <li key={s.key} className="pipe-step">
              <span className="pipe-num mono">{String(i + 1).padStart(2, '0')}</span>
              <div>
                <h3 className="pipe-name">{s.name}</h3>
                <p className="pipe-detail">{s.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}

function Interactive() {
  return (
    <section id="band">
      <div className="wrap">
        <p className="eyebrow">Signature · interactive</p>
        <h2 className="section-title">The gate is derived from the costs</h2>
        <p className="section-lead">
          This is the whole thesis in one control. The escalate zone is computed live from the
          four costs below, exactly as <code>edge/router.py</code> does it. Move a cost and
          watch which parts pass locally, escalate, or reject.
        </p>
        <BandWidget />
      </div>
    </section>
  )
}

function Results() {
  return (
    <section id="results">
      <div className="wrap">
        <p className="eyebrow">Results · real MVTec data</p>
        <h2 className="section-title">Cloud accuracy at a fraction of the cost</h2>
        <p className="section-lead">
          Local <code>p</code> comes from the trained ONNX classifier over held-out eval images
          disjoint from training and calibration. Hybrid keeps recall within a whisker of
          cloud-everything while escalating only the uncertain band.
        </p>

        <div className="chart-row">
          <div className="chart-card">
            <h3 className="chart-title">Recall vs. cloud cost per 1k items (bottle)</h3>
            <CostChart />
          </div>
          <div className="chart-card">
            <h3 className="chart-title">Hybrid recall across 6 categories</h3>
            <RobustnessChart />
          </div>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Condition</th>
                <th>Cost-weighted recall</th>
                <th>p50 / p99 (ms)</th>
                <th>Bytes to cloud</th>
                <th>Cost / 1k</th>
                <th>PII</th>
              </tr>
            </thead>
            <tbody>
              {HEADLINE_ROWS.map((r) => (
                <tr key={r.condition} className={r.us ? 'row-us' : ''}>
                  <td>{r.condition}{r.us && <span className="tag-ours">ours</span>}</td>
                  <td className="mono">
                    {r.recall === null ? '—' : r.recall.toFixed(3)}
                    {r.recallLo !== undefined && r.recallLo !== r.recallHi && (
                      <span className="ci"> [{r.recallLo.toFixed(3)}–{r.recallHi!.toFixed(3)}]</span>
                    )}
                  </td>
                  <td className="mono">{r.p50 === null ? '—' : `${r.p50} / ${r.p99}`}</td>
                  <td className="mono">{r.bytes === null ? '—' : r.bytes}</td>
                  <td className="mono">{r.costPer1k === null ? '—' : `$${r.costPer1k.toFixed(0)}`}</td>
                  <td className="mono pii-zero">{r.pii}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="callout">
          <span className="callout-k mono">Live cloud, measured</span>
          <p>
            {LIVE_CLOUD.calls} real Qwen-VL calls on bottle ROIs: accuracy{' '}
            <b>{LIVE_CLOUD.accuracy.toFixed(1)}</b>, latency p50/p99{' '}
            <b className="mono">{LIVE_CLOUD.p50ms}/{LIVE_CLOUD.p99ms} ms</b>. That multi-second
            p99 is exactly why escalating only the uncertain band, not every item, matters.
          </p>
        </div>
      </div>
    </section>
  )
}

function Ablation() {
  return (
    <section id="ablation">
      <div className="wrap">
        <p className="eyebrow">Ablation · backbone swap</p>
        <h2 className="section-title">The router absorbs local-model variance</h2>
        <p className="section-lead">
          Swap the frozen feature backbone (hand-crafted features vs. an ImageNet MobileNetV2)
          with the router, privacy filter, and outbox untouched. The local model swings a lot;
          the hybrid barely moves.
        </p>
        <div className="delta-cards">
          <div className="delta-card">
            <div className="delta-head">Local-only Δ</div>
            <div className="delta-range mono">
              {ABLATION_DELTAS.localMin.toFixed(3)} … +{ABLATION_DELTAS.localMax.toFixed(3)}
            </div>
            <p>The backbone choice matters a lot when the local model decides alone.</p>
          </div>
          <div className="delta-card delta-hl">
            <div className="delta-head">Hybrid Δ</div>
            <div className="delta-range mono">
              {ABLATION_DELTAS.hybridMin.toFixed(3)} … +{ABLATION_DELTAS.hybridMax.toFixed(3)}
            </div>
            <p>
              Nearly flat and positive in most categories. The router escalates whatever the
              local model is unsure about, regardless of why, so orchestration carries the
              accuracy.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

function Offline() {
  const beats = [
    { k: 'full', label: 'Full link', body: 'Confident parts decide locally; the uncertain band escalates to Qwen-VL.' },
    { k: 'cut', label: 'Network cut', body: 'In-band items reject conservatively and queue to the outbox. The line never stalls.' },
    { k: 'back', label: 'Reconnect', body: 'The outbox drains in parallel and the late cloud verdict back-fills the log.' },
  ]
  return (
    <section id="offline">
      <div className="wrap">
        <p className="eyebrow">Graceful degradation</p>
        <h2 className="section-title">It keeps working when the cloud is gone</h2>
        <p className="section-lead">
          The network tier is measured, not assumed. When the probe fails, the same loop falls
          back to a safe local action and defers the diagnosis instead of blocking.
        </p>
        <div className="timeline">
          {beats.map((b, i) => (
            <div key={b.k} className={`tl-node tl-${b.k}`}>
              <div className="tl-dot" />
              <div className="tl-label mono">{i + 1} · {b.label}</div>
              <p className="tl-body">{b.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function Privacy() {
  return (
    <section id="privacy">
      <div className="wrap">
        <p className="eyebrow">Privacy accounting</p>
        <h2 className="section-title">Zero PII egress, and it is measurable</h2>
        <p className="section-lead">
          Every byte that crosses the device boundary is logged against its event, so the
          privacy claim is a number in the audit log, not a promise.
        </p>
        <div className="privacy-grid">
          <div className="priv-col priv-stays">
            <h3>Never leaves the device</h3>
            <ul>
              <li>Raw camera frames</li>
              <li>Skin-tone regions (blurred via HSV mask before encoding)</li>
              <li>Any recoverable biometric</li>
            </ul>
          </div>
          <div className="priv-col priv-crosses">
            <h3>Crosses only on escalation</h3>
            <ul>
              <li>The cropped ROI, or an abstracted embedding</li>
              <li>A non-PII context label (part category)</li>
              <li>Logged, byte-counted, PII count pinned at 0</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}

function GetStarted() {
  return (
    <section id="start">
      <div className="wrap">
        <p className="eyebrow">Get started</p>
        <h2 className="section-title">Run the real pipeline in three ways</h2>
        <p className="section-lead">
          <code>camera</code> and <code>data</code> run the same real pipeline and differ only
          in where frames come from. <code>demo</code> is a deterministic scripted walkthrough
          for a quick look, with no camera or cloud key required.
        </p>
        <pre className="code-block"><code>{`# 3.11 recommended
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q                         # 168 tests

# real device or a phone streaming over wifi (falls back to data/ if no camera)
python -m edge.app camera --camera http://<phone-ip>:8080/video

# replay a folder through the real pipeline
python -m edge.app data --category bottle --limit 20

# scripted walkthrough, no hardware needed
python -m edge.app demo`}</code></pre>
        <p className="reproduce">
          Reproduce every number above with{' '}
          <code>python -m eval.run_real_eval</code> and{' '}
          <code>python -m eval.run_multi_category</code>.
        </p>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer">
      <div className="wrap footer-inner">
        <div>
          <div className="brand"><span className="brand-mark" aria-hidden="true" />Tollgate</div>
          <p className="footer-tag">Edge inspection that routes by cost, not confidence.</p>
        </div>
        <div className="footer-links">
          <a href={LINKS.repo}>GitHub repository</a>
          <a href={LINKS.video}>Demo video</a>
          <span className="footer-meta">Qwen-VL · Alibaba Cloud · MVTec AD</span>
        </div>
      </div>
    </footer>
  )
}
