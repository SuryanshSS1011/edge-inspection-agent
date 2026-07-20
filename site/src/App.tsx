import BandWidget from './BandWidget'
import LiveDiagnose from './LiveDiagnose'
import Modalities from './Modalities'
import Playground from './Playground'
import { CostChart, RobustnessChart } from './Charts'
import {
  ABLATION_DELTAS,
  HEADLINE_ROWS,
  LINKS,
  LIVE_CLOUD,
  PIPELINE,
} from './data'
import './App.css'

export default function App() {
  return (
    <main className="paper">
      <TitleBlock />
      <Abstract />
      <Idea />
      <Method />
      <BandFigure />
      <PlaygroundFigure />
      <Results />
      <Modalities />
      <Ablation />
      <Robustness />
      <LiveDiagnoseFigure />
      <Reproduce />
      <References />
    </main>
  )
}

function TitleBlock() {
  return (
    <header className="title-block col" id="top">
      <p className="venue">Qwen Cloud Hackathon · EdgeAgent track</p>
      <h1 className="title">
        Tollgate: cost-based cloud escalation for edge visual inspection
      </h1>
      <p className="byline">
        A hybrid edge–cloud inspection agent that routes frames to a vision–language model by
        the <em>cost</em> of being wrong, not a fixed confidence threshold.
      </p>
      <nav className="links">
        <a href={LINKS.repo}>Code</a>
        <span>·</span>
        <a href="#try">Live demo</a>
        <span>·</span>
        <a href={LINKS.video}>Video</a>
        <span>·</span>
        <a href="https://www.mvtec.com/company/research/datasets">Data (MVTec)</a>
      </nav>
    </header>
  )
}

function Abstract() {
  return (
    <section className="col abstract">
      <h2 className="ab-head">Abstract</h2>
      <p>
        Edge inspection systems usually bolt on privacy, offline support, and cloud
        orchestration as separate features. We derive all three from a single decision: route
        a frame to the cloud when the expected <em>cost</em> of deciding locally exceeds the
        cost of a cloud call, rather than at a fixed confidence cutoff. The resulting
        escalation band, <span className="mono">[T/C_FN, 1−T/C_FP]</span> with{' '}
        <span className="mono">T = C_cloud + ε</span>, is derived from operating costs and
        never hand-tuned. A confident part is decided locally in milliseconds; an uncertain
        one escalates only a cropped ROI to Qwen3-VL, so raw frames and biometrics never leave
        the device. When the network is down the line keeps running: uncertain items are
        deferred to an outbox and reconciled on reconnect. On real MVTec data the hybrid
        matches cloud-only accuracy at 57% of its cost, and the router recovers most of what a
        deliberately weak local model misses across structural defects, logical constraint
        violations (MVTec LOCO), and 3D point clouds (MVTec 3D-AD), with zero change to the
        orchestration.
      </p>
    </section>
  )
}

function Idea() {
  return (
    <section className="col" id="idea">
      <h2 className="sec-head"><span className="sec-num">1</span> One decision, three requirements</h2>
      <p>
        Most inspection systems treat privacy, offline operation, and cloud orchestration as
        independent problems. Tollgate treats them as consequences of one choice: escalate a
        frame only when a cloud call is cost-justified. The escalation band is derived from the
        operating costs, so it adapts as they change and is never hand-tuned:
      </p>
      <div className="equation">
        <span className="mono">band = [ T / C_FN , 1 − T / C_FP ]</span>
        <span className="eq-where mono">where T = C_cloud + ε</span>
      </div>
      <p>Three behaviours follow directly:</p>
      <ul className="claims">
        <li><strong>In-band (uncertain).</strong> Escalate only the cropped ROI to Qwen3-VL. Zero raw frames, zero PII.</li>
        <li><strong>Out-of-band (confident).</strong> Decide locally in milliseconds. No cloud call.</li>
        <li><strong>Offline and in-band.</strong> Reject conservatively and queue to the outbox; the line never stalls.</li>
        <li><strong>Reconnect.</strong> The outbox drains concurrently and the cloud verdict back-fills the record.</li>
      </ul>
    </section>
  )
}

function Method() {
  return (
    <section className="col" id="how">
      <h2 className="sec-head"><span className="sec-num">2</span> Method</h2>
      <p>
        Every frame follows one path; all policy lives in the router, so perception, privacy,
        and actuation stay simple and swappable behind a common interface.
      </p>
      <ol className="pipeline">
        {PIPELINE.map((s, i) => (
          <li key={s.key} className="pipe-step">
            <span className="pipe-num mono">{i + 1}</span>
            <div>
              <span className="pipe-name">{s.name}.</span>{' '}
              <span className="pipe-detail">{s.detail}</span>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

function BandFigure() {
  return (
    <section className="col-wide figure" id="band">
      <BandWidget />
      <p className="caption">
        <strong>Figure 1 (interactive).</strong> The escalation band derived from the four
        operating costs, computed live exactly as <code>edge/router.py</code> does. Drag a cost
        to reshape the band and watch which parts pass locally, escalate, or reject. Raising the
        price of a missed defect widens the band; raising the cloud toll narrows it.
      </p>
    </section>
  )
}

function PlaygroundFigure() {
  return <Playground />
}

function Results() {
  return (
    <section className="col-wide" id="results">
      <div className="col-inner">
        <h2 className="sec-head"><span className="sec-num">4</span> Results on real MVTec data</h2>
        <p>
          Local <code>p</code> comes from the trained ONNX classifier over held-out eval images
          disjoint from training and calibration. The hybrid stays within a whisker of
          cloud-only accuracy while escalating only the uncertain band.
        </p>
      </div>

      <div className="fig-row">
        <figure className="fig-cell">
          <CostChart />
          <figcaption className="caption">
            <strong>Figure 2.</strong> Cost-weighted recall versus cloud cost per 1k items
            (bottle). Hybrid sits near cloud-only accuracy at a fraction of the cost.
          </figcaption>
        </figure>
        <figure className="fig-cell">
          <RobustnessChart />
          <figcaption className="caption">
            <strong>Figure 3.</strong> Hybrid recall across six categories, mean 0.969 ± 0.015.
            Local-only in grey; hybrid in colour.
          </figcaption>
        </figure>
      </div>

      <div className="col-inner">
        <div className="table-wrap">
          <table className="paper-table">
            <caption><strong>Table 1.</strong> Accuracy, latency, and cost by condition (bottle).</caption>
            <thead>
              <tr>
                <th>Condition</th><th>Cost-weighted recall</th><th>p50 / p99 (ms)</th>
                <th>Bytes to cloud</th><th>Cost / 1k</th><th>PII</th>
              </tr>
            </thead>
            <tbody>
              {HEADLINE_ROWS.map((r) => (
                <tr key={r.condition} className={r.us ? 'row-us' : ''}>
                  <td>{r.condition}{r.us && <span className="tag-ours">ours</span>}</td>
                  <td className="mono">
                    {r.recall === null ? '—' : r.recall.toFixed(3)}
                    {r.recallLo !== undefined && r.recallLo !== r.recallHi && (
                      <span className="ci"> [{r.recallLo.toFixed(3)}, {r.recallHi!.toFixed(3)}]</span>
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
        <p className="footnote">
          Live cloud, measured on {LIVE_CLOUD.calls} real Qwen3-VL calls on bottle ROIs:
          accuracy {LIVE_CLOUD.accuracy.toFixed(1)}, latency p50/p99{' '}
          <span className="mono">{LIVE_CLOUD.p50ms}/{LIVE_CLOUD.p99ms} ms</span>. The multi-second
          p99 is precisely why escalating only the uncertain band, not every item, matters.
        </p>
      </div>
    </section>
  )
}

function Ablation() {
  return (
    <section className="col" id="ablation">
      <h2 className="sec-head"><span className="sec-num">6</span> Ablation: the router absorbs backbone variance</h2>
      <p>
        We swap the frozen feature backbone across a weak (hand-crafted), medium (ImageNet
        MobileNetV2), and SOTA self-supervised (DINOv2) extractor, holding the router, privacy
        filter, and outbox fixed. Local-only recall swings widely; hybrid recall stays tight.
      </p>
      <div className="two-stat">
        <div className="stat-block">
          <div className="stat-num mono">≤ {ABLATION_DELTAS.localSpread.toFixed(3)}</div>
          <div className="stat-cap">local-only recall spread across backbones</div>
        </div>
        <div className="stat-block hl">
          <div className="stat-num mono">≤ {ABLATION_DELTAS.hybridSpread.toFixed(3)}</div>
          <div className="stat-cap">hybrid recall spread — the router more than halves it</div>
        </div>
      </div>
      <p className="footnote">
        The router escalates whatever the local model is unsure about regardless of why, so the
        orchestration, not the choice of backbone, carries the accuracy.
      </p>
    </section>
  )
}

function Robustness() {
  return (
    <section className="col" id="robustness">
      <h2 className="sec-head"><span className="sec-num">7</span> Graceful degradation and privacy</h2>
      <div className="two-col">
        <div>
          <h3 className="sub-head">Offline resilience</h3>
          <p>
            The network tier is measured, not assumed. When the probe fails, the same loop
            falls back to a safe local action and defers the diagnosis to the outbox rather than
            blocking. On reconnect the outbox drains in parallel and the late cloud verdict
            back-fills the log, so the line never stalls.
          </p>
        </div>
        <div>
          <h3 className="sub-head">Measurable privacy</h3>
          <p>
            Every byte crossing the device boundary is logged against its event. Raw frames,
            skin-tone regions (blurred by an HSV mask before encoding), and recoverable
            biometrics never leave; only the cropped ROI or an abstracted embedding crosses, and
            the PII byte count is pinned at zero, a number in the audit log rather than a promise.
          </p>
        </div>
      </div>
    </section>
  )
}

function LiveDiagnoseFigure() {
  return <LiveDiagnose />
}

function Reproduce() {
  return (
    <section className="col" id="start">
      <h2 className="sec-head"><span className="sec-num">8</span> Reproducibility</h2>
      <p>
        <code>camera</code> and <code>data</code> run the same real pipeline and differ only in
        the frame source; <code>demo</code> is a deterministic scripted walkthrough.
      </p>
      <pre className="code-block"><code>{`python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q                              # 172 tests

python -m edge.app camera --camera http://<phone-ip>:8080/video
python -m edge.app data   --category bottle --limit 20
python -m edge.app demo

# reproduce the tables
python -m eval.run_real_eval
python -m eval.run_multi_category
python -m eval.run_loco   --data <loco>   --real-cloud
python -m eval.run_3d     --data <3d-ad>  --real-cloud`}</code></pre>
    </section>
  )
}

function References() {
  return (
    <footer className="col references" id="refs">
      <h2 className="sec-head">References &amp; artifacts</h2>
      <ol className="ref-list">
        <li>Bergmann et al. <em>MVTec AD</em> and <em>MVTec LOCO AD</em>, and <em>MVTec 3D-AD</em>. CC BY-NC-SA.</li>
        <li>Oquab et al. <em>DINOv2: Learning Robust Visual Features without Supervision.</em> 2023.</li>
        <li>Qi et al. <em>PointNet: Deep Learning on Point Sets.</em> CVPR 2017.</li>
        <li>Reasoning tier: Qwen3-VL via Alibaba Cloud DashScope, served from a Docker container on Alibaba SAS.</li>
      </ol>
      <p className="colophon">
        <a href={LINKS.repo}>Source code</a> · Qwen3-VL · Alibaba Cloud · Evaluated on Penn State ROAR.
      </p>
    </footer>
  )
}
