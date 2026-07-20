import { LOCO, LOCO_AGG } from './data'
import './Modalities.css'

// MVTec 3D-AD, measured with real qwen3-vl-plus on each escalated cloud's paired RGB image.
// From eval/results_3d.md.
const THREED = {
  categories: 10,
  localRecall: 0.56,
  escalation: 0.47,
  hybridRecall: 0.68 as number | null,
}

export default function Modalities() {
  return (
    <section id="modalities" className="col-wide">
      <div className="col-inner">
        <h2 className="sec-head"><span className="sec-num">5</span> One router, three kinds of defect</h2>
        <p>
          The cost decision only sees a calibrated <code>p</code>, so it does not care where the
          uncertainty comes from. The same router handles surface defects (MVTec AD), logical
          constraint violations (MVTec LOCO), and 3D point clouds (MVTec 3D-AD) with zero change
          to the orchestration, privacy filter, or outbox.
        </p>

        <Loco />
        <ThreeD />
      </div>
    </section>
  )
}

function Loco() {
  return (
    <div className="mod-block">
      <h3 className="mod-title">Logical anomalies · MVTec LOCO</h3>
      <p className="mod-lead">
        Logical anomalies (a wrong count, a missing or misplaced part) leave every surface
        looking normal, so a local texture model is near-blind to them. Trained on good images
        only (an unsupervised anomaly score), the local model catches barely half. The router
        escalates the uncertain cases and <strong>real qwen3-vl-plus verdicts</strong> recover
        most of the rest.
      </p>

      <div className="mod-agg">
        <AggCard kind="Logical" a={LOCO_AGG.logical} highlight />
        <AggCard kind="Structural" a={LOCO_AGG.structural} />
      </div>

      <div className="table-scroll">
        <table className="data-table mod-table">
          <thead>
            <tr>
              <th>Category</th><th>Kind</th><th>n</th><th>Escalation</th>
              <th>Local recall</th><th>Hybrid recall</th>
            </tr>
          </thead>
          <tbody>
            {LOCO.map((r) => (
              <tr key={`${r.category}-${r.kind}`} className={r.kind === 'logical' ? 'row-logical' : ''}>
                <td>{r.category}</td>
                <td className="mono">{r.kind}</td>
                <td className="mono">{r.n}</td>
                <td className="mono">{(r.escalation * 100).toFixed(0)}%</td>
                <td className="mono">{r.localRecall.toFixed(2)}</td>
                <td className="mono lift">{r.hybridRecall.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mod-note">
        The finding is honestly category-dependent: where logical anomalies are geometrically
        subtle (pushpins: local 0.49 → hybrid 0.98), the router earns its keep; where the
        cloud also struggles (splicing_connectors), hybrid gains less. Hybrid recall is
        measured with live Qwen-VL calls, not modeled.
      </p>
    </div>
  )
}

function AggCard({ kind, a, highlight }: { kind: string; a: { escalation: number; localRecall: number; hybridRecall: number }; highlight?: boolean }) {
  const lift = a.hybridRecall - a.localRecall
  return (
    <div className={`agg-card ${highlight ? 'agg-hl' : ''}`}>
      <div className="agg-kind mono">{kind}</div>
      <div className="agg-flow">
        <span className="agg-local mono">{a.localRecall.toFixed(2)}</span>
        <span className="agg-arrow">→</span>
        <span className="agg-hybrid mono">{a.hybridRecall.toFixed(2)}</span>
      </div>
      <div className="agg-lift mono">+{lift.toFixed(2)} recall</div>
      <div className="agg-sub">escalation {(a.escalation * 100).toFixed(0)}% · measured hybrid</div>
    </div>
  )
}

function ThreeD() {
  return (
    <div className="mod-block">
      <h3 className="mod-title">3D point clouds · MVTec 3D-AD</h3>
      <p className="mod-lead">
        Proof the routing is genuinely modality-agnostic: swap the image backbone for a frozen
        PointNet encoder over organized point clouds and the <em>same</em> cost inequality
        bands a calibrated <code>p</code>. Preprocessing doubles as privacy: the cloud is
        centered to local coordinates, stripping the absolute sensor-frame position so only
        local surface shape ever leaves the device.
      </p>
      <div className="mod-3d-stats">
        <div className="stat3d"><span className="stat3d-v mono">{THREED.categories}</span><span className="stat3d-l">categories</span></div>
        <div className="stat3d">
          <span className="stat3d-v mono">
            {THREED.localRecall.toFixed(2)}
            {THREED.hybridRecall != null && <span className="stat3d-arrow"> → {THREED.hybridRecall.toFixed(2)}</span>}
          </span>
          <span className="stat3d-l">local → hybrid recall</span>
        </div>
        <div className="stat3d"><span className="stat3d-v mono">{(THREED.escalation * 100).toFixed(0)}%</span><span className="stat3d-l">escalation</span></div>
        <div className="stat3d"><span className="stat3d-v mono">0</span><span className="stat3d-l">orchestration changes</span></div>
      </div>
      <p className="mod-note">
        Same weak local model, same router, same outbox — only the feature extractor differs.
        With real Qwen-VL verdicts on the escalated clouds' paired RGB, hybrid recall rises
        from {THREED.localRecall.toFixed(2)} to {THREED.hybridRecall?.toFixed(2)}. The lift is
        modest because the reasoner sees colour, not the depth channel, an honest limitation,
        but the cost-routing decision transfers to a new modality with zero orchestration
        change, which is the "one inequality" claim made concrete.
      </p>
    </div>
  )
}
