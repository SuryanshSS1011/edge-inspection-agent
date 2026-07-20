import {
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AGGREGATE, CATEGORIES } from './data'

// Read palette from CSS custom properties so charts match the paper theme (light/dark).
function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}
const AXIS = cssVar('--muted', '#5b6270')
const GRID = cssVar('--line', '#e0e2e7')
const C_HYBRID = cssVar('--toll', '#b8791b')
const C_CLOUD = cssVar('--link', '#2f5fd0')
const C_LOCAL = cssVar('--pass', '#2f8f5b')

// Recall vs. cloud cost per 1k. The point of the chart: hybrid sits near cloud accuracy
// but far left on cost. Bottle numbers from results_table_real.md.
const COST_POINTS = [
  { name: 'Cloud-everything', cost: 2000, recall: 0.998, fill: C_CLOUD },
  { name: 'Hybrid (ours)', cost: 1148, recall: 0.988, fill: C_HYBRID },
  { name: 'Local-only', cost: 0, recall: 0.908, fill: C_LOCAL },
]

// Anchor each point's label so it never overruns the plot edges: the rightmost point
// (Cloud-everything) is end-anchored, the leftmost (Local-only) start-anchored, and every
// label is lifted clear of its dot so none crowd the axis numbers. Each label is
// double-stacked: the method name on top, its cloud cost on a second line beneath.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CostLabel(props: any) {
  const x = Number(props.x ?? 0)
  const y = Number(props.y ?? 0)
  const index: number = props.index ?? 0
  const point = COST_POINTS[index]
  const anchor = index === 0 ? 'end' : index === COST_POINTS.length - 1 ? 'start' : 'middle'
  // Nudge the right-most (Cloud-everything) label rightward a touch; keep the left-most padded.
  const dx = anchor === 'end' ? 6 : anchor === 'start' ? 8 : 0
  return (
    <text
      x={x + dx}
      y={y - 22}
      textAnchor={anchor}
      style={{ fill: AXIS, fontFamily: 'Inter, system-ui, sans-serif' }}
    >
      <tspan x={x + dx} style={{ fontSize: 10, fontWeight: 600 }}>
        {point?.name}
      </tspan>
      <tspan
        x={x + dx}
        dy={12}
        style={{ fontSize: 9, fontFamily: 'IBM Plex Mono', fill: cssVar('--muted', '#5b6270') }}
      >
        {`$${point?.cost}/1k`}
      </tspan>
    </text>
  )
}

export function CostChart() {
  return (
    <ResponsiveContainer width="100%" height={284}>
      <ScatterChart margin={{ top: 40, right: 28, bottom: 34, left: 8 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis
          type="number"
          dataKey="cost"
          name="cost"
          domain={[0, 2100]}
          ticks={[0, 500, 1000, 1500, 2000]}
          tick={{ fill: AXIS, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          tickFormatter={(v) => `$${v}`}
          label={{ value: 'cloud cost / 1k', position: 'insideBottom', offset: -18, fill: AXIS, fontSize: 11 }}
          stroke={GRID}
        />
        <YAxis
          type="number"
          dataKey="recall"
          name="recall"
          domain={[0.88, 1.0]}
          tick={{ fill: AXIS, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          stroke={GRID}
          width={44}
        />
        <Tooltip
          cursor={{ strokeDasharray: '3 3', stroke: AXIS }}
          contentStyle={tooltipStyle}
          itemStyle={tooltipItemStyle}
          labelStyle={tooltipItemStyle}
          formatter={(v, n) => (n === 'cost' ? `$${v}` : Number(v).toFixed(3))}
        />
        <Scatter data={COST_POINTS}>
          {COST_POINTS.map((p) => (
            <Cell key={p.name} fill={p.fill} />
          ))}
          <LabelList dataKey="name" content={CostLabel} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// Figures 3a/3b show the router's core behaviour directly: escalation rate vs. local recall,
// one point per category. The downward trend (AD r = -0.80, AD 2 r = -0.94) is the thesis: the
// router escalates in proportion to how weak the local model is, spending cloud budget only
// where it is needed. Split by dataset so each is legible: AD's easy parts sit high-right
// (confident, quiet), AD 2's hard parts sit low-left (weak local, heavy escalation).
type RoutePoint = { category: string; local: number; escalation: number }

const routePoints = (dataset: 'AD' | 'AD2'): RoutePoint[] =>
  CATEGORIES.filter((c) => c.dataset === dataset).map((c) => ({
    category: c.category,
    local: Number((c.local * 100).toFixed(0)),
    escalation: Number((c.escalation * 100).toFixed(0)),
  }))

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function PointLabel(props: any) {
  const x = Number(props.x ?? 0)
  const y = Number(props.y ?? 0)
  return (
    <text
      x={x + 9}
      y={y + 3}
      textAnchor="start"
      style={{ fill: AXIS, fontSize: 10, fontFamily: 'IBM Plex Mono' }}
    >
      {props.value}
    </text>
  )
}

function RoutingScatter({ dataset, fill }: { dataset: 'AD' | 'AD2'; fill: string }) {
  const data = routePoints(dataset)
  return (
    <ResponsiveContainer width="100%" height={380}>
      <ScatterChart margin={{ top: 20, right: 64, bottom: 46, left: 12 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis
          type="number"
          dataKey="local"
          name="local recall"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tick={{ fill: AXIS, fontSize: 12, fontFamily: 'IBM Plex Mono' }}
          tickFormatter={(v) => `${v}%`}
          label={{ value: 'local-only recall', position: 'insideBottom', offset: -24, fill: AXIS, fontSize: 12 }}
          stroke={GRID}
        />
        <YAxis
          type="number"
          dataKey="escalation"
          name="escalation"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tick={{ fill: AXIS, fontSize: 12, fontFamily: 'IBM Plex Mono' }}
          tickFormatter={(v) => `${v}%`}
          label={{ value: 'escalation rate', angle: -90, position: 'insideLeft', offset: 18, fill: AXIS, fontSize: 12 }}
          stroke={GRID}
          width={52}
        />
        <Tooltip
          cursor={{ strokeDasharray: '3 3', stroke: AXIS }}
          contentStyle={tooltipStyle}
          itemStyle={tooltipItemStyle}
          labelStyle={tooltipItemStyle}
          formatter={(v, n) => [`${v}%`, n]}
        />
        <Scatter data={data} fill={fill}>
          <LabelList dataKey="category" content={PointLabel} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function RoutingChartAD() {
  return <RoutingScatter dataset="AD" fill={C_CLOUD} />
}
export function RoutingChartAD2() {
  return <RoutingScatter dataset="AD2" fill={C_HYBRID} />
}
export const ROUTING_R = { ad: AGGREGATE.ad.r, ad2: AGGREGATE.ad2.r }

const tooltipStyle = {
  background: cssVar('--panel', '#f3f4f6'),
  border: `1px solid ${cssVar('--line-2', '#cfd2d9')}`,
  borderRadius: 4,
  color: cssVar('--ink', '#16181d'),
  fontSize: 12,
  fontFamily: 'IBM Plex Mono',
}

// Force tooltip rows to the readable ink color. Recharts otherwise tints each row with the
// scatter point's fill (green/gold/blue), which is near-invisible on the panel background.
const tooltipItemStyle = {
  color: cssVar('--ink', '#16181d'),
  fontFamily: 'IBM Plex Mono',
  fontSize: 12,
}
