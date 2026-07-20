import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { CATEGORIES } from './data'

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
// A visible slate-blue for the local-only bar. The previous --line-2 gray sat almost on
// top of the paper background, so both the column and its tooltip swatch disappeared.
const C_BAR_LOCAL = cssVar('--link-dim', '#6f8fdd')

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

const ROBUST = CATEGORIES.map((c) => ({
  category: c.category,
  hybrid: Number((c.hybrid * 100).toFixed(1)),
  local: Number((c.local * 100).toFixed(1)),
}))

export function RobustnessChart() {
  return (
    <ResponsiveContainer width="100%" height={268}>
      <BarChart data={ROBUST} margin={{ top: 24, right: 12, bottom: 8, left: 4 }} barCategoryGap="24%">
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="category"
          tick={{ fill: AXIS, fontSize: 10, fontFamily: 'IBM Plex Mono' }}
          stroke={GRID}
          interval={0}
          angle={-18}
          textAnchor="end"
          height={44}
        />
        <YAxis
          domain={[80, 101]}
          ticks={[80, 85, 90, 95, 100]}
          tick={{ fill: AXIS, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          stroke={GRID}
          width={36}
        />
        <Tooltip cursor={{ fill: 'rgba(128,128,128,0.08)' }} contentStyle={tooltipStyle}
          formatter={(v) => `${v}%`} />
        <ReferenceLine y={99.7} stroke={C_HYBRID} strokeDasharray="4 4"
          label={{ value: 'mean 0.997', fill: C_HYBRID, fontSize: 10, position: 'insideTopRight', dy: -20, dx: 8 }} />
        <Bar dataKey="local" name="local-only" fill={C_BAR_LOCAL} radius={[3, 3, 0, 0]} />
        <Bar dataKey="hybrid" name="hybrid" fill={C_HYBRID} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

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
