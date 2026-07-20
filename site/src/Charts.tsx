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
const C_BAR_LOCAL = cssVar('--line-2', '#cfd2d9')

// Recall vs. cloud cost per 1k. The point of the chart: hybrid sits near cloud accuracy
// but far left on cost. Bottle numbers from results_table_real.md.
const COST_POINTS = [
  { name: 'Cloud-everything', cost: 2000, recall: 0.998, fill: C_CLOUD },
  { name: 'Hybrid (ours)', cost: 1148, recall: 0.988, fill: C_HYBRID },
  { name: 'Local-only', cost: 0, recall: 0.908, fill: C_LOCAL },
]

export function CostChart() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 16, right: 20, bottom: 34, left: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
        <XAxis
          type="number"
          dataKey="cost"
          name="cost"
          domain={[-100, 2100]}
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
          formatter={(v, n) => (n === 'cost' ? `$${v}` : Number(v).toFixed(3))}
        />
        <Scatter data={COST_POINTS}>
          {COST_POINTS.map((p) => (
            <Cell key={p.name} fill={p.fill} />
          ))}
          <LabelList dataKey="name" position="top" style={{ fill: AXIS, fontSize: 10 }} />
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
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={ROBUST} margin={{ top: 16, right: 12, bottom: 8, left: 4 }} barCategoryGap="24%">
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
          domain={[80, 100]}
          tick={{ fill: AXIS, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          stroke={GRID}
          width={36}
        />
        <Tooltip cursor={{ fill: 'rgba(128,128,128,0.08)' }} contentStyle={tooltipStyle}
          formatter={(v) => `${v}%`} />
        <ReferenceLine y={96.9} stroke={C_HYBRID} strokeDasharray="4 4"
          label={{ value: 'mean 0.969', fill: C_HYBRID, fontSize: 10, position: 'insideTopRight' }} />
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
