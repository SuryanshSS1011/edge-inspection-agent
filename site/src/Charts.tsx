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

const AXIS = '#67748a'
const GRID = '#232c39'

// Recall vs. cloud cost per 1k. The point of the chart: hybrid sits near cloud accuracy
// but far left on cost. Bottle numbers from results_table_real.md.
const COST_POINTS = [
  { name: 'Cloud-everything', cost: 2000, recall: 0.998, fill: '#6cb6ff' },
  { name: 'Hybrid (ours)', cost: 1148, recall: 0.988, fill: '#e3a008' },
  { name: 'Local-only', cost: 0, recall: 0.908, fill: '#4cc38a' },
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
          <LabelList dataKey="name" position="top" style={{ fill: '#9aa7b8', fontSize: 10 }} />
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
        <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={tooltipStyle}
          formatter={(v) => `${v}%`} />
        <ReferenceLine y={96.9} stroke="#e3a008" strokeDasharray="4 4"
          label={{ value: 'mean 0.969', fill: '#e3a008', fontSize: 10, position: 'insideTopRight' }} />
        <Bar dataKey="local" name="local-only" fill="#2f3a4a" radius={[3, 3, 0, 0]} />
        <Bar dataKey="hybrid" name="hybrid" fill="#4cc38a" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

const tooltipStyle = {
  background: '#12171f',
  border: '1px solid #2f3a4a',
  borderRadius: 8,
  color: '#e6edf3',
  fontSize: 12,
  fontFamily: 'IBM Plex Mono',
}
