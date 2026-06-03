import { useEffect, useState } from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { api } from '../api/client.js'
import { Card, SectionTitle, Loading, EmptyState } from '../components/UI.jsx'
import { CHART_COLORS } from '../config.js'

const ts = {
  contentStyle: { background: '#0f1729', border: '1px solid #1f2a44', borderRadius: 12 },
  labelStyle: { color: '#94a3b8' },
}

export default function WellDetail() {
  const [wells, setWells] = useState([])
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.dashboardSummary().then((d) => {
      if (d?.available && d.wells?.length) {
        setWells(d.wells)
        setSelected(d.wells[0].well_id)
      } else {
        setLoading(false)
      }
    })
  }, [])

  useEffect(() => {
    if (!selected) return
    setLoading(true)
    api.well(selected).then(setDetail).finally(() => setLoading(false))
  }, [selected])

  if (!wells.length && !loading)
    return <EmptyState message="No wells found. Run the batch prediction pipeline." />

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Well Detail</h1>
          <p className="text-sm text-slate-500">Depth progression, drilling parameters & predicted risk</p>
        </div>
        <select
          value={selected || ''}
          onChange={(e) => setSelected(e.target.value)}
          className="btn bg-rig-panel text-slate-200 font-mono"
        >
          {wells.map((w) => (
            <option key={w.well_id} value={w.well_id}>{w.well_id}</option>
          ))}
        </select>
      </header>

      {loading || !detail?.available ? (
        <Loading label="Loading well telemetry..." />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Stat label="Avg ROP" value={`${detail.summary.avg_rop} ft/hr`} />
            <Stat label="Max Depth" value={`${detail.summary.max_depth.toLocaleString()} ft`} />
            <Stat label="Avg Bit Wear" value={`${(detail.summary.avg_bit_wear * 100).toFixed(0)}%`} />
            <Stat label="Data Points" value={detail.summary.points.toLocaleString()} />
          </div>

          <Card>
            <SectionTitle sub="Actual vs predicted ROP over depth">ROP Trend</SectionTitle>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={detail.series}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                <XAxis dataKey="depth" stroke="#64748b" fontSize={11} unit="ft" />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip {...ts} />
                <Legend />
                <Line type="monotone" dataKey="rate_of_penetration" name="Actual ROP" stroke={CHART_COLORS.amber} dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="pred_rop" name="Predicted ROP" stroke={CHART_COLORS.rop} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <SectionTitle sub="WOB & RPM vs depth">Drilling Parameters</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={detail.series}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="depth" stroke="#64748b" fontSize={11} unit="ft" />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip {...ts} />
                  <Legend />
                  <Line type="monotone" dataKey="weight_on_bit" name="WOB (klbs)" stroke={CHART_COLORS.green} dot={false} />
                  <Line type="monotone" dataKey="rpm" name="RPM" stroke={CHART_COLORS.purple} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <SectionTitle sub="Torque & vibration analysis">Torque & Vibration</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={detail.series}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="depth" stroke="#64748b" fontSize={11} unit="ft" />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip {...ts} />
                  <Legend />
                  <Line type="monotone" dataKey="torque" name="Torque" stroke={CHART_COLORS.amber} dot={false} />
                  <Line type="monotone" dataKey="vibration_level" name="Vibration" stroke={CHART_COLORS.red} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card>
            <SectionTitle sub="Cumulative bit wear progression">Bit Wear Progression</SectionTitle>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={detail.series}>
                <defs>
                  <linearGradient id="bw" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={CHART_COLORS.amber} stopOpacity={0.5} />
                    <stop offset="100%" stopColor={CHART_COLORS.amber} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                <XAxis dataKey="depth" stroke="#64748b" fontSize={11} unit="ft" />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} />
                <Tooltip {...ts} />
                <Area type="monotone" dataKey="bit_wear" name="Bit Wear" stroke={CHART_COLORS.amber} fill="url(#bw)" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <SectionTitle>Formation Transitions</SectionTitle>
            <div className="flex flex-wrap gap-2">
              {detail.formations.map((f, i) => (
                <span key={i} className="pill">
                  <span className="text-rig-neon">{f.formation_type}</span>
                  <span className="text-slate-500"> · {f.lo}–{f.hi} ft</span>
                </span>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <Card className="py-4">
      <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1 text-xl font-bold text-rig-neon">{value}</p>
    </Card>
  )
}
