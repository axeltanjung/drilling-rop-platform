import { useEffect, useState } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { Gauge, Activity, Layers, ShieldAlert, DollarSign } from 'lucide-react'
import { api } from '../api/client.js'
import { KpiCard, Card, SectionTitle, Loading, EmptyState } from '../components/UI.jsx'
import { CHART_COLORS } from '../config.js'

const tooltipStyle = {
  contentStyle: { background: '#0f1729', border: '1px solid #1f2a44', borderRadius: 12 },
  labelStyle: { color: '#94a3b8' },
}

export default function Overview() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.dashboardSummary().then(setData).finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading />
  if (!data?.available)
    return <EmptyState message={data?.message || 'Train models and run batch prediction to populate the dashboard.'} />

  const k = data.kpis
  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Executive Overview</h1>
          <p className="text-sm text-slate-500">Fleet-wide drilling performance & AI predictions</p>
        </div>
        <span className="pill neon-text">{k.total_records.toLocaleString()} records analyzed</span>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <KpiCard icon={Gauge} label="Avg ROP" value={k.avg_rop} unit="ft/hr" accent="neon" delay={0.05} />
        <KpiCard icon={Activity} label="Drilling Efficiency" value={`${k.avg_efficiency}%`} accent="green" delay={0.1} />
        <KpiCard icon={Layers} label="Active Wells" value={k.active_wells} accent="neon" delay={0.15} />
        <KpiCard icon={ShieldAlert} label="Operational Risk" value={`${k.operational_risk}%`} accent={k.operational_risk > 50 ? 'red' : 'amber'} delay={0.2} />
        <KpiCard icon={DollarSign} label="Cost Index" value={k.cost_index} sub="lower is better" accent="amber" delay={0.25} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <SectionTitle sub="Predicted ROP by formation type">Formation Performance</SectionTitle>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.formation_comparison}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
              <XAxis dataKey="formation_type" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="avg_rop" name="Avg ROP (ft/hr)" fill={CHART_COLORS.rop} radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle sub="Average predicted ROP across depth">Drilling Performance Trend</SectionTitle>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.rop_trend}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
              <XAxis dataKey="depth_bucket" stroke="#64748b" fontSize={12} unit="ft" />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="avg_rop" name="Avg ROP" stroke={CHART_COLORS.neon2 || '#0ea5e9'} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card>
        <SectionTitle sub="Per-well predicted performance & risk">Well Fleet Summary</SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-rig-border">
                <th className="py-2 px-3">Well</th>
                <th className="py-2 px-3">Avg ROP</th>
                <th className="py-2 px-3">Max Depth</th>
                <th className="py-2 px-3">Efficiency</th>
                <th className="py-2 px-3">Vibration Risk</th>
                <th className="py-2 px-3">Bit Damage Risk</th>
              </tr>
            </thead>
            <tbody>
              {data.wells.map((w) => (
                <tr key={w.well_id} className="border-b border-rig-border/50 hover:bg-rig-panel/40">
                  <td className="py-2 px-3 font-mono text-rig-neon">{w.well_id}</td>
                  <td className="py-2 px-3">{w.avg_rop?.toFixed(1)}</td>
                  <td className="py-2 px-3">{w.max_depth?.toLocaleString()} ft</td>
                  <td className="py-2 px-3">{(w.avg_efficiency * 100).toFixed(0)}%</td>
                  <td className="py-2 px-3">{(w.avg_vibration_risk * 100).toFixed(0)}%</td>
                  <td className="py-2 px-3">{(w.avg_bit_damage_risk * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
