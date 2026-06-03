import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'
import { ShieldAlert } from 'lucide-react'
import { api } from '../api/client.js'
import { Card, SectionTitle, Loading, EmptyState, RiskBadge } from '../components/UI.jsx'
import { CHART_COLORS } from '../config.js'

const ts = {
  contentStyle: { background: '#0f1729', border: '1px solid #1f2a44', borderRadius: 12 },
  labelStyle: { color: '#94a3b8' },
}

const levelOf = (v) => (v > 0.6 ? 'High' : v > 0.35 ? 'Medium' : 'Low')

export default function RiskAnalysis() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.riskOverview().then(setData).finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading />
  if (!data?.available) return <EmptyState message={data?.message || 'Run batch prediction to populate risk analytics.'} />

  const wellBars = data.by_well.map((w) => ({
    well: w.well_id,
    'Stuck Pipe': +(w.stuck * 100).toFixed(1),
    Vibration: +(w.vibration * 100).toFixed(1),
    'Bit Damage': +(w.bit_damage * 100).toFixed(1),
  }))

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <ShieldAlert className="text-rig-danger" /> Risk Analysis
        </h1>
        <p className="text-sm text-slate-500">Stuck pipe, vibration, bit damage & drilling instability</p>
      </header>

      <Card>
        <SectionTitle sub="Wells ranked by drilling instability index">Instability Ranking</SectionTitle>
        <div className="space-y-2">
          {data.by_well.map((w) => (
            <div key={w.well_id} className="flex items-center gap-3">
              <span className="font-mono text-rig-neon w-24">{w.well_id}</span>
              <div className="flex-1 h-3 rounded-full bg-rig-panel overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${w.instability_index * 100}%`,
                    background: w.instability_index > 0.6 ? '#ef4444' : w.instability_index > 0.35 ? '#f59e0b' : '#10b981',
                  }}
                />
              </div>
              <span className="w-12 text-right text-sm text-slate-300">{(w.instability_index * 100).toFixed(0)}%</span>
              <RiskBadge level={levelOf(w.instability_index)} />
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <SectionTitle sub="Risk breakdown per well (%)">Risk by Well</SectionTitle>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={wellBars}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
              <XAxis dataKey="well" stroke="#64748b" fontSize={10} angle={-30} textAnchor="end" height={60} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip {...ts} /><Legend />
              <Bar dataKey="Stuck Pipe" stackId="a" fill={CHART_COLORS.red} />
              <Bar dataKey="Vibration" stackId="a" fill={CHART_COLORS.amber} />
              <Bar dataKey="Bit Damage" stackId="a" fill={CHART_COLORS.purple} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle sub="Average risk by formation (%)">Risk by Formation</SectionTitle>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data.by_formation.map((f) => ({
              formation: f.formation_type,
              'Stuck Pipe': +(f.stuck * 100).toFixed(1),
              Vibration: +(f.vibration * 100).toFixed(1),
              'Bit Damage': +(f.bit_damage * 100).toFixed(1),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
              <XAxis dataKey="formation" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip {...ts} /><Legend />
              <Bar dataKey="Stuck Pipe" fill={CHART_COLORS.red} radius={[4, 4, 0, 0]} />
              <Bar dataKey="Vibration" fill={CHART_COLORS.amber} radius={[4, 4, 0, 0]} />
              <Bar dataKey="Bit Damage" fill={CHART_COLORS.purple} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  )
}
