import { useState } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import { motion } from 'framer-motion'
import { Settings2, Zap, Download, ArrowRight } from 'lucide-react'
import { api } from '../api/client.js'
import { Card, SectionTitle } from '../components/UI.jsx'
import ConditionsForm from '../components/ConditionsForm.jsx'
import { DEFAULT_CONDITIONS, CHART_COLORS } from '../config.js'

const ts = {
  contentStyle: { background: '#0f1729', border: '1px solid #1f2a44', borderRadius: 12 },
  labelStyle: { color: '#94a3b8' },
}

export default function Optimization() {
  const [conditions, setConditions] = useState(DEFAULT_CONDITIONS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = async () => {
    setLoading(true); setError(null)
    try {
      setResult(await api.optimize(conditions))
    } catch (e) {
      setError(e.response?.data?.detail || 'Train models first to enable optimization.')
    } finally {
      setLoading(false)
    }
  }

  const downloadPdf = async () => {
    const blob = await api.downloadReport(conditions)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'optimization_report.pdf'; a.click()
    URL.revokeObjectURL(url)
  }

  const paramData = result
    ? Object.keys(result.recommended_parameters).map((k) => ({
        param: k.replace(/_/g, ' '),
        Current: result.baseline_parameters[k],
        Recommended: result.recommended_parameters[k],
      }))
    : []

  const riskData = result
    ? Object.keys(result.risk_before).map((k) => ({
        risk: k.replace(/_/g, ' '),
        Before: +(result.risk_before[k] * 100).toFixed(1),
        After: +(result.risk_after[k] * 100).toFixed(1),
      }))
    : []

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <Settings2 className="text-rig-neon" /> Drilling Optimization
        </h1>
        <p className="text-sm text-slate-500">AI-recommended WOB / RPM / Mud Flow for higher ROP & lower risk</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <SectionTitle sub="Current drilling state">Conditions</SectionTitle>
          <ConditionsForm conditions={conditions} setConditions={setConditions} />
          <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 flex items-center justify-center gap-2">
            <Zap size={16} /> {loading ? 'Optimizing...' : 'Optimize Parameters'}
          </button>
          {error && <p className="text-xs text-rig-danger mt-3">{error}</p>}
        </Card>

        <div className="lg:col-span-2 space-y-6">
          {result ? (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Card>
                  <p className="text-xs uppercase text-slate-400">Baseline ROP</p>
                  <p className="text-3xl font-bold text-slate-300 mt-1">{result.baseline_rop}</p>
                </Card>
                <Card className="shadow-glow">
                  <p className="text-xs uppercase text-slate-400">Optimized ROP</p>
                  <p className="text-3xl font-bold text-rig-neon mt-1">{result.optimized_rop}</p>
                </Card>
                <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }} className="kpi">
                  <p className="text-xs uppercase text-slate-400">Improvement</p>
                  <p className={`text-3xl font-bold mt-1 ${result.improvement_pct >= 0 ? 'text-rig-ok' : 'text-rig-danger'}`}>
                    {result.improvement_pct >= 0 ? '+' : ''}{result.improvement_pct}%
                  </p>
                </motion.div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <SectionTitle sub="Current vs recommended">Parameter Comparison</SectionTitle>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={paramData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                      <XAxis dataKey="param" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip {...ts} /><Legend />
                      <Bar dataKey="Current" fill="#475569" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="Recommended" fill={CHART_COLORS.rop} radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card>
                  <SectionTitle sub="Risk before vs after (%)">Risk Trade-offs</SectionTitle>
                  <ResponsiveContainer width="100%" height={260}>
                    <RadarChart data={riskData}>
                      <PolarGrid stroke={CHART_COLORS.grid} />
                      <PolarAngleAxis dataKey="risk" stroke="#94a3b8" fontSize={11} />
                      <Radar name="Before" dataKey="Before" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.25} />
                      <Radar name="After" dataKey="After" stroke={CHART_COLORS.rop} fill={CHART_COLORS.rop} fillOpacity={0.3} />
                      <Legend /><Tooltip {...ts} />
                    </RadarChart>
                  </ResponsiveContainer>
                </Card>
              </div>

              <Card>
                <div className="flex items-center justify-between">
                  <SectionTitle sub="AI recommendations">Operational Trade-offs</SectionTitle>
                  <button onClick={downloadPdf} className="btn flex items-center gap-2">
                    <Download size={15} /> Export PDF
                  </button>
                </div>
                <ul className="space-y-2">
                  {result.trade_offs.map((t, i) => (
                    <motion.li
                      key={i}
                      initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="flex items-center gap-2 text-sm text-slate-300"
                    >
                      <ArrowRight size={15} className="text-rig-neon" /> {t}
                    </motion.li>
                  ))}
                </ul>
              </Card>
            </>
          ) : (
            <Card className="py-20 text-center text-slate-500">
              Set current drilling conditions and run the optimizer to get AI-recommended parameters.
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
