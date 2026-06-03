import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { motion } from 'framer-motion'
import { Brain, Sparkles, TrendingUp, TrendingDown } from 'lucide-react'
import { api } from '../api/client.js'
import { Card, SectionTitle } from '../components/UI.jsx'
import ConditionsForm from '../components/ConditionsForm.jsx'
import { DEFAULT_CONDITIONS, CHART_COLORS } from '../config.js'

const ts = {
  contentStyle: { background: '#0f1729', border: '1px solid #1f2a44', borderRadius: 12 },
  labelStyle: { color: '#94a3b8' },
}

export default function AIInsights() {
  const [conditions, setConditions] = useState(DEFAULT_CONDITIONS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = async () => {
    setLoading(true); setError(null)
    try {
      setResult(await api.explain(conditions))
    } catch (e) {
      setError(e.response?.data?.detail || 'Train models first to enable explainability.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <Brain className="text-rig-neon" /> AI Insights
        </h1>
        <p className="text-sm text-slate-500">SHAP explainability & drilling factor contribution</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <SectionTitle sub="Set drilling conditions">Scenario</SectionTitle>
          <ConditionsForm conditions={conditions} setConditions={setConditions} />
          <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 flex items-center justify-center gap-2">
            <Sparkles size={16} /> {loading ? 'Explaining...' : 'Explain Prediction'}
          </button>
          {error && <p className="text-xs text-rig-danger mt-3">{error}</p>}
        </Card>

        <div className="lg:col-span-2 space-y-6">
          {result ? (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Card>
                  <p className="text-xs uppercase text-slate-400">Predicted ROP</p>
                  <p className="text-3xl font-bold text-rig-neon mt-1">{result.predicted_rop}</p>
                  <p className="text-xs text-slate-500">ft/hr</p>
                </Card>
                <Card>
                  <p className="text-xs uppercase text-slate-400">Base Value</p>
                  <p className="text-3xl font-bold text-slate-300 mt-1">{result.base_value}</p>
                  <p className="text-xs text-slate-500">expected ROP</p>
                </Card>
                <Card>
                  <p className="text-xs uppercase text-slate-400">Confidence</p>
                  <p className={`text-2xl font-bold mt-1 ${
                    result.confidence?.level === 'high' ? 'text-rig-ok'
                    : result.confidence?.level === 'low' ? 'text-rig-danger' : 'text-rig-accent'}`}>
                    {result.confidence?.level || 'n/a'}
                  </p>
                  {result.confidence && (
                    <p className="text-xs text-slate-500">
                      ±{((result.confidence.interval_high - result.confidence.interval_low) / 2).toFixed(1)} ft/hr
                    </p>
                  )}
                </Card>
              </div>

              <Card>
                <SectionTitle sub="AI narrative">Why this prediction?</SectionTitle>
                <motion.p
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="text-slate-300 leading-relaxed"
                >
                  {result.narrative}
                </motion.p>
              </Card>

              <Card>
                <SectionTitle sub="SHAP contribution per drilling factor">Feature Contributions</SectionTitle>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={result.contributions} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis type="number" stroke="#64748b" fontSize={11} />
                    <YAxis type="category" dataKey="label" stroke="#64748b" fontSize={11} width={130} />
                    <Tooltip {...ts} />
                    <Bar dataKey="contribution" radius={[0, 6, 6, 0]}>
                      {result.contributions.map((c, i) => (
                        <Cell key={i} fill={c.contribution >= 0 ? CHART_COLORS.green : CHART_COLORS.red} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-3 space-y-1">
                  {result.contributions.slice(0, 5).map((c, i) => (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <span className="text-slate-300 flex items-center gap-2">
                        {c.contribution >= 0 ? <TrendingUp size={14} className="text-rig-ok" /> : <TrendingDown size={14} className="text-rig-danger" />}
                        {c.label} <span className="text-slate-600 font-mono">({c.value})</span>
                      </span>
                      <span className={c.contribution >= 0 ? 'text-rig-ok' : 'text-rig-danger'}>
                        {c.contribution >= 0 ? '+' : ''}{c.contribution}
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          ) : (
            <Card className="py-20 text-center text-slate-500">
              Configure a drilling scenario and run the explainer to see SHAP contributions.
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
