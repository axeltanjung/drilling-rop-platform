import { motion } from 'framer-motion'

export function Card({ children, className = '', ...props }) {
  return (
    <div className={`glass p-5 ${className}`} {...props}>
      {children}
    </div>
  )
}

export function KpiCard({ icon: Icon, label, value, unit, accent = 'neon', delay = 0, sub }) {
  const accentMap = {
    neon: 'text-rig-neon',
    amber: 'text-rig-accent',
    green: 'text-rig-ok',
    red: 'text-rig-danger',
  }
  return (
    <motion.div
      className="kpi"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
          <div className="mt-2 flex items-baseline gap-1">
            <span className={`text-3xl font-bold ${accentMap[accent]}`}>{value}</span>
            {unit && <span className="text-sm text-slate-500">{unit}</span>}
          </div>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        {Icon && (
          <div className={`p-2 rounded-xl bg-rig-panel/60 ${accentMap[accent]}`}>
            <Icon size={22} />
          </div>
        )}
      </div>
      <div className="absolute -bottom-6 -right-6 h-20 w-20 rounded-full bg-rig-neon2/10 blur-2xl" />
    </motion.div>
  )
}

export function SectionTitle({ children, sub }) {
  return (
    <div className="mb-4">
      <h2 className="text-lg font-semibold text-slate-100">{children}</h2>
      {sub && <p className="text-sm text-slate-500">{sub}</p>}
    </div>
  )
}

export function Loading({ label = 'Loading drilling telemetry...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-slate-400">
      <div className="h-10 w-10 rounded-full border-2 border-rig-border border-t-rig-neon animate-spin" />
      <p className="mt-4 text-sm">{label}</p>
    </div>
  )
}

export function EmptyState({ message }) {
  return (
    <Card className="text-center py-16">
      <p className="text-slate-300 font-medium">No data yet</p>
      <p className="text-sm text-slate-500 mt-2">{message}</p>
      <code className="mt-4 inline-block text-xs bg-rig-panel px-3 py-2 rounded-lg text-rig-neon">
        python backend/training/train_pipeline.py && python backend/training/batch_predict.py
      </code>
    </Card>
  )
}

export function RiskBadge({ level }) {
  const map = {
    High: 'text-rig-danger border-rig-danger/40 bg-rig-danger/10',
    Medium: 'text-rig-accent border-rig-accent/40 bg-rig-accent/10',
    Low: 'text-rig-ok border-rig-ok/40 bg-rig-ok/10',
  }
  return <span className={`pill ${map[level] || ''}`}>{level} Risk</span>
}
