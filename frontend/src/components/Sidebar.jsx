import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import { LayoutDashboard, Activity, Brain, Settings2, ShieldAlert, Gauge } from 'lucide-react'

const links = [
  { to: '/', label: 'Executive Overview', icon: LayoutDashboard, end: true },
  { to: '/wells', label: 'Well Detail', icon: Activity },
  { to: '/insights', label: 'AI Insights', icon: Brain },
  { to: '/optimize', label: 'Drilling Optimization', icon: Settings2 },
  { to: '/risk', label: 'Risk Analysis', icon: ShieldAlert },
]

export default function Sidebar({ health }) {
  return (
    <aside className="w-64 shrink-0 h-screen sticky top-0 border-r border-rig-border bg-rig-panel/40 backdrop-blur p-4 flex flex-col">
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex items-center gap-3 px-2 py-4"
      >
        <div className="p-2 rounded-xl bg-rig-neon2/20 text-rig-neon shadow-glow">
          <Gauge size={24} />
        </div>
        <div>
          <h1 className="text-sm font-bold text-slate-100 leading-tight">DrillOptAI</h1>
          <p className="text-[10px] uppercase tracking-widest text-slate-500">ROP Platform</p>
        </div>
      </motion.div>

      <nav className="mt-4 flex flex-col gap-1">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.end}
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <l.icon size={18} />
            <span className="text-sm">{l.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto px-2 py-4">
        <div className="glass p-3 flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              health?.status === 'ok' ? 'bg-rig-ok' : 'bg-rig-danger'
            } animate-pulse`}
          />
          <div className="text-xs">
            <p className="text-slate-300">API {health?.status === 'ok' ? 'Online' : 'Offline'}</p>
            <p className="text-slate-500">
              Models {health?.models_available ? 'ready' : 'not trained'}
            </p>
          </div>
        </div>
      </div>
    </aside>
  )
}
