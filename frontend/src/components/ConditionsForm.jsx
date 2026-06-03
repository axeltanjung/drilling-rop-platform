import { FORMATIONS, BIT_TYPES } from '../config.js'

const SLIDERS = [
  { key: 'weight_on_bit', label: 'Weight on Bit', unit: 'klbs', min: 8, max: 55, step: 1 },
  { key: 'rpm', label: 'RPM', unit: 'rev/min', min: 60, max: 200, step: 1 },
  { key: 'mud_flow_rate', label: 'Mud Flow Rate', unit: 'gpm', min: 350, max: 850, step: 5 },
  { key: 'depth', label: 'Depth', unit: 'ft', min: 500, max: 20000, step: 100 },
  { key: 'bit_wear', label: 'Bit Wear', unit: 'frac', min: 0, max: 1, step: 0.01 },
  { key: 'vibration_level', label: 'Vibration', unit: '0-10', min: 0, max: 10, step: 0.1 },
]

export default function ConditionsForm({ conditions, setConditions }) {
  const update = (key, value) =>
    setConditions((c) => ({ ...c, [key]: Number(value) }))
  const updateStr = (key, value) => setConditions((c) => ({ ...c, [key]: value }))

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-400">Formation</label>
          <select
            value={conditions.formation_type}
            onChange={(e) => updateStr('formation_type', e.target.value)}
            className="btn w-full mt-1"
          >
            {FORMATIONS.map((f) => <option key={f}>{f}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-slate-400">Bit Type</label>
          <select
            value={conditions.bit_type}
            onChange={(e) => updateStr('bit_type', e.target.value)}
            className="btn w-full mt-1"
          >
            {BIT_TYPES.map((b) => <option key={b}>{b}</option>)}
          </select>
        </div>
      </div>

      {SLIDERS.map((s) => (
        <div key={s.key}>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">{s.label}</span>
            <span className="font-mono text-rig-neon">
              {conditions[s.key]} <span className="text-slate-600">{s.unit}</span>
            </span>
          </div>
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={conditions[s.key]}
            onChange={(e) => update(s.key, e.target.value)}
            className="w-full mt-1 accent-rig-neon"
          />
        </div>
      ))}
    </div>
  )
}
