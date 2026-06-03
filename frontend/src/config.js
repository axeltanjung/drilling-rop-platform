export const DEFAULT_CONDITIONS = {
  depth: 8500,
  formation_type: 'Shale',
  bit_type: 'PDC',
  weight_on_bit: 28,
  rpm: 120,
  torque: 35,
  mud_flow_rate: 600,
  standpipe_pressure: 2800,
  hook_load: 220,
  bit_wear: 0.2,
  mud_density: 11.5,
  vibration_level: 2.0,
  temperature: 160,
  drilling_hours: 40,
  pump_pressure: 2700,
  flow_out: 580,
  differential_pressure: 80,
}

export const FORMATIONS = ['Sandstone', 'Shale', 'Limestone', 'Dolomite', 'Granite', 'Salt']
export const BIT_TYPES = ['PDC', 'Tricone', 'Diamond', 'Hybrid']

export const CHART_COLORS = {
  rop: '#22d3ee',
  pred: '#0ea5e9',
  amber: '#f59e0b',
  green: '#10b981',
  red: '#ef4444',
  purple: '#a78bfa',
  grid: '#1f2a44',
}
