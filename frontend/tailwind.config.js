/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        rig: {
          bg: '#0a0e1a',
          panel: '#111827',
          card: '#0f1729',
          border: '#1f2a44',
          neon: '#22d3ee',
          neon2: '#0ea5e9',
          accent: '#f59e0b',
          danger: '#ef4444',
          warn: '#f59e0b',
          ok: '#10b981',
        },
      },
      boxShadow: {
        glow: '0 0 20px rgba(34, 211, 238, 0.25)',
        glowAmber: '0 0 20px rgba(245, 158, 11, 0.25)',
      },
      backdropBlur: { xs: '2px' },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
