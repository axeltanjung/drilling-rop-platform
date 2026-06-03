import { useEffect, useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Sidebar from './components/Sidebar.jsx'
import { api } from './api/client.js'
import Overview from './pages/Overview.jsx'
import WellDetail from './pages/WellDetail.jsx'
import AIInsights from './pages/AIInsights.jsx'
import Optimization from './pages/Optimization.jsx'
import RiskAnalysis from './pages/RiskAnalysis.jsx'

export default function App() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    const ping = () => api.health().then(setHealth).catch(() => setHealth({ status: 'down' }))
    ping()
    const t = setInterval(ping, 15000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="flex min-h-screen">
      <Sidebar health={health} />
      <main className="flex-1 p-6 max-w-[1600px] mx-auto w-full">
        <AnimatePresence mode="wait">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/wells" element={<WellDetail />} />
              <Route path="/insights" element={<AIInsights />} />
              <Route path="/optimize" element={<Optimization />} />
              <Route path="/risk" element={<RiskAnalysis />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
