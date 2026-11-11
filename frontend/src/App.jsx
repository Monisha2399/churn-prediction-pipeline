import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { BarChart2, Search, Activity, Home } from 'lucide-react'
import Dashboard   from './pages/Dashboard'
import Predict     from './pages/Predict'
import ModelInfo   from './pages/ModelInfo'
import clsx from 'clsx'

const NAV = [
  { to: '/',        icon: Home,     label: 'Overview'   },
  { to: '/predict', icon: Search,   label: 'Predict'    },
  { to: '/model',   icon: BarChart2, label: 'Model Info' },
]

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Top bar ── */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="text-brand-500 w-7 h-7" />
          <span className="text-xl font-bold tracking-tight text-white">
            Churn<span className="text-brand-500">Guard</span>
          </span>
        </div>
        <nav className="flex gap-1">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-brand-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                )
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      {/* ── Page content ── */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        <Routes>
          <Route path="/"        element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/model"   element={<ModelInfo />} />
        </Routes>
      </main>

      {/* ── Footer ── */}
      <footer className="text-center text-gray-600 text-xs py-4 border-t border-gray-800">
        Customer Churn Prediction Pipeline · FastAPI + MLflow + Airflow + Grafana
      </footer>
    </div>
  )
}
