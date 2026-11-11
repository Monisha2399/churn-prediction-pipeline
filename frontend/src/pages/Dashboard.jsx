import React, { useEffect, useState } from 'react'
import { Activity, AlertTriangle, CheckCircle, Clock, TrendingUp, Users } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import { getHealth } from '../api'

const MOCK_TREND = [
  { date: 'Oct 1',  churn_rate: 14.2 },
  { date: 'Oct 8',  churn_rate: 15.1 },
  { date: 'Oct 15', churn_rate: 13.8 },
  { date: 'Oct 22', churn_rate: 16.4 },
  { date: 'Oct 29', churn_rate: 17.0 },
  { date: 'Nov 1',  churn_rate: 18.3 },
]

const MOCK_PIE = [
  { name: 'Will Stay',  value: 82, color: '#22c55e' },
  { name: 'At Risk',    value: 18, color: '#ef4444' },
]

function StatCard({ icon: Icon, label, value, sub, color = 'text-brand-400' }) {
  return (
    <div className="card flex items-start gap-4">
      <div className={`p-3 rounded-lg bg-gray-800 ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-gray-400 text-sm">{label}</p>
        <p className="text-2xl font-bold text-white mt-0.5">{value}</p>
        {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-6">
      {/* Page title */}
      <div>
        <h1 className="text-2xl font-bold text-white">Overview</h1>
        <p className="text-gray-400 text-sm mt-1">
          Customer churn prediction pipeline — live status and trend metrics
        </p>
      </div>

      {/* API status banner */}
      <div className={`flex items-center gap-3 p-4 rounded-xl border ${
        loading ? 'border-gray-700 bg-gray-800/50' :
        health   ? 'border-green-700 bg-green-900/20' :
                   'border-red-700 bg-red-900/20'
      }`}>
        {loading ? (
          <Clock className="w-5 h-5 text-gray-400 animate-spin" />
        ) : health ? (
          <CheckCircle className="w-5 h-5 text-green-400" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-red-400" />
        )}
        <div>
          <p className="font-medium text-white text-sm">
            {loading ? 'Checking API…' : health ? 'API Online' : 'API Offline'}
          </p>
          {health && (
            <p className="text-xs text-gray-400">
              Model: <span className="text-brand-400">{health.model_version}</span>
              {' · '}Uptime: {Math.round(health.uptime_seconds)}s
            </p>
          )}
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Users}     label="Total Customers"    value="7,043"  sub="IBM Telco dataset"     color="text-blue-400" />
        <StatCard icon={AlertTriangle} label="At-Risk (Latest)" value="1,268" sub="churn_label = 1"    color="text-red-400"  />
        <StatCard icon={TrendingUp} label="ROC-AUC (Best)"    value="—"      sub="Run pipeline first"   color="text-purple-400" />
        <StatCard icon={Activity}   label="Model Version"     value={health?.model_version ?? '—'} sub="@champion alias" color="text-green-400" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Churn rate trend */}
        <div className="card lg:col-span-2">
          <h2 className="text-base font-semibold text-white mb-4">Weekly Predicted Churn Rate (%)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={MOCK_TREND} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="churnGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} unit="%" />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#f9fafb' }}
              />
              <Area type="monotone" dataKey="churn_rate" stroke="#ef4444" strokeWidth={2} fill="url(#churnGrad)" name="Churn Rate %" />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">* Sample data — populate by running the Airflow pipeline</p>
        </div>

        {/* Pie chart */}
        <div className="card">
          <h2 className="text-base font-semibold text-white mb-4">Latest Snapshot Split</h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={MOCK_PIE} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={3} dataKey="value">
                {MOCK_PIE.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Legend
                formatter={(value) => <span className="text-gray-300 text-sm">{value}</span>}
              />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Quick links */}
      <div className="card">
        <h2 className="text-base font-semibold text-white mb-3">Service Links</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'FastAPI Docs',  url: 'http://localhost:8000/docs',  color: 'text-green-400'  },
            { label: 'MLflow UI',     url: 'http://localhost:5000',        color: 'text-blue-400'   },
            { label: 'Airflow UI',    url: 'http://localhost:8080',        color: 'text-yellow-400' },
            { label: 'Grafana',       url: 'http://localhost:3000',        color: 'text-orange-400' },
          ].map(({ label, url, color }) => (
            <a
              key={label}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className={`block p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors ${color}`}
            >
              {label} →
            </a>
          ))}
        </div>
      </div>
    </div>
  )
}
