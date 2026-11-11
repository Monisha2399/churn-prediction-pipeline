import React, { useState } from 'react'
import { Search, AlertTriangle, CheckCircle, Loader } from 'lucide-react'
import { predict } from '../api'

const FIELDS = [
  { key: 'customer_id',      label: 'Customer ID',          type: 'text',   placeholder: 'e.g. 7590-VHVEG', required: true },
  { key: 'tenure_months',    label: 'Tenure (months)',       type: 'number', placeholder: '12'  },
  { key: 'monthly_charges',  label: 'Monthly Charges ($)',   type: 'number', placeholder: '70.5' },
  { key: 'contract_encoded', label: 'Contract Type',         type: 'select', options: [
      { value: '0', label: 'Month-to-month' },
      { value: '1', label: 'One year' },
      { value: '2', label: 'Two year' },
  ]},
  { key: 'logins_30d',       label: 'Logins (last 30d)',     type: 'number', placeholder: '15'  },
  { key: 'session_min_30d',  label: 'Session Minutes (30d)', type: 'number', placeholder: '300' },
  { key: 'failed_pay_30d',   label: 'Failed Payments (30d)', type: 'number', placeholder: '0'   },
  { key: 'days_since_login', label: 'Days Since Last Login',  type: 'number', placeholder: '2'   },
  { key: 'ticket_count_30d', label: 'Support Tickets (30d)', type: 'number', placeholder: '1'   },
  { key: 'escalations_30d',  label: 'Escalations (30d)',     type: 'number', placeholder: '0'   },
  { key: 'sentiment_mean',   label: 'Avg Sentiment Score',   type: 'number', placeholder: '-0.2', step: '0.01', min: '-1', max: '1' },
]

function ProbBar({ prob }) {
  const pct   = Math.round(prob * 100)
  const color = pct >= 60 ? 'bg-red-500' : pct >= 40 ? 'bg-yellow-500' : 'bg-green-500'
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>0%</span>
        <span className="font-bold text-white text-base">{pct}%</span>
        <span>100%</span>
      </div>
      <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function Predict() {
  const [form, setForm]       = useState({ contract_encoded: '0' })
  const [threshold, setThreshold] = useState(0.4)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (key, val) => setForm(f => ({ ...f, [key]: val }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)

    // Build customer payload — convert numeric strings to numbers
    const customer = Object.fromEntries(
      Object.entries(form).map(([k, v]) => {
        if (k === 'customer_id') return [k, v]
        const n = parseFloat(v)
        return [k, isNaN(n) ? undefined : n]
      }).filter(([, v]) => v !== undefined && v !== '')
    )

    try {
      const res = await predict(customer, threshold)
      setResult(res)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'API error — is the service running?')
    } finally {
      setLoading(false)
    }
  }

  const isChurn = result?.churn_label === 1

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Predict Churn</h1>
        <p className="text-gray-400 text-sm mt-1">
          Enter customer features to get a 30-day churn probability.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* ── Form ── */}
        <form onSubmit={handleSubmit} className="card lg:col-span-3 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {FIELDS.map(({ key, label, type, placeholder, options, step, min, max, required }) => (
              <div key={key}>
                <label className="label">{label}{required && <span className="text-red-400 ml-1">*</span>}</label>
                {type === 'select' ? (
                  <select
                    className="input-field"
                    value={form[key] ?? '0'}
                    onChange={e => handleChange(key, e.target.value)}
                  >
                    {options.map(o => (
                      <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type={type}
                    className="input-field"
                    placeholder={placeholder}
                    step={step ?? (type === 'number' ? '0.01' : undefined)}
                    min={min}
                    max={max}
                    required={required}
                    onChange={e => handleChange(key, e.target.value)}
                  />
                )}
              </div>
            ))}
          </div>

          {/* Threshold slider */}
          <div>
            <label className="label">
              Classification Threshold: <span className="text-brand-400 font-semibold">{threshold}</span>
            </label>
            <input
              type="range" min="0.1" max="0.9" step="0.05"
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-brand-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>More sensitive (0.1)</span>
              <span>More conservative (0.9)</span>
            </div>
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
            {loading ? <Loader className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {loading ? 'Predicting…' : 'Predict Churn'}
          </button>
        </form>

        {/* ── Result panel ── */}
        <div className="card lg:col-span-2 flex flex-col justify-center">
          {!result && !error && !loading && (
            <div className="text-center text-gray-500 py-12">
              <Search className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p className="text-sm">Fill in the form and click Predict</p>
            </div>
          )}

          {loading && (
            <div className="text-center text-gray-400 py-12">
              <Loader className="w-10 h-10 mx-auto mb-3 animate-spin text-brand-400" />
              <p className="text-sm">Running inference…</p>
            </div>
          )}

          {error && (
            <div className="text-center py-6">
              <AlertTriangle className="w-10 h-10 mx-auto mb-3 text-red-400" />
              <p className="text-red-400 text-sm font-medium">{error}</p>
            </div>
          )}

          {result && (
            <div className="space-y-5">
              <div className="text-center">
                {isChurn ? (
                  <AlertTriangle className="w-14 h-14 mx-auto text-red-400 mb-2" />
                ) : (
                  <CheckCircle className="w-14 h-14 mx-auto text-green-400 mb-2" />
                )}
                <span className={isChurn ? 'badge-churn text-base' : 'badge-safe text-base'}>
                  {isChurn ? '⚠ High Churn Risk' : '✓ Low Churn Risk'}
                </span>
              </div>

              <div>
                <p className="text-gray-400 text-sm mb-2 text-center">Churn Probability</p>
                <ProbBar prob={result.churn_probability} />
              </div>

              <div className="bg-gray-800 rounded-lg p-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Customer</span>
                  <span className="text-white font-mono">{result.customer_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Probability</span>
                  <span className="text-white font-semibold">{(result.churn_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Model</span>
                  <span className="text-brand-400">{result.model_version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Threshold</span>
                  <span className="text-white">{threshold}</span>
                </div>
              </div>

              {isChurn && (
                <div className="bg-red-900/20 border border-red-800 rounded-lg p-3 text-xs text-red-300">
                  <strong>Action:</strong> Trigger retention workflow — offer discount or personal outreach within 48h.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
