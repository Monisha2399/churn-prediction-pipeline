import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts'
import { GitBranch, Database, Cpu, Shield } from 'lucide-react'

const EXPERIMENTS = [
  { name: 'Baseline LR',     features: 'CRM + Behavioural',            roc_auc: null, f1: null, recall: null },
  { name: '+ Classic NLP',   features: '+ TF-IDF + Sentiment',         roc_auc: null, f1: null, recall: null },
  { name: '+ BERT LR',       features: '+ BERT Embeddings',            roc_auc: null, f1: null, recall: null },
  { name: 'Best (HGBC)',     features: 'All Features',                  roc_auc: null, f1: null, recall: null },
]

const TECH = [
  { icon: Database, label: 'Data',       items: ['IBM Telco CRM (Kaggle)', 'Synthetic behavioural logs', 'Synthetic support tickets'] },
  { icon: Cpu,      label: 'Models',     items: ['Logistic Regression (3×)', 'HistGradientBoostingClassifier', 'all-MiniLM-L6-v2 (BERT)'] },
  { icon: GitBranch,label: 'MLOps',      items: ['MLflow tracking + registry', 'Apache Airflow DAG (weekly)', 'Docker Compose on EC2'] },
  { icon: Shield,   label: 'Security',   items: ['IAM role (no long-lived keys)', 'SG restricted to My IP', 'Secrets in .env only'] },
]

const RADAR_DATA = [
  { metric: 'No Leakage',       score: 100 },
  { metric: 'NLP Quality',      score: 85  },
  { metric: 'Reproducibility',  score: 90  },
  { metric: 'Engineering',      score: 88  },
  { metric: 'Monitoring',       score: 80  },
  { metric: 'Security',         score: 92  },
]

export default function ModelInfo() {
  const hasMetrics = EXPERIMENTS.some(e => e.roc_auc !== null)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Model Info</h1>
        <p className="text-gray-400 text-sm mt-1">
          Experiment comparison, feature pipeline, and architecture overview.
        </p>
      </div>

      {/* Experiment table */}
      <div className="card">
        <h2 className="text-base font-semibold text-white mb-4">Experiment Comparison</h2>
        {!hasMetrics && (
          <div className="text-center py-6 text-gray-500 text-sm">
            Metrics will populate here after running the Airflow pipeline.
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Experiment</th>
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Features</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">ROC-AUC</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">F1</th>
                <th className="text-right py-3 px-4 text-gray-400 font-medium">Recall</th>
              </tr>
            </thead>
            <tbody>
              {EXPERIMENTS.map((exp, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/40 transition-colors">
                  <td className="py-3 px-4 text-white font-medium">{exp.name}</td>
                  <td className="py-3 px-4 text-gray-400">{exp.features}</td>
                  <td className="py-3 px-4 text-right">
                    {exp.roc_auc ? (
                      <span className="text-brand-400 font-semibold">{exp.roc_auc.toFixed(4)}</span>
                    ) : (
                      <span className="text-gray-600">—</span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {exp.f1 ? (
                      <span className="text-green-400">{exp.f1.toFixed(4)}</span>
                    ) : (
                      <span className="text-gray-600">—</span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right">
                    {exp.recall ? (
                      <span className="text-yellow-400">{exp.recall.toFixed(4)}</span>
                    ) : (
                      <span className="text-gray-600">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-600 mt-3">
          Note: Accuracy alone is misleading for imbalanced churn data (~15–20% positive). 
          Recall and ROC-AUC are primary metrics.
        </p>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pipeline quality radar */}
        <div className="card">
          <h2 className="text-base font-semibold text-white mb-4">Pipeline Quality Dimensions</h2>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 10 }} />
              <Radar name="Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} strokeWidth={2} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature importance placeholder */}
        <div className="card">
          <h2 className="text-base font-semibold text-white mb-4">Feature Groups Used</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart
              layout="vertical"
              data={[
                { name: 'BERT Embeddings', dims: 384 },
                { name: 'TF-IDF',          dims: 5000 },
                { name: 'Behavioural',     dims: 5    },
                { name: 'Ticket Counts',   dims: 6    },
                { name: 'Sentiment',       dims: 3    },
                { name: 'CRM',             dims: 4    },
              ]}
              margin={{ top: 0, right: 20, left: 10, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} width={110} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#f9fafb' }}
                formatter={(v) => [`${v} dims`, 'Dimensions']}
              />
              <Bar dataKey="dims" fill="#6366f1" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tech stack cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {TECH.map(({ icon: Icon, label, items }) => (
          <div key={label} className="card">
            <div className="flex items-center gap-2 mb-3">
              <Icon className="w-5 h-5 text-brand-400" />
              <h3 className="font-semibold text-white">{label}</h3>
            </div>
            <ul className="space-y-1.5">
              {items.map((item, i) => (
                <li key={i} className="text-xs text-gray-400 flex items-start gap-1.5">
                  <span className="text-brand-500 mt-0.5">·</span>{item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Label definition callout */}
      <div className="card border-yellow-700 bg-yellow-900/10">
        <h3 className="text-yellow-400 font-semibold mb-2">Label Definition (No Leakage)</h3>
        <p className="text-gray-300 text-sm font-mono bg-gray-800 rounded p-3">
          label = 1  if  churn_date ∈ (snapshot_date, snapshot_date + 30 days]
        </p>
        <p className="text-gray-400 text-xs mt-2">
          All features use only data strictly before the snapshot date. 
          Train/test split is time-based — never random.
        </p>
      </div>
    </div>
  )
}
