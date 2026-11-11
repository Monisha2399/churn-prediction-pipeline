# Architecture Documentation

## System Overview

All services run on a **single AWS EC2 instance** (`m7i-flex.large`, 8 GB RAM)
with data stored in S3 and code in a public GitHub repository.

```
┌─────────────────────────────────────────────────────────┐
│                    EC2 (m7i-flex.large)                  │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ PostgreSQL│  │  MLflow  │  │  Airflow │  │Grafana │  │
│  │  :5432   │  │  :5000   │  │  :8080   │  │ :3000  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│       │              │              │             │       │
│  ┌────┴──────────────┴──────────────┴─────────────┴───┐  │
│  │              FastAPI  :8000   /  Frontend :3001     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Prometheus  :9090                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
    AWS S3 Bucket             GitHub Repository
  (raw data, embeddings,      (code, DAGs, configs)
   MLflow artifacts)
```

---

## Component Details

### 1. Data Ingestion
- **boto3** reads raw CSVs from S3 (`raw/crm/`, `raw/tickets/`, `raw/logs/`)
- **SQLAlchemy** upserts into PostgreSQL tables
- All writes are idempotent (upsert / truncate-and-load)

### 2. Preprocessing
- **Pandas** for tabular data cleaning, type coercion, missing value imputation
- **NLTK** for text: tokenisation, stopword removal, lemmatisation
- Two text cleaning paths:
  - Light cleaning → BERT (preserves case and punctuation)
  - Heavy cleaning → TF-IDF (lowercase, no stopwords, lemmatised)

### 3. Feature Engineering
| Feature Group | Columns | Method |
|---------------|---------|--------|
| CRM | tenure, charges, contract | ordinal encoding |
| Behavioural | logins, session_minutes, login_trend | aggregation over 30d window |
| Ticket counts | ticket_count_30d/60d/90d, escalations | COUNT per window |
| Sentiment | mean, min, trend | NLTK VADER compound scores |
| TF-IDF | 5,000 n-gram features | sklearn TfidfVectorizer |
| BERT embeddings | 384-dim vectors | all-MiniLM-L6-v2 (mean pooled) |

### 4. Model Training
Four experiments run in sequence, each as a separate MLflow run:

| Name | Features | Model |
|------|----------|-------|
| baseline_lr | CRM + Behavioural | LogisticRegression |
| classic_nlp_lr | + TF-IDF + Sentiment | LogisticRegression |
| bert_lr | + BERT embeddings | LogisticRegression |
| best_effort_hgbc | All features | HistGradientBoostingClassifier |

**Anti-leakage measures:**
- `TfidfVectorizer` fitted on training data only
- BERT embeddings cached per snapshot; never recomputed from test data
- Time-based train/test split (no random splits)

### 5. MLflow
- **Backend store:** PostgreSQL (`churndb`)
- **Artifact store:** S3 bucket (`mlflow-artifacts/`)
- **Model Registry:** best model registered as `churn_classifier@champion`
- FastAPI loads `@champion` at startup via `mlflow.sklearn.load_model`

### 6. Airflow
- **Executor:** LocalExecutor (single instance, saves memory vs Celery)
- **Metadata DB:** same PostgreSQL container (`airflowdb` database)
- **DAG:** `churn_pipeline` — weekly, 6 tasks, all idempotent
- Promotes new champion only if ROC-AUC improves by ≥ 0.001

### 7. FastAPI
- Loads `@champion` model once at startup (singleton)
- Prometheus metrics exposed at `/metrics`
- Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`
- Pydantic v2 schemas with input validation

### 8. Monitoring
- **Prometheus** scrapes FastAPI `/metrics` every 15s
- **Grafana** dashboards:
  - API request rate, p95 latency, error rate
  - Daily predicted churn rate (from `predictions` table)
  - Score distribution histogram
  - Mean sentiment drift check

### 9. Frontend (React)
- Built with React 18 + Vite + Tailwind CSS
- Served via Nginx in Docker on port 3001
- Features: live churn prediction form, results display, model metrics panel

---

## Security

| Control | Implementation |
|---------|----------------|
| IAM Role | EC2 instance role grants S3 access — no long-lived keys |
| Security Group | All ports restricted to `My IP` only — no `0.0.0.0/0` |
| Secrets | `.env` on server only, `.gitignore`d — never in Git |
| Budget Alarm | AWS Budgets alert at $10/month |
| Non-root containers | API runs as `appuser` in Docker |
| Grafana | Default admin password changed on first login |

---

## Cost Controls

- Stop EC2 every evening (`docker compose stop` preserves data in volumes)
- Use `m7i-flex.large` — ~$0.095/hr; ~$2.28/day if left running
- Embeddings cached in S3 — no repeated transformer inference on retraining
- PostgreSQL in Docker (not RDS) — saves ~$20-30/month
