# Customer Churn Prediction Pipeline

> Predict customer churn **30 days in advance** by combining CRM records, support ticket text
> (NLP) and product behavioural logs, served through FastAPI and monitored with Grafana.

---

## Architecture

```
S3 / PostgreSQL
      │
      ▼
  Ingestion (boto3 + SQLAlchemy)
      │
      ▼
  Preprocessing (Pandas + NLTK)
      │
      ▼
  NLP Features (TF-IDF · Sentiment · BERT embeddings)
      │
      ▼
  Feature Engineering (structured + unstructured join)
      │
      ▼
  Model Training (LR · BERT · HGBC)  ──►  MLflow Tracking + Registry
      │
      ▼
  FastAPI Prediction Service  ──►  Prometheus + Grafana
      │
      ▼
  Apache Airflow DAG (weekly schedule)
```

All services run on **one AWS EC2 instance** (recommended: `m7i-flex.large`).

---

## Repository Layout

```
churn-prediction-pipeline/
├── README.md
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── data/                      # gitignored (except sample/)
│   └── sample/
├── notebooks/                 # exploration only
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── s3_loader.py
│   │   ├── db_loader.py
│   │   └── schema.sql
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tfidf_features.py
│   │   ├── sentiment_features.py
│   │   ├── bert_embeddings.py
│   │   └── feature_store.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluate.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── model_loader.py
├── dags/
│   └── churn_pipeline.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── datasources/
│       │   └── datasources.yml
│       └── dashboards/
│           └── churn_dashboard.json
├── scripts/
│   ├── generate_data.py
│   └── setup_db.py
├── tests/
│   ├── test_cleaner.py
│   ├── test_features.py
│   └── test_api.py
├── docs/
│   ├── data.md
│   └── architecture.md
└── .github/
    └── workflows/
        └── ci.yml
```

---


## GitHub Setup and Workflow

Create a public GitHub repository named `churn-prediction-pipeline`, with a
README, the Python `.gitignore` template, and the MIT license. Enable two-factor
authentication in GitHub settings. Install Git, then configure your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
ssh-keygen -t ed25519 -C "you@example.com"
```

Add the generated public key (`.pub`) in GitHub Settings → SSH and GPG keys.
Never share or commit the private key. Work on a feature branch and open a pull
request to `main`:

```bash
git clone https://github.com/Monisha2399/churn-prediction-pipeline.git
cd churn-prediction-pipeline
git switch -c feature/add-feature
```

Keep `.env` and real credentials out of Git. The checked-in `.env.example`
contains variable names only; set values in your local `.env`. Keep datasets in
`data/` out of Git apart from small, reviewed examples in `data/sample/`.
Store models and embeddings in S3 or MLflow, not the repository.

GitHub Actions runs Ruff and pytest on every push and pull request.

---

## Run on Ubuntu 24.04 EC2 (tested flow)

```bash
git clone https://github.com/Monisha2399/churn-prediction-pipeline.git
cd churn-prediction-pipeline
bash scripts/bootstrap_ubuntu.sh        # once: swap, Docker, AWS CLI  -> then log out / SSH back in
bash scripts/deploy.sh <your-s3-bucket> # build, generate data, start stack, run the DAG, smoke-test API
```
Security group (source = My IP): 22, 8000 (API), 5000 (MLflow), 8080 (Airflow), 3000 (Grafana), 3001 (frontend).
Design notes: Airflow tasks run `python -m src.pipeline_steps <step>` inside an isolated virtualenv; the API serves
only structured-feature models (`baseline_lr`, `structured_hgbc`) while the TF-IDF / BERT runs are logged to MLflow for comparison.

## Quick Start (manual)

### 1. Prerequisites

- Docker & Docker Compose installed
- AWS account with an S3 bucket
- EC2 instance (m7i-flex.large recommended) with IAM role for S3

### 2. Configure environment

```bash
cp .env.example .env
# Fill in real values in .env (never commit .env)
```

### 3. Generate synthetic data & upload to S3

```bash
pip install -r requirements.txt
python scripts/generate_data.py   # creates data/raw/
python scripts/setup_db.py        # loads into PostgreSQL
```

### 4. Run all services

```bash
docker compose up -d
```

Services:
| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |
| Grafana | http://localhost:3000 |

### 5. Trigger the pipeline

Open Airflow at `http://localhost:8080`, enable and trigger the `churn_pipeline` DAG.

---

## ML Experiments

| Experiment | Features | Model | ROC-AUC | F1 |
|------------|----------|-------|---------|-----|
| Baseline | CRM + Behavioural | Logistic Regression | — | — |
| + Classic NLP | + TF-IDF + Sentiment | Logistic Regression | — | — |
| + BERT | + BERT Embeddings | Logistic Regression | — | — |
| Best Effort | All features | HistGradientBoosting | — | — |

> Metrics are populated after a training run. See MLflow UI for full results.

---

## Data

See [docs/data.md](docs/data.md) for full description.

**Summary:** Synthetic dataset built on top of the IBM Telco Customer Churn base (~7,000 customers).
Behavioural logs and support tickets are synthetically generated with a configurable random seed (`RANDOM_SEED=42`).
The synthetic signal is intentionally noisy — churners skew toward complaints but many exceptions exist.

**Label definition:** For each customer at snapshot date T, label = 1 if the customer churns between T and T+30 days.
Features use only data strictly before T. Train/test split is **time-based** (no random split) to avoid leakage.

---

## Results

*(Populate after running experiments)*

**What we would do next:**
1. Real-time streaming via Kafka for support ticket ingestion
2. SHAP explainability reports for business teams
3. Champion vs Challenger A/B testing via MLflow aliases
4. Automated retraining trigger on detected data drift

---

## UI Screenshots

Capture the frontend, FastAPI docs, MLflow, Airflow, Prometheus, and Grafana
screens after launching the stack, and save them under `docs/screenshots/`.
Use synthetic or redacted data and ensure no credentials are visible.

---

## Security

- IAM role on EC2 (no long-lived access keys in code or environment)
- Security group: all ports restricted to your IP only
- Secrets in `.env` on server only, never committed
- Budget alarm set at $10/month

---

## License

MIT
