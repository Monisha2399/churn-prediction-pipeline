#!/usr/bin/env bash
# Usage (from the repo root, after bootstrap + re-login):   bash scripts/deploy.sh <your-s3-bucket> [region]
# Builds everything, generates data -> S3, starts all services, runs the Airflow DAG once.
set -euo pipefail
BUCKET="${1:?usage: deploy.sh <s3-bucket> [region]}"
REGION="${2:-ap-south-1}"
cd "$(dirname "$0")/.."

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

say "0/8 pre-flight checks"
docker ps >/dev/null || { echo "docker needs the group refresh: log out and SSH back in"; exit 1; }
aws s3 ls "s3://$BUCKET" >/dev/null || { echo "Cannot read s3://$BUCKET - check bucket name and that the EC2 IAM role is attached"; exit 1; }
# containers reach the instance role through the metadata service; needs hop limit >= 2
code=$(docker run --rm curlimages/curl -s -m 5 -o /dev/null -w '%{http_code}' -X PUT \
  http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
if [ "$code" != "200" ]; then
  IID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-id)
  echo "Containers cannot reach the IAM role (metadata hop limit = 1). From your LAPTOP (admin user) run:"
  echo "  aws ec2 modify-instance-metadata-options --instance-id $IID --http-put-response-hop-limit 2 --http-endpoint enabled --region $REGION"
  echo "then re-run this script."; exit 1
fi
echo "OK: docker, S3 access, IAM role inside containers"

say "1/8 .env (secrets generated once, stays on this server)"
if [ ! -f .env ]; then
  rnd() { openssl rand -hex 12; }
  cat > .env <<ENV
S3_BUCKET=$BUCKET
AWS_DEFAULT_REGION=$REGION
POSTGRES_DB=churndb
POSTGRES_USER=churnuser
POSTGRES_PASSWORD=$(rnd)
AIRFLOW_UID=$(id -u)
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=$(rnd)
AIRFLOW__CORE__FERNET_KEY=$(python3 -c 'import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())')
AIRFLOW__WEBSERVER__SECRET_KEY=$(openssl rand -hex 32)
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=$(rnd)
RANDOM_SEED=42
DATA_SNAPSHOT=2026-11-01
CHAMPION_MODEL_NAME=churn_classifier
MODEL_NAME=champion
ENV
  chmod 600 .env
fi
mkdir -p data/raw/crm artifacts airflow_logs
if [ "${RESET:-0}" = "1" ]; then
  say "RESET=1: removing any old containers and the old Postgres volume"
  docker compose down -v --remove-orphans || true
fi

say "2/8 Telco CRM dataset"
CRM=data/raw/crm/WA_Fn-UseC_-Telco-Customer-Churn.csv
if [ ! -s "$CRM" ]; then
  curl -fsSL -o "$CRM" "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" || true
fi
if [ ! -s "$CRM" ] || ! head -1 "$CRM" | grep -qi customerID; then
  rm -f "$CRM"
  echo "Could not download the Telco CSV. Download it from Kaggle (blastchar/telco-customer-churn) on your laptop and:"
  echo "  scp -i churn-key.pem WA_Fn-UseC_-Telco-Customer-Churn.csv ubuntu@<ec2-ip>:$(pwd)/$CRM"
  exit 1
fi
wc -l "$CRM"

say "3/8 build images (first build ~10-15 min: torch, transformers, airflow venv)"
docker compose build

say "4/8 start postgres + mlflow"
docker compose up -d postgres mlflow
until [ "$(docker inspect -f '{{.State.Health.Status}}' churn-mlflow 2>/dev/null)" = healthy ]; do sleep 5; done

say "5/8 generate synthetic data and upload to s3://$BUCKET/raw/"
docker compose run --rm -T --no-deps --entrypoint /opt/venv/bin/python airflow scripts/generate_data.py

say "6/8 start remaining services"
docker compose up -d
echo "waiting for Airflow to register the DAG ..."
for i in $(seq 1 60); do
  docker compose exec -T airflow airflow dags list 2>/dev/null | grep -q churn_pipeline && break; sleep 5
done
docker compose exec -T airflow airflow dags unpause churn_pipeline

say "7/8 trigger the pipeline (ingest -> ... -> batch_score)"
RUN_ID="manual__$(date +%Y%m%dT%H%M%S)"
docker compose exec -T airflow airflow dags trigger churn_pipeline --run-id "$RUN_ID"
while true; do
  STATE=$(docker compose exec -T airflow airflow dags state churn_pipeline "$RUN_ID" 2>/dev/null | tail -1 | tr -d '\r')
  echo "  $(date +%T)  DAG run: ${STATE:-unknown}"
  case "$STATE" in success) break;; failed) echo "Run failed - see logs: docker compose exec airflow ls /opt/airflow/logs"; exit 1;; esac
  sleep 30
done

say "8/8 reload API with the new champion and smoke-test it"
docker compose restart api
sleep 15
curl -s localhost:8000/health; echo
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"customer":{"customer_id":"demo-high-risk","tenure_months":2,"monthly_charges":95,"contract_encoded":0,"logins_30d":3,"session_min_30d":20,"login_trend":-0.8,"failed_pay_30d":2,"days_since_login":21,"ticket_count_30d":4,"open_tickets":2,"escalations_30d":1,"sentiment_mean":-0.7,"sentiment_min":-0.9}}'; echo
IP=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/public-ipv4 || echo '<ec2-ip>')
cat <<MSG

All done. Open (security group must allow your IP on these ports):
  API docs   http://$IP:8000/docs        MLflow   http://$IP:5000
  Airflow    http://$IP:8080             Grafana  http://$IP:3000
  Frontend   http://$IP:3001
Logins are in .env (AIRFLOW_ADMIN_PASSWORD, GF_SECURITY_ADMIN_PASSWORD).
Remember: stop the instance when you finish for the day.
MSG
