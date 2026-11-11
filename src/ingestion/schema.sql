-- ─────────────────────────────────────────────────────────────────
-- Customer Churn Prediction — Database Schema
-- PostgreSQL 16
-- Run: psql -U churnuser -d churndb -f schema.sql
-- ─────────────────────────────────────────────────────────────────

-- ── Customers (CRM) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id        VARCHAR(20) PRIMARY KEY,
    signup_date        DATE        NOT NULL,
    churn_date         DATE,                          -- NULL = still active
    gender             VARCHAR(10),
    senior_citizen     SMALLINT    DEFAULT 0,
    partner            BOOLEAN,
    dependents         BOOLEAN,
    tenure_months      INTEGER,
    phone_service      BOOLEAN,
    multiple_lines     VARCHAR(20),
    internet_service   VARCHAR(20),
    online_security    VARCHAR(20),
    online_backup      VARCHAR(20),
    device_protection  VARCHAR(20),
    tech_support       VARCHAR(20),
    streaming_tv       VARCHAR(20),
    streaming_movies   VARCHAR(20),
    contract_type      VARCHAR(20),                   -- Month-to-month / One year / Two year
    paperless_billing  BOOLEAN,
    payment_method     VARCHAR(40),
    monthly_charges    NUMERIC(8, 2),
    total_charges      NUMERIC(10, 2),
    churn_label        SMALLINT    DEFAULT 0,         -- historical churn flag
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customers_churn_date ON customers (churn_date);
CREATE INDEX IF NOT EXISTS idx_customers_contract   ON customers (contract_type);

-- ── Support Tickets ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id          SERIAL      PRIMARY KEY,
    customer_id        VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    created_at         TIMESTAMPTZ NOT NULL,
    resolved_at        TIMESTAMPTZ,
    category           VARCHAR(40),                   -- billing / technical / cancellation / general
    priority           VARCHAR(10),                   -- low / medium / high / critical
    status             VARCHAR(15) DEFAULT 'open',    -- open / resolved / escalated
    subject            TEXT,
    body               TEXT        NOT NULL,
    sentiment_score    NUMERIC(5, 4),                 -- populated by feature pipeline
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tickets_customer    ON tickets (customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at  ON tickets (created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_category    ON tickets (category);

-- ── Behavioural / Usage Events ────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    event_id           BIGSERIAL   PRIMARY KEY,
    customer_id        VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    event_date         DATE        NOT NULL,
    logins             INTEGER     DEFAULT 0,
    session_minutes    NUMERIC(8, 2) DEFAULT 0,
    features_used      INTEGER     DEFAULT 0,
    failed_payments    INTEGER     DEFAULT 0,
    support_calls      INTEGER     DEFAULT 0,
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_customer   ON events (customer_id);
CREATE INDEX IF NOT EXISTS idx_events_event_date ON events (event_date);

-- ── Feature Store (snapshot features) ────────────────────────────
CREATE TABLE IF NOT EXISTS features (
    feature_id         BIGSERIAL   PRIMARY KEY,
    customer_id        VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    snapshot_date      DATE        NOT NULL,
    label              SMALLINT    NOT NULL,           -- 1 = churns within 30 days
    -- CRM features
    tenure_months      NUMERIC(6, 2),
    monthly_charges    NUMERIC(8, 2),
    contract_encoded   SMALLINT,
    senior_citizen     SMALLINT,
    -- Behavioural features
    logins_30d         NUMERIC(8, 2),
    session_min_30d    NUMERIC(10, 2),
    login_trend        NUMERIC(8, 4),
    failed_pay_30d     INTEGER,
    days_since_login   INTEGER,
    -- Ticket features
    ticket_count_30d   INTEGER,
    ticket_count_60d   INTEGER,
    ticket_count_90d   INTEGER,
    open_tickets       INTEGER,
    escalations_30d    INTEGER,
    days_since_ticket  INTEGER,
    -- NLP aggregates
    sentiment_mean     NUMERIC(6, 4),
    sentiment_min      NUMERIC(6, 4),
    sentiment_trend    NUMERIC(8, 4),
    -- Embedding vector stored as JSON array (small model = 384 dims)
    -- For large-scale use, store in Parquet/S3 instead
    bert_embedding     TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (customer_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_features_snapshot ON features (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_features_label    ON features (label);

-- ── Predictions (batch scores written by Airflow) ─────────────────
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id      BIGSERIAL   PRIMARY KEY,
    customer_id        VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    snapshot_date      DATE        NOT NULL,
    churn_probability  NUMERIC(6, 4) NOT NULL,
    churn_label        SMALLINT    NOT NULL,           -- 0 or 1 at threshold
    model_version      VARCHAR(40),
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer  ON predictions (customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_snapshot  ON predictions (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_predictions_label     ON predictions (churn_label);
