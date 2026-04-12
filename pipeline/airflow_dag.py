"""
airflow_dag.py
--------------
Apache Airflow DAG that orchestrates the full AEP Customer Intelligence
Platform pipeline on a daily schedule.

Pipeline steps (in order):
  1. generate_mock_data   – produce fresh CSV files for the day
  2. aws_setup            – ensure S3 buckets / prefixes exist (idempotent)
  3. data_quality         – PySpark DQ checks on raw CSVs
  4. glue_transform       – PySpark join & enrich → Parquet on S3
  5. llm_batch_process    – OpenAI batch summarisation & classification
  6. notify_success       – send a Slack/email summary (stub)

This file is designed to be placed in the Airflow DAGs folder.
Dependencies: apache-airflow>=2.8, apache-airflow-providers-amazon
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

# ── Default Args ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.getenv("AEP_PROJECT_ROOT", "/opt/airflow/dags/aep-customer-intelligence-platform")
PYTHON_BIN   = os.getenv("AEP_PYTHON_BIN",   "python")
SPARK_SUBMIT = os.getenv("AEP_SPARK_SUBMIT",  "spark-submit")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET",    "aep-customer-intelligence-platform")
AWS_REGION    = os.getenv("AWS_REGION",       "us-east-1")

default_args = {
    "owner": "aep-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Python Callables ──────────────────────────────────────────────────────────

def check_env_vars(**context) -> bool:
    """
    Gate task: verify required environment variables are present.
    Returns False to short-circuit the DAG if any are missing.
    """
    required = ["AWS_S3_BUCKET", "AWS_REGION", "OPENAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")
    return True


def notify_success(**context) -> None:
    """
    Stub notification task.  Replace with SlackWebhookOperator or
    SesOperator in production.
    """
    dag_run = context.get("dag_run")
    execution_date = context.get("execution_date")
    print(
        f"[AEP Pipeline] Daily run completed successfully.\n"
        f"  DAG     : {dag_run.dag_id if dag_run else 'unknown'}\n"
        f"  Date    : {execution_date}\n"
        f"  Bucket  : s3://{AWS_S3_BUCKET}/processed/\n"
        f"  Status  : SUCCESS"
    )


# ── DAG Definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="aep_customer_intelligence_pipeline",
    description="Daily AEP Customer Intelligence Platform end-to-end pipeline",
    schedule_interval="0 2 * * *",           # 02:00 UTC daily
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["aep", "customer-intelligence", "production"],
    doc_md="""
## AEP Customer Intelligence Platform – Daily Pipeline

Runs every day at 02:00 UTC.

### Steps
| # | Task | Description |
|---|------|-------------|
| 1 | `check_env`         | Validate required env vars |
| 2 | `generate_data`     | Generate mock CSV data |
| 3 | `provision_aws`     | Idempotent S3 setup |
| 4 | `data_quality`      | PySpark DQ checks |
| 5 | `glue_transform`    | Join & enrich → Parquet |
| 6 | `llm_batch_process` | OpenAI batch LLM enrichment |
| 7 | `notify_success`    | Send completion notification |
""",
) as dag:

    # ── Task 1: Environment gate ──────────────────────────────────────────────
    check_env = ShortCircuitOperator(
        task_id="check_env",
        python_callable=check_env_vars,
        provide_context=True,
        doc_md="Verify required environment variables before proceeding.",
    )

    # ── Task 2: Generate mock data ────────────────────────────────────────────
    generate_data = BashOperator(
        task_id="generate_mock_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_BIN} data/raw/generate_mock_data.py"
        ),
        doc_md="Generate fresh mock CSV files for chatbot, clickstream, and metadata.",
    )

    # ── Task 3: Provision AWS infrastructure ──────────────────────────────────
    provision_aws = BashOperator(
        task_id="provision_aws",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_BIN} infrastructure/aws_setup.py"
        ),
        doc_md="Create S3 buckets / prefixes and upload raw CSVs (idempotent).",
    )

    # ── Task 4: Data quality ──────────────────────────────────────────────────
    data_quality = BashOperator(
        task_id="data_quality_checks",
        bash_command=(
            f"{SPARK_SUBMIT} "
            f"  --master local[*] "
            f"  {PROJECT_ROOT}/pipeline/data_quality.py "
            f"  --chatbot     s3a://{AWS_S3_BUCKET}/raw/chatbot_interactions.csv "
            f"  --clickstream s3a://{AWS_S3_BUCKET}/raw/clickstream_events.csv "
            f"  --metadata    s3a://{AWS_S3_BUCKET}/raw/conversation_metadata.csv"
        ),
        doc_md="Run PySpark null-check, range-validation, and duplicate-detection checks.",
    )

    # ── Task 5: Glue transform ────────────────────────────────────────────────
    glue_transform = BashOperator(
        task_id="glue_transform",
        bash_command=(
            f"{SPARK_SUBMIT} "
            f"  --master local[*] "
            f"  {PROJECT_ROOT}/pipeline/glue_transform.py"
        ),
        doc_md=(
            "Join the three raw sources into a unified Parquet data product "
            "and write to s3://{bucket}/processed/."
        ),
    )

    # ── Task 6: LLM batch processing ─────────────────────────────────────────
    llm_batch_process = BashOperator(
        task_id="llm_batch_process",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_BIN} llm/batch_processor.py "
            f"  --input  s3a://{AWS_S3_BUCKET}/processed/unified_interactions "
            f"  --output s3a://{AWS_S3_BUCKET}/results/llm_enriched"
        ),
        doc_md="Run OpenAI batch API calls for conversation summarisation and topic classification.",
    )

    # ── Task 7: Notify success ────────────────────────────────────────────────
    notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
        provide_context=True,
        doc_md="Send a success notification (Slack/email stub).",
    )

    # ── Dependency chain ──────────────────────────────────────────────────────
    (
        check_env
        >> generate_data
        >> provision_aws
        >> data_quality
        >> glue_transform
        >> llm_batch_process
        >> notify
    )
