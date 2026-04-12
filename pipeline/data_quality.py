"""
data_quality.py
---------------
PySpark data quality checks for the three raw AEP data sources.

Checks performed:
  1. Null / missing value rate per column  (threshold: < 5 % for critical fields)
  2. Confidence score range validation     (must be in [0.0, 1.0])
  3. Duplicate detection                  (exact-row and key-level)
  4. Referential integrity                (business_unit / application_channel enums)
  5. Schema validation                    (expected columns present)

Exit codes:
  0  – all checks passed
  1  – one or more checks failed (details logged)

Usage:
  spark-submit pipeline/data_quality.py \
      --chatbot   s3://bucket/raw/chatbot_interactions.csv \
      --clickstream s3://bucket/raw/clickstream_events.csv \
      --metadata  s3://bucket/raw/conversation_metadata.csv
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "aep-customer-intelligence-platform")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("data_quality")


# ── Schema Definitions ────────────────────────────────────────────────────────

CHATBOT_REQUIRED_COLS = [
    "contact_id", "session_id", "topic_name", "confidence_score",
    "response_type", "thumbs_up", "thumbs_down", "escalated_to_agent",
    "business_unit", "application_channel", "handle_seconds_dur",
    "derived_resolution_score", "created_at",
]

CLICKSTREAM_REQUIRED_COLS = [
    "contact_id", "session_id", "event_type", "page_name",
    "application_channel", "business_unit", "time_on_page_seconds",
    "scroll_depth_pct", "is_bounce", "event_timestamp",
]

METADATA_REQUIRED_COLS = [
    "contact_id", "session_id", "topic_name", "business_unit",
    "application_channel", "language", "num_turns", "contained",
    "escalated_to_agent", "session_duration_seconds", "confidence_score",
    "derived_resolution_score", "session_start",
]

VALID_BUSINESS_UNITS = {"Retail", "Financial Services", "Healthcare", "Technology", "Travel"}
VALID_CHANNELS = {"Web", "Mobile App", "Kiosk", "IVR", "Partner API"}

# Critical columns that must have < 5 % nulls
CRITICAL_COLS_CHATBOT = ["contact_id", "session_id", "confidence_score", "business_unit"]
CRITICAL_COLS_CLICKSTREAM = ["contact_id", "session_id", "event_type", "business_unit"]
CRITICAL_COLS_METADATA = ["contact_id", "session_id", "confidence_score", "contained"]

NULL_THRESHOLD = 0.05  # 5 %


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    source: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)
        log.error("[%s] FAIL: %s", self.source, msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("[%s] WARN: %s", self.source, msg)

    def ok(self, msg: str) -> None:
        log.info("[%s] OK  : %s", self.source, msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"[{self.source}] {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN : {w}")
        return "\n".join(lines)


# ── Check Functions ───────────────────────────────────────────────────────────

def check_schema(df: DataFrame, required_cols: list[str], report: QualityReport) -> None:
    """Verify all expected columns are present."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        report.fail(f"Missing columns: {sorted(missing)}")
    else:
        report.ok(f"Schema valid – {len(required_cols)} required columns present.")


def check_nulls(
    df: DataFrame,
    critical_cols: list[str],
    report: QualityReport,
    threshold: float = NULL_THRESHOLD,
) -> None:
    """Flag columns with null rate above threshold."""
    total = df.count()
    if total == 0:
        report.fail("DataFrame is empty.")
        return

    for col in critical_cols:
        if col not in df.columns:
            continue  # already reported in schema check
        null_count = df.filter(F.col(col).isNull()).count()
        null_rate = null_count / total
        if null_rate > threshold:
            report.fail(
                f"Column '{col}' null rate {null_rate:.1%} exceeds threshold {threshold:.0%}"
            )
        else:
            report.ok(f"Column '{col}' null rate {null_rate:.1%} ≤ {threshold:.0%}")


def check_confidence_scores(df: DataFrame, report: QualityReport) -> None:
    """Validate confidence_score is in [0.0, 1.0]."""
    if "confidence_score" not in df.columns:
        return

    df_typed = df.withColumn("confidence_score", F.col("confidence_score").cast(DoubleType()))
    invalid = df_typed.filter(
        (F.col("confidence_score") < 0.0) | (F.col("confidence_score") > 1.0)
    ).count()

    if invalid > 0:
        report.fail(f"confidence_score: {invalid} values outside [0.0, 1.0]")
    else:
        report.ok("confidence_score: all values in [0.0, 1.0]")

    # Warn if mean is suspiciously low (possible model degradation)
    mean_score = df_typed.select(F.mean("confidence_score")).collect()[0][0]
    if mean_score is not None and mean_score < 0.4:
        report.warn(f"Mean confidence_score is low: {mean_score:.4f} — possible model degradation")
    else:
        report.ok(f"Mean confidence_score: {mean_score:.4f}")


def check_duplicates(
    df: DataFrame,
    key_cols: list[str],
    report: QualityReport,
) -> None:
    """Detect exact duplicate rows and key-level duplicates."""
    total = df.count()

    # Exact row duplicates
    distinct_count = df.distinct().count()
    exact_dupes = total - distinct_count
    if exact_dupes > 0:
        report.warn(f"Exact duplicate rows: {exact_dupes:,} ({exact_dupes / total:.2%})")
    else:
        report.ok("No exact duplicate rows.")

    # Key-level duplicates
    available_keys = [c for c in key_cols if c in df.columns]
    if available_keys:
        key_dupes = (
            df.groupBy(available_keys)
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if key_dupes > 0:
            report.warn(
                f"Key-level duplicates on {available_keys}: {key_dupes:,} groups have >1 row"
            )
        else:
            report.ok(f"No key-level duplicates on {available_keys}.")


def check_enum_values(
    df: DataFrame,
    col: str,
    valid_values: set[str],
    report: QualityReport,
) -> None:
    """Check that a categorical column only contains expected values."""
    if col not in df.columns:
        return

    invalid_count = df.filter(~F.col(col).isin(list(valid_values))).count()
    if invalid_count > 0:
        report.warn(
            f"Column '{col}': {invalid_count:,} rows with unexpected values "
            f"(valid: {sorted(valid_values)})"
        )
    else:
        report.ok(f"Column '{col}': all values within allowed set.")


def check_numeric_range(
    df: DataFrame,
    col: str,
    min_val: Optional[float],
    max_val: Optional[float],
    report: QualityReport,
) -> None:
    """Check a numeric column stays within expected bounds."""
    if col not in df.columns:
        return

    filters = []
    if min_val is not None:
        filters.append(F.col(col) < min_val)
    if max_val is not None:
        filters.append(F.col(col) > max_val)

    if not filters:
        return

    combined = filters[0]
    for f in filters[1:]:
        combined = combined | f

    invalid = df.filter(combined).count()
    if invalid > 0:
        report.fail(
            f"Column '{col}': {invalid:,} values outside range "
            f"[{min_val}, {max_val}]"
        )
    else:
        report.ok(f"Column '{col}': all values within [{min_val}, {max_val}]")


# ── Per-Source Quality Runners ────────────────────────────────────────────────

def run_chatbot_checks(df: DataFrame) -> QualityReport:
    report = QualityReport(source="chatbot_interactions")
    check_schema(df, CHATBOT_REQUIRED_COLS, report)
    check_nulls(df, CRITICAL_COLS_CHATBOT, report)
    check_confidence_scores(df, report)
    check_duplicates(df, key_cols=["contact_id", "session_id"], report=report)
    check_enum_values(df, "business_unit", VALID_BUSINESS_UNITS, report)
    check_enum_values(df, "application_channel", VALID_CHANNELS, report)
    check_numeric_range(df, "handle_seconds_dur", 0, None, report)
    check_numeric_range(df, "derived_resolution_score", 0.0, 1.0, report)
    return report


def run_clickstream_checks(df: DataFrame) -> QualityReport:
    report = QualityReport(source="clickstream_events")
    check_schema(df, CLICKSTREAM_REQUIRED_COLS, report)
    check_nulls(df, CRITICAL_COLS_CLICKSTREAM, report)
    check_duplicates(df, key_cols=["contact_id", "session_id", "event_type"], report=report)
    check_enum_values(df, "business_unit", VALID_BUSINESS_UNITS, report)
    check_enum_values(df, "application_channel", VALID_CHANNELS, report)
    check_numeric_range(df, "scroll_depth_pct", 0.0, 100.0, report)
    check_numeric_range(df, "time_on_page_seconds", 0, None, report)
    return report


def run_metadata_checks(df: DataFrame) -> QualityReport:
    report = QualityReport(source="conversation_metadata")
    check_schema(df, METADATA_REQUIRED_COLS, report)
    check_nulls(df, CRITICAL_COLS_METADATA, report)
    check_confidence_scores(df, report)
    check_duplicates(df, key_cols=["contact_id", "session_id"], report=report)
    check_enum_values(df, "business_unit", VALID_BUSINESS_UNITS, report)
    check_enum_values(df, "application_channel", VALID_CHANNELS, report)
    check_numeric_range(df, "num_turns", 1, None, report)
    check_numeric_range(df, "session_duration_seconds", 0, None, report)
    return report


# ── Spark Session ─────────────────────────────────────────────────────────────

def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("AEP-DataQuality")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
        .getOrCreate()
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQ checks on AEP raw data.")
    parser.add_argument("--chatbot",      default=f"s3a://{S3_BUCKET}/raw/chatbot_interactions.csv")
    parser.add_argument("--clickstream",  default=f"s3a://{S3_BUCKET}/raw/clickstream_events.csv")
    parser.add_argument("--metadata",     default=f"s3a://{S3_BUCKET}/raw/conversation_metadata.csv")
    args = parser.parse_args()

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info("Loading raw data sources …")
    df_chatbot = spark.read.option("header", "true").option("inferSchema", "true").csv(args.chatbot)
    df_click   = spark.read.option("header", "true").option("inferSchema", "true").csv(args.clickstream)
    df_meta    = spark.read.option("header", "true").option("inferSchema", "true").csv(args.metadata)

    log.info("Row counts – chatbot: %d, clickstream: %d, metadata: %d",
             df_chatbot.count(), df_click.count(), df_meta.count())

    reports = [
        run_chatbot_checks(df_chatbot),
        run_clickstream_checks(df_click),
        run_metadata_checks(df_meta),
    ]

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    for r in reports:
        print(r.summary())
    print("=" * 60)

    all_passed = all(r.passed for r in reports)
    if all_passed:
        log.info("All data quality checks passed.")
    else:
        log.error("One or more data quality checks FAILED. See report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
