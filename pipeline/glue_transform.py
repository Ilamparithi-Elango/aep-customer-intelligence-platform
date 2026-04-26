"""
glue_transform.py
-----------------
PySpark transformation that joins the three raw AEP data sources into a
single unified data product and writes it as snappy-compressed Parquet to S3.

Join strategy:
  chatbot_interactions  LEFT JOIN  conversation_metadata  ON contact_id, session_id
  ↳ result              LEFT JOIN  clickstream_events     ON contact_id, session_id
  (left join preserves all chatbot records even if metadata / clickstream is missing)

Enrichments added:
  - date partitions (year, month, day)
  - resolution_tier bucket (HIGH / MEDIUM / LOW)
  - is_self_served flag
  - session_quality_score composite KPI

Output: s3://{S3_BUCKET}/processed/unified_interactions/year=…/month=…/day=…/*.parquet

Usage:
  spark-submit pipeline/glue_transform.py [--local]

  --local  reads from local data/raw/ instead of S3 (for dev/test)
"""

import argparse
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "aep-customer-intelligence-platform")
LOCAL_RAW  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("glue_transform")


# ── Spark Session ─────────────────────────────────────────────────────────────

def get_spark(app_name: str = "AEP-GlueTransform") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
    )
    return builder.getOrCreate()


# ── Load ──────────────────────────────────────────────────────────────────────

def load_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("nullValue", "")
        .csv(path)
    )


def load_sources(spark: SparkSession, local: bool = False) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Return (df_chatbot, df_click, df_meta) from S3 or local disk."""
    if local:
        base = LOCAL_RAW
        chatbot_path      = os.path.join(base, "chatbot_interactions.csv")
        clickstream_path  = os.path.join(base, "clickstream_events.csv")
        metadata_path     = os.path.join(base, "conversation_metadata.csv")
    else:
        base = f"s3a://{S3_BUCKET}/raw"
        chatbot_path      = f"{base}/chatbot_interactions.csv"
        clickstream_path  = f"{base}/clickstream_events.csv"
        metadata_path     = f"{base}/conversation_metadata.csv"

    log.info("Loading chatbot interactions …")
    df_chatbot = load_csv(spark, chatbot_path)

    log.info("Loading clickstream events …")
    df_click = load_csv(spark, clickstream_path)

    log.info("Loading conversation metadata …")
    df_meta = load_csv(spark, metadata_path)

    return df_chatbot, df_click, df_meta


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_chatbot(df: DataFrame) -> DataFrame:
    """Cast types, drop nulls on primary keys, normalise strings."""
    return (
        df
        .withColumn("confidence_score", F.col("confidence_score").cast(DoubleType()))
        .withColumn("thumbs_up",         F.col("thumbs_up").cast(IntegerType()))
        .withColumn("thumbs_down",       F.col("thumbs_down").cast(IntegerType()))
        .withColumn("escalated_to_agent",F.col("escalated_to_agent").cast(IntegerType()))
        .withColumn("handle_seconds_dur",F.col("handle_seconds_dur").cast(IntegerType()))
        .withColumn("derived_resolution_score", F.col("derived_resolution_score").cast(DoubleType()))
        .withColumn("business_unit",     F.trim(F.col("business_unit")))
        .withColumn("application_channel", F.trim(F.col("application_channel")))
        .dropna(subset=["contact_id", "session_id"])
    )


def clean_clickstream(df: DataFrame) -> DataFrame:
    """Aggregate clickstream to session level for the join."""
    return (
        df
        .withColumn("time_on_page_seconds", F.col("time_on_page_seconds").cast(IntegerType()))
        .dropna(subset=["contact_id", "session_id"])
        .groupBy("contact_id", "session_id")
        .agg(
            F.count("*").alias("click_event_count"),
            F.sum("is_bounce").cast(IntegerType()).alias("bounce_events"),
            F.avg("scroll_depth_pct").alias("avg_scroll_depth_pct"),
            F.sum("time_on_page_seconds").alias("total_time_on_page_seconds"),
            F.countDistinct("page_name").alias("unique_pages_visited"),
            F.countDistinct("event_type").alias("unique_event_types"),
        )
    )


def clean_metadata(df: DataFrame) -> DataFrame:
    """Cast and normalise conversation metadata."""
    return (
        df
        .withColumn("confidence_score",       F.col("confidence_score").cast(DoubleType()))
        .withColumn("num_turns",              F.col("num_turns").cast(IntegerType()))
        .withColumn("contained",              F.col("contained").cast(IntegerType()))
        .withColumn("escalated_to_agent",     F.col("escalated_to_agent").cast(IntegerType()))
        .withColumn("session_duration_seconds", F.col("session_duration_seconds").cast(IntegerType()))
        .withColumn("derived_resolution_score", F.col("derived_resolution_score").cast(DoubleType()))
        .withColumnRenamed("confidence_score",       "meta_confidence_score")
        .withColumnRenamed("derived_resolution_score", "meta_resolution_score")
        .withColumnRenamed("escalated_to_agent",     "meta_escalated")
        .withColumnRenamed("business_unit",           "meta_business_unit")
        .withColumnRenamed("application_channel",     "meta_channel")
        .withColumnRenamed("topic_name",              "meta_topic_name")
        .dropna(subset=["contact_id", "session_id"])
    )


# ── Join ──────────────────────────────────────────────────────────────────────

def join_sources(
    df_chatbot: DataFrame,
    df_click_agg: DataFrame,
    df_meta: DataFrame,
) -> DataFrame:
    """
    Unified join:
      chatbot ← LEFT JOIN → metadata   (on contact_id + session_id)
      result  ← LEFT JOIN → clickstream (on contact_id + session_id)
    """
    joined = (
        df_chatbot
        .join(df_meta,         on=["contact_id", "session_id"], how="left")
        .join(df_click_agg,    on=["contact_id", "session_id"], how="left")
    )
    return joined


# ── Enrich ────────────────────────────────────────────────────────────────────

def enrich(df: DataFrame) -> DataFrame:
    """Add derived columns and date partitions."""

    # Coalesce confidence from chatbot + metadata
    df = df.withColumn(
        "effective_confidence_score",
        F.coalesce(F.col("confidence_score"), F.col("meta_confidence_score")),
    )

    # Resolution tier
    df = df.withColumn(
        "resolution_tier",
        F.when(F.col("derived_resolution_score") >= 0.7, "HIGH")
         .when(F.col("derived_resolution_score") >= 0.4, "MEDIUM")
         .otherwise("LOW"),
    )

    # Self-served: not escalated AND contained (uses meta_escalated if available)
    df = df.withColumn(
        "is_self_served",
        (
            (F.coalesce(F.col("escalated_to_agent"), F.lit(0)) == 0)
            & (F.coalesce(F.col("meta_escalated"), F.lit(0)) == 0)
            & (F.coalesce(F.col("contained"), F.lit(1)) == 1)
        ).cast(IntegerType()),
    )

    # Session quality score: weighted composite
    # Null confidence/resolution default to 0.0 (not 0.5) so missing data
    # doesn't artificially inflate scores
    df = df.withColumn(
        "session_quality_score",
        F.round(
            F.coalesce(F.col("effective_confidence_score"), F.lit(0.0)) * 0.35
            + F.coalesce(F.col("derived_resolution_score"), F.lit(0.0)) * 0.35
            + (1.0 - F.coalesce(F.col("escalated_to_agent"), F.lit(0)).cast(DoubleType())) * 0.15
            + F.coalesce(F.col("thumbs_up"), F.lit(0)).cast(DoubleType()) * 0.15,
            4,
        ),
    )

    # Date partitions from chatbot created_at
    df = df.withColumn("created_at_ts", F.to_timestamp("created_at"))
    df = df.withColumn("year",  F.year("created_at_ts").cast("string"))
    df = df.withColumn("month", F.lpad(F.month("created_at_ts").cast("string"), 2, "0"))
    df = df.withColumn("day",   F.lpad(F.dayofmonth("created_at_ts").cast("string"), 2, "0"))

    # Pipeline audit columns
    run_ts = datetime.utcnow().isoformat()
    df = df.withColumn("pipeline_run_at", F.lit(run_ts))

    return df


# ── Write ─────────────────────────────────────────────────────────────────────

def write_parquet(df: DataFrame, output_path: str) -> None:
    """Write the unified DataFrame as partitioned Parquet."""
    log.info("Writing unified data to: %s", output_path)
    (
        df.repartition("year", "month", "day")
          .write
          .mode("overwrite")
          .partitionBy("year", "month", "day")
          .parquet(output_path)
    )
    log.info("Write complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AEP Glue-style PySpark transform.")
    parser.add_argument("--local", action="store_true", help="Read from local data/raw/.")
    args = parser.parse_args()

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Load
    df_chatbot_raw, df_click_raw, df_meta_raw = load_sources(spark, local=args.local)
    log.info("Loaded rows – chatbot: %d, clickstream: %d, metadata: %d",
             df_chatbot_raw.count(), df_click_raw.count(), df_meta_raw.count())

    # Clean
    df_chatbot  = clean_chatbot(df_chatbot_raw)
    df_click    = clean_clickstream(df_click_raw)
    df_meta     = clean_metadata(df_meta_raw)

    # Join
    df_joined = join_sources(df_chatbot, df_click, df_meta)

    # Enrich
    df_unified = enrich(df_joined)

    log.info("Unified schema: %s", df_unified.columns)
    log.info("Unified row count: %d", df_unified.count())

    # Write
    if args.local:
        import tempfile
        out_path = os.path.join(tempfile.gettempdir(), "aep_unified_interactions")
    else:
        out_path = f"s3a://{S3_BUCKET}/processed/unified_interactions"

    write_parquet(df_unified, out_path)
    log.info("Transform finished. Output: %s", out_path)


if __name__ == "__main__":
    main()
