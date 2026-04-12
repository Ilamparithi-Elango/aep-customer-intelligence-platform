"""
aws_setup.py
------------
Provisions S3 buckets and uploads raw mock data for the AEP Customer
Intelligence Platform.

Buckets created:
  {S3_BUCKET}/raw/         – raw CSV source files
  {S3_BUCKET}/processed/   – Parquet output from Glue transform
  {S3_BUCKET}/results/     – LLM batch results & agent artefacts
  {S3_BUCKET}/logs/        – pipeline run logs

Usage:
  python infrastructure/aws_setup.py [--dry-run]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "aep-customer-intelligence-platform")

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# S3 prefixes to pre-create (done by uploading a zero-byte placeholder)
S3_PREFIXES = ["raw/", "processed/", "results/", "logs/"]

# Raw CSV files to upload
RAW_FILES = [
    "chatbot_interactions.csv",
    "clickstream_events.csv",
    "conversation_metadata.csv",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── S3 Helpers ────────────────────────────────────────────────────────────────

def get_s3_client():
    """Return a boto3 S3 client for the configured region."""
    return boto3.client("s3", region_name=AWS_REGION)


def bucket_exists(s3, bucket_name: str) -> bool:
    """Return True if the bucket exists and is accessible."""
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            return False
        raise


def create_bucket(s3, bucket_name: str, region: str, dry_run: bool = False) -> None:
    """Create an S3 bucket with versioning and server-side encryption enabled."""
    if dry_run:
        log.info("[DRY-RUN] Would create bucket: s3://%s", bucket_name)
        return

    try:
        if region == "us-east-1":
            # us-east-1 does NOT accept a LocationConstraint
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )

        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Enabled"},
        )

        # Enable AES-256 server-side encryption by default
        s3.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        },
                        "BucketKeyEnabled": True,
                    }
                ]
            },
        )

        # Block all public access
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )

        log.info("Created bucket s3://%s (region=%s)", bucket_name, region)

    except ClientError as exc:
        log.error("Failed to create bucket %s: %s", bucket_name, exc)
        raise


def create_prefixes(s3, bucket_name: str, prefixes: list[str], dry_run: bool = False) -> None:
    """Upload zero-byte placeholder objects to simulate folder structure."""
    for prefix in prefixes:
        key = f"{prefix}.keep"
        if dry_run:
            log.info("[DRY-RUN] Would create prefix placeholder: s3://%s/%s", bucket_name, key)
            continue
        try:
            s3.put_object(Bucket=bucket_name, Key=key, Body=b"")
            log.info("Created prefix placeholder: s3://%s/%s", bucket_name, key)
        except ClientError as exc:
            log.warning("Could not create prefix %s: %s", key, exc)


def upload_raw_files(
    s3,
    bucket_name: str,
    raw_dir: Path,
    files: list[str],
    dry_run: bool = False,
) -> None:
    """Upload raw CSV files to s3://{bucket}/raw/."""
    for filename in files:
        local_path = raw_dir / filename
        s3_key = f"raw/{filename}"

        if not local_path.exists():
            log.warning(
                "Raw file not found, skipping: %s  (run generate_mock_data.py first)",
                local_path,
            )
            continue

        file_size_mb = local_path.stat().st_size / (1024 * 1024)

        if dry_run:
            log.info(
                "[DRY-RUN] Would upload %s (%.2f MB) → s3://%s/%s",
                filename, file_size_mb, bucket_name, s3_key,
            )
            continue

        try:
            s3.upload_file(
                str(local_path),
                bucket_name,
                s3_key,
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
            log.info(
                "Uploaded %s (%.2f MB) → s3://%s/%s",
                filename, file_size_mb, bucket_name, s3_key,
            )
        except (ClientError, BotoCoreError) as exc:
            log.error("Upload failed for %s: %s", filename, exc)
            raise


# ── Entry Point ───────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Main setup routine."""
    log.info("=== AEP Customer Intelligence Platform – AWS Setup ===")
    log.info("Region : %s", AWS_REGION)
    log.info("Bucket : s3://%s", S3_BUCKET)
    log.info("Dry-run: %s", dry_run)

    s3 = get_s3_client()

    # 1. Create bucket (idempotent)
    if bucket_exists(s3, S3_BUCKET):
        log.info("Bucket already exists, skipping creation.")
    else:
        create_bucket(s3, S3_BUCKET, AWS_REGION, dry_run=dry_run)

    # 2. Create prefix placeholders
    create_prefixes(s3, S3_BUCKET, S3_PREFIXES, dry_run=dry_run)

    # 3. Upload raw data files
    upload_raw_files(s3, S3_BUCKET, RAW_DATA_DIR, RAW_FILES, dry_run=dry_run)

    log.info("=== Setup complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision S3 infrastructure for the AEP Customer Intelligence Platform."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing them.",
    )
    args = parser.parse_args()

    try:
        run(dry_run=args.dry_run)
    except (ClientError, BotoCoreError) as exc:
        log.error("AWS error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
