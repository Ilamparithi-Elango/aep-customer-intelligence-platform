"""
batch_processor.py
------------------
OpenAI batch processor for the AEP Customer Intelligence Platform.

Processes conversation records from the unified Parquet dataset using
concurrent OpenAI API calls for:
  - Conversation summarisation     (SummarizationTemplate v2)
  - Topic classification           (TopicClassificationTemplate v1)
  - Sentiment analysis             (SentimentTemplate v1)

Concurrency model:
  ThreadPoolExecutor with configurable max_workers (default: 10).
  Each worker calls the OpenAI chat completions endpoint independently.
  Results are written to S3 as JSONL (one record per line).

Cost tracking:
  Token usage is accumulated and logged at the end of each batch.

Usage:
  python llm/batch_processor.py \
    --input  s3a://bucket/processed/unified_interactions \
    --output s3a://bucket/results/llm_enriched \
    [--limit 500] [--workers 10] [--local]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from llm.prompt_templates import TemplateKey, get_template

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AWS_REGION     = os.getenv("AWS_REGION",     "us-east-1")
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET",  "aep-customer-intelligence-platform")

LOCAL_PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw"
)

DEFAULT_MODEL   = "gpt-4o-mini"   # cost-efficient for bulk classification
DEFAULT_WORKERS = 10
MAX_RETRIES     = 3
RETRY_SLEEP_S   = 2.0

TOPIC_LIST = [
    "Account Management", "Billing Inquiry", "Product Support",
    "Order Status", "Returns & Refunds", "Technical Troubleshooting",
    "Subscription Management", "Password Reset", "Loyalty Program",
    "General Inquiry",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("batch_processor")


# ── Token / Cost Tracker ──────────────────────────────────────────────────────

@dataclass
class UsageTracker:
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_cost_usd:    float = 0.0

    # gpt-4o-mini pricing (per 1M tokens) as of 2025
    _INPUT_PRICE_PER_M:  float = field(default=0.15,  init=False, repr=False)
    _OUTPUT_PRICE_PER_M: float = field(default=0.60,  init=False, repr=False)

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.prompt_tokens     += getattr(usage, "prompt_tokens",     0)
        self.completion_tokens += getattr(usage, "completion_tokens", 0)
        self.total_cost_usd    += (
            self.prompt_tokens     / 1_000_000 * self._INPUT_PRICE_PER_M
            + self.completion_tokens / 1_000_000 * self._OUTPUT_PRICE_PER_M
        )

    def summary(self) -> str:
        return (
            f"Tokens used — prompt: {self.prompt_tokens:,}, "
            f"completion: {self.completion_tokens:,}  |  "
            f"Estimated cost: ${self.total_cost_usd:.4f}"
        )


# ── Single-Record Processor ───────────────────────────────────────────────────

def build_synthetic_transcript(row: dict) -> str:
    """
    Construct a minimal synthetic transcript from structured row fields.
    Topic is intentionally excluded from the transcript text so the LLM
    classifies it independently — avoiding circular classification.
    In production this would reference actual raw conversation text.
    """
    channel   = row.get("application_channel", "Web")
    turns     = row.get("num_turns", 3)
    response  = row.get("response_type", "FAQ")
    conf      = row.get("confidence_score", 0.7)
    escalated = row.get("escalated_to_agent", 0)
    thumbs_up = row.get("thumbs_up", 0)
    thumbs_dn = row.get("thumbs_down", 0)

    outcome = "escalated to a human agent" if escalated else "resolved by the bot"
    feedback = (
        "Customer gave a thumbs up at the end." if thumbs_up
        else "Customer gave a thumbs down." if thumbs_dn
        else "No explicit feedback given."
    )

    return (
        f"Channel: {channel}\n"
        f"Response type: {response}\n"
        f"Bot confidence: {conf:.2f}\n"
        f"User: Hi, I need some help with an issue on my account.\n"
        f"Bot:  Of course, I can assist you with that. Could you provide more details?\n"
        f"[... {max(0, int(turns) - 2)} additional exchange(s) ...]\n"
        f"Outcome: Session {outcome}.\n"
        f"{feedback}"
    )


def call_openai_with_retry(
    client: OpenAI,
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    tracker: UsageTracker,
) -> str | None:
    """Call the OpenAI chat completions endpoint with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            tracker.add(resp.usage)
            return resp.choices[0].message.content.strip()

        except RateLimitError:
            wait = RETRY_SLEEP_S * (2 ** (attempt - 1))
            log.warning("Rate limit hit, retry %d/%d in %.1fs …", attempt, MAX_RETRIES, wait)
            time.sleep(wait)

        except APIError as exc:
            log.error("OpenAI API error on attempt %d: %s", attempt, exc)
            if attempt == MAX_RETRIES:
                return None

    return None


def process_record(
    row: dict,
    client: OpenAI,
    model: str,
    tracker: UsageTracker,
) -> dict:
    """
    Run summarisation, topic classification, and sentiment for one record.
    Returns the original row dict enriched with LLM outputs.
    """
    transcript = build_synthetic_transcript(row)

    result = {
        **row,
        "llm_summary":        None,
        "llm_core_issue":     None,
        "llm_outcome":        None,
        "llm_sentiment":      None,
        "llm_sentiment_score": None,
        "llm_frustration":    None,
        "llm_churn_risk":     None,
        "llm_topic":          None,
        "llm_topic_confidence": None,
        "llm_processing_error": None,
    }

    # 1. Summarisation (v2 → structured JSON)
    try:
        sum_tmpl = get_template(TemplateKey.SUMMARIZATION, version=2)
        sum_msgs = sum_tmpl.to_openai_messages(transcript=transcript, max_words=60)
        sum_raw  = call_openai_with_retry(client, sum_msgs, model,
                                          sum_tmpl.max_tokens, sum_tmpl.temperature, tracker)
        if sum_raw:
            sum_data = json.loads(sum_raw)
            result["llm_summary"]    = sum_data.get("summary")
            result["llm_core_issue"] = sum_data.get("core_issue")
            result["llm_outcome"]    = sum_data.get("outcome")
    except Exception as exc:
        log.debug("Summarisation failed for %s: %s", row.get("session_id"), exc)
        result["llm_processing_error"] = str(exc)

    # 2. Sentiment
    try:
        sent_tmpl = get_template(TemplateKey.SENTIMENT, version=1)
        sent_msgs = sent_tmpl.to_openai_messages(conversation=transcript)
        sent_raw  = call_openai_with_retry(client, sent_msgs, model,
                                           sent_tmpl.max_tokens, sent_tmpl.temperature, tracker)
        if sent_raw:
            sent_data = json.loads(sent_raw)
            result["llm_sentiment"]       = sent_data.get("overall_sentiment")
            result["llm_sentiment_score"] = sent_data.get("sentiment_score")
            result["llm_frustration"]     = sent_data.get("frustration_signal")
            result["llm_churn_risk"]      = sent_data.get("churn_risk")
    except Exception as exc:
        log.debug("Sentiment failed for %s: %s", row.get("session_id"), exc)

    # 3. Topic classification
    try:
        topic_tmpl = get_template(TemplateKey.TOPIC_CLASSIFICATION, version=1)
        topic_msgs = topic_tmpl.to_openai_messages(
            conversation=transcript,
            topic_list="\n".join(f"  - {t}" for t in TOPIC_LIST),
        )
        topic_raw = call_openai_with_retry(client, topic_msgs, model,
                                           topic_tmpl.max_tokens, topic_tmpl.temperature, tracker)
        if topic_raw:
            topic_data = json.loads(topic_raw)
            result["llm_topic"]            = topic_data.get("topic")
            result["llm_topic_confidence"] = topic_data.get("confidence")
    except Exception as exc:
        log.debug("Topic classification failed for %s: %s", row.get("session_id"), exc)

    return result


# ── Batch Runner ──────────────────────────────────────────────────────────────

def run_batch(
    records: list[dict],
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_WORKERS,
) -> tuple[list[dict], UsageTracker]:
    """Process a list of records concurrently. Returns (results, tracker)."""
    client  = OpenAI(api_key=OPENAI_API_KEY)
    tracker = UsageTracker()
    results: list[dict] = []

    log.info("Starting batch: %d records, model=%s, workers=%d", len(records), model, max_workers)
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, row, client, model, tracker): i
            for i, row in enumerate(records)
        }
        for done_future in as_completed(futures):
            idx = futures[done_future]
            try:
                results.append(done_future.result())
            except Exception as exc:
                log.error("Record %d failed: %s", idx, exc)

    elapsed = time.monotonic() - start
    log.info("Batch complete in %.1fs. %s", elapsed, tracker.summary())
    return results, tracker


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_parquet_sample(path: str, limit: int) -> list[dict]:
    """Load up to `limit` rows from a local Parquet directory."""
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def load_csv_sample(path: str, limit: int) -> list[dict]:
    """Fallback: load from CSV for local dev."""
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def write_results_s3(results: list[dict], s3_output_path: str) -> None:
    """Write JSONL results to S3."""
    s3    = boto3.client("s3", region_name=AWS_REGION)
    # Parse s3a://bucket/key → bucket + key
    path  = s3_output_path.replace("s3a://", "").replace("s3://", "")
    parts = path.split("/", 1)
    bucket, prefix = parts[0], parts[1] if len(parts) > 1 else ""

    jsonl_body = "\n".join(json.dumps(r, default=str) for r in results)
    key = f"{prefix}/llm_results.jsonl".lstrip("/")

    try:
        s3.put_object(Bucket=bucket, Key=key, Body=jsonl_body.encode("utf-8"))
        log.info("Results written to s3://%s/%s", bucket, key)
    except ClientError as exc:
        log.error("S3 write failed: %s", exc)
        raise


def write_results_local(results: list[dict], output_path: str) -> None:
    """Write JSONL results to a local file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    log.info("Results written locally → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AEP LLM batch processor.")
    parser.add_argument("--input",   default=f"s3a://{AWS_S3_BUCKET}/processed/unified_interactions")
    parser.add_argument("--output",  default=f"s3a://{AWS_S3_BUCKET}/results/llm_enriched")
    parser.add_argument("--limit",   type=int, default=100, help="Max records to process.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--local",   action="store_true", help="Use local CSV fallback for input.")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    # Load records
    if args.local:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "chatbot_interactions.csv"
        )
        log.info("Local mode: loading from %s", csv_path)
        records = load_csv_sample(csv_path, args.limit)
    else:
        records = load_parquet_sample(args.input, args.limit)

    log.info("Loaded %d records for processing.", len(records))

    # Run batch
    results, tracker = run_batch(records, model=args.model, max_workers=args.workers)

    # Write output
    if args.local:
        local_out = os.path.join(
            os.path.dirname(__file__), "..", "data", "results", "llm_results.jsonl"
        )
        write_results_local(results, local_out)
    else:
        write_results_s3(results, args.output)

    log.info("Done. %s", tracker.summary())


if __name__ == "__main__":
    main()
