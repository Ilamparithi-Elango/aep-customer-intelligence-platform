"""
metrics_tool.py
---------------
LangChain tool that queries aggregated KPI metrics from the processed
AEP Customer Intelligence dataset.

Metrics available:
  - Containment rate       (% of sessions self-served without agent escalation)
  - Escalation rate        (% of sessions escalated to a human agent)
  - Thumbs-up rate         (positive feedback ratio among rated turns)
  - Thumbs-down rate       (negative feedback ratio among rated turns)
  - Average confidence score
  - Average resolution score
  - Average handle time

Data is loaded from the local unified CSV (or Parquet if available) and
aggregated at query time.  For production, this would query Athena or
a pre-aggregated DynamoDB table.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Type

import pandas as pd
from dotenv import load_dotenv
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

log = logging.getLogger(__name__)

# Default path to processed data (CSV fallback for local dev)
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "chatbot_interactions.csv"
)
METADATA_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "conversation_metadata.csv"
)


# ── Input Schema ──────────────────────────────────────────────────────────────

class MetricsInput(BaseModel):
    business_unit: Optional[str] = Field(
        default=None,
        description=(
            "Filter by business unit. One of: Retail, Financial Services, "
            "Healthcare, Technology, Travel. Leave blank for all units."
        ),
    )
    application_channel: Optional[str] = Field(
        default=None,
        description=(
            "Filter by channel. One of: Web, Mobile App, Kiosk, IVR, Partner API. "
            "Leave blank for all channels."
        ),
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date filter in YYYY-MM-DD format (inclusive).",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format (inclusive).",
    )


# ── Data Loader ───────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chatbot interactions and metadata. Returns (df_chatbot, df_meta)."""
    df_chatbot = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["created_at"])
    df_meta    = pd.read_csv(METADATA_DATA_PATH, parse_dates=["session_start"])
    return df_chatbot, df_meta


def _apply_filters(
    df: pd.DataFrame,
    business_unit: Optional[str],
    application_channel: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    date_col: str,
) -> pd.DataFrame:
    """Apply dimension and date filters to a DataFrame."""
    if business_unit:
        df = df[df["business_unit"].str.lower() == business_unit.lower()]
    if application_channel:
        df = df[df["application_channel"].str.lower() == application_channel.lower()]
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    return df


# ── Metric Computations ───────────────────────────────────────────────────────

def compute_metrics(
    business_unit: Optional[str] = None,
    application_channel: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Compute and return a dictionary of KPI metrics."""
    df_chatbot, df_meta = _load_data()

    df_chatbot = _apply_filters(
        df_chatbot, business_unit, application_channel,
        start_date, end_date, date_col="created_at",
    )
    df_meta = _apply_filters(
        df_meta, business_unit, application_channel,
        start_date, end_date, date_col="session_start",
    )

    if df_chatbot.empty and df_meta.empty:
        return {"error": "No data found for the specified filters."}

    metrics: dict = {}

    # ── Chatbot-level metrics ─────────────────────────────────────────────────
    total_turns = len(df_chatbot)
    metrics["total_chatbot_turns"] = total_turns

    if total_turns > 0:
        metrics["escalation_rate_chatbot"] = round(
            df_chatbot["escalated_to_agent"].mean() * 100, 2
        )
        metrics["avg_confidence_score"] = round(
            df_chatbot["confidence_score"].mean(), 4
        )
        metrics["avg_handle_seconds"] = round(
            df_chatbot["handle_seconds_dur"].mean(), 1
        )
        metrics["avg_resolution_score"] = round(
            df_chatbot["derived_resolution_score"].mean(), 4
        )

        # Thumbs feedback (only records that have feedback)
        rated = df_chatbot[(df_chatbot["thumbs_up"] == 1) | (df_chatbot["thumbs_down"] == 1)]
        if len(rated) > 0:
            metrics["thumbs_up_rate"]   = round(rated["thumbs_up"].sum() / len(rated) * 100, 2)
            metrics["thumbs_down_rate"] = round(rated["thumbs_down"].sum() / len(rated) * 100, 2)
            metrics["feedback_volume"]  = len(rated)
        else:
            metrics["thumbs_up_rate"]   = None
            metrics["thumbs_down_rate"] = None
            metrics["feedback_volume"]  = 0

    # ── Session-level metrics (from metadata) ─────────────────────────────────
    total_sessions = len(df_meta)
    metrics["total_sessions"] = total_sessions

    if total_sessions > 0:
        metrics["containment_rate"] = round(
            df_meta["contained"].mean() * 100, 2
        )
        metrics["escalation_rate_session"] = round(
            df_meta["escalated_to_agent"].mean() * 100, 2
        )
        metrics["avg_turns_per_session"] = round(
            df_meta["num_turns"].mean(), 2
        )
        metrics["avg_session_duration_seconds"] = round(
            df_meta["session_duration_seconds"].mean(), 1
        )

        csat = df_meta["csat_score"].dropna()
        if len(csat) > 0:
            metrics["avg_csat_score"]     = round(csat.mean(), 2)
            metrics["csat_response_rate"] = round(len(csat) / total_sessions * 100, 2)
        else:
            metrics["avg_csat_score"]     = None
            metrics["csat_response_rate"] = 0.0

    # ── Applied filters (for transparency) ───────────────────────────────────
    metrics["filters_applied"] = {
        "business_unit":       business_unit or "ALL",
        "application_channel": application_channel or "ALL",
        "start_date":          start_date or "ALL",
        "end_date":            end_date or "ALL",
    }

    return metrics


# ── LangChain Tool ────────────────────────────────────────────────────────────

class MetricsTool(BaseTool):
    """
    LangChain tool to query AEP platform KPI metrics.

    Returns a JSON string containing containment rate, escalation rate,
    thumbs-up/down feedback rates, average confidence score, handle time,
    CSAT, and more — optionally filtered by business unit, channel, or date.
    """

    name: str = "aep_metrics"
    description: str = (
        "Query aggregated AEP customer intelligence KPI metrics. "
        "Returns containment rate, escalation rate, thumbs-up/down rate, "
        "average confidence score, handle time, CSAT, and session counts. "
        "Accepts optional filters: business_unit, application_channel, start_date, end_date. "
        "Input must be a JSON string with optional filter keys, or an empty string for all data."
    )
    args_schema: Type[BaseModel] = MetricsInput

    def _run(
        self,
        business_unit: Optional[str] = None,
        application_channel: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        try:
            metrics = compute_metrics(
                business_unit=business_unit,
                application_channel=application_channel,
                start_date=start_date,
                end_date=end_date,
            )
            return json.dumps(metrics, indent=2)
        except FileNotFoundError as exc:
            return json.dumps({"error": f"Data file not found: {exc}. Run generate_mock_data.py first."})
        except Exception as exc:
            log.error("MetricsTool error: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Async not supported — use _run.")


if __name__ == "__main__":
    tool = MetricsTool()
    print(tool.run({"business_unit": "Retail"}))
