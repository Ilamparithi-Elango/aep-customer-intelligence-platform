"""
topic_tool.py
-------------
LangChain tool that analyses trending topics and customer sentiment from
the AEP Customer Intelligence dataset.

Capabilities:
  - Top N topics by volume / escalation rate / resolution score
  - Topic-level sentiment distribution (positive / neutral / negative)
  - Week-over-week topic velocity (trending up/down)
  - Channel and business-unit breakdowns by topic

Data source: local chatbot_interactions.csv + conversation_metadata.csv
(in production, would query Athena or a pre-computed analytics table)
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

CHATBOT_PATH  = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "chatbot_interactions.csv"
)
METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "conversation_metadata.csv"
)


# ── Input Schema ──────────────────────────────────────────────────────────────

class TopicInput(BaseModel):
    top_n: int = Field(
        default=5,
        description="Number of top topics to return (default 5, max 10).",
        ge=1,
        le=10,
    )
    sort_by: str = Field(
        default="volume",
        description=(
            "Sort topics by: 'volume' (record count), 'escalation_rate', "
            "'resolution_score', or 'thumbs_up_rate'."
        ),
    )
    business_unit: Optional[str] = Field(
        default=None,
        description="Filter to a specific business unit (e.g. Retail, Healthcare).",
    )
    application_channel: Optional[str] = Field(
        default=None,
        description="Filter to a specific channel (e.g. Web, Mobile App).",
    )
    include_sentiment: bool = Field(
        default=True,
        description="Include a simple sentiment breakdown per topic.",
    )
    include_trend: bool = Field(
        default=False,
        description=(
            "If True, include a week-over-week volume trend indicator "
            "(requires ≥ 2 weeks of data in the dataset)."
        ),
    )


# ── Data Helpers ──────────────────────────────────────────────────────────────

def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_chat = pd.read_csv(CHATBOT_PATH,  parse_dates=["created_at"])
    df_meta = pd.read_csv(METADATA_PATH, parse_dates=["session_start"])
    return df_chat, df_meta


def _filter(df: pd.DataFrame, business_unit: Optional[str], channel: Optional[str]) -> pd.DataFrame:
    if business_unit:
        df = df[df["business_unit"].str.lower() == business_unit.lower()]
    if channel:
        df = df[df["application_channel"].str.lower() == channel.lower()]
    return df


def _sentiment_label(score: float) -> str:
    """Map a resolution score proxy to a sentiment label."""
    if score >= 0.65:
        return "POSITIVE"
    if score >= 0.40:
        return "NEUTRAL"
    return "NEGATIVE"


# ── Core Analysis ─────────────────────────────────────────────────────────────

def get_trending_topics(
    top_n: int = 5,
    sort_by: str = "volume",
    business_unit: Optional[str] = None,
    application_channel: Optional[str] = None,
    include_sentiment: bool = True,
    include_trend: bool = False,
) -> dict:
    """
    Aggregate topic-level metrics and return the top_n results.
    """
    df_chat, df_meta = _load()
    df_chat = _filter(df_chat, business_unit, application_channel)
    df_meta = _filter(df_meta, business_unit, application_channel)

    if df_chat.empty:
        return {"error": "No data found for the specified filters."}

    # ── Aggregate by topic ────────────────────────────────────────────────────
    agg = (
        df_chat
        .groupby("topic_name")
        .agg(
            volume              =("topic_name",              "count"),
            escalation_count    =("escalated_to_agent",      "sum"),
            thumbs_up_count     =("thumbs_up",               "sum"),
            thumbs_down_count   =("thumbs_down",             "sum"),
            avg_confidence_score=("confidence_score",        "mean"),
            avg_resolution_score=("derived_resolution_score","mean"),
            avg_handle_seconds  =("handle_seconds_dur",      "mean"),
        )
        .reset_index()
    )

    agg["escalation_rate"]  = (agg["escalation_count"] / agg["volume"] * 100).round(2)
    agg["thumbs_up_rate"]   = (agg["thumbs_up_count"] / agg["volume"] * 100).round(2)
    agg["thumbs_down_rate"] = (agg["thumbs_down_count"] / agg["volume"] * 100).round(2)
    agg["avg_confidence_score"] = agg["avg_confidence_score"].round(4)
    agg["avg_resolution_score"] = agg["avg_resolution_score"].round(4)
    agg["avg_handle_seconds"]   = agg["avg_handle_seconds"].round(1)

    # Sort
    sort_col_map = {
        "volume":           ("volume",              False),
        "escalation_rate":  ("escalation_rate",     False),
        "resolution_score": ("avg_resolution_score",False),
        "thumbs_up_rate":   ("thumbs_up_rate",      False),
    }
    sort_col, ascending = sort_col_map.get(sort_by, ("volume", False))
    agg = agg.sort_values(sort_col, ascending=ascending).head(top_n)

    # ── Sentiment breakdown (from metadata) ───────────────────────────────────
    sentiment_map: dict[str, dict] = {}
    if include_sentiment and not df_meta.empty:
        meta_agg = (
            df_meta
            .groupby("topic_name")
            .agg(
                meta_avg_resolution=("derived_resolution_score", "mean"),
                meta_count         =("topic_name",               "count"),
            )
            .reset_index()
        )
        for _, row in meta_agg.iterrows():
            label = _sentiment_label(row["meta_avg_resolution"])
            sentiment_map[row["topic_name"]] = {
                "dominant_sentiment": label,
                "avg_resolution":    round(row["meta_avg_resolution"], 4),
                "session_count":     int(row["meta_count"]),
            }

    # ── Week-over-week trend ──────────────────────────────────────────────────
    trend_map: dict[str, str] = {}
    if include_trend:
        df_chat["week"] = df_chat["created_at"].dt.isocalendar().week
        weeks = sorted(df_chat["week"].unique())
        if len(weeks) >= 2:
            w_prev, w_curr = weeks[-2], weeks[-1]
            vol_prev = df_chat[df_chat["week"] == w_prev].groupby("topic_name").size()
            vol_curr = df_chat[df_chat["week"] == w_curr].groupby("topic_name").size()
            for topic in agg["topic_name"]:
                prev = vol_prev.get(topic, 0)
                curr = vol_curr.get(topic, 0)
                if prev == 0:
                    trend_map[topic] = "NEW"
                elif curr > prev * 1.10:
                    trend_map[topic] = "TRENDING_UP"
                elif curr < prev * 0.90:
                    trend_map[topic] = "TRENDING_DOWN"
                else:
                    trend_map[topic] = "STABLE"

    # ── Build output ─────────────────────────────────────────────────────────
    topics_out = []
    for _, row in agg.iterrows():
        topic_name = row["topic_name"]
        entry: dict = {
            "topic":               topic_name,
            "volume":              int(row["volume"]),
            "escalation_rate_pct": float(row["escalation_rate"]),
            "thumbs_up_rate_pct":  float(row["thumbs_up_rate"]),
            "thumbs_down_rate_pct":float(row["thumbs_down_rate"]),
            "avg_confidence_score":float(row["avg_confidence_score"]),
            "avg_resolution_score":float(row["avg_resolution_score"]),
            "avg_handle_seconds":  float(row["avg_handle_seconds"]),
        }
        if include_sentiment and topic_name in sentiment_map:
            entry["sentiment"] = sentiment_map[topic_name]
        if include_trend:
            entry["wow_trend"] = trend_map.get(topic_name, "UNKNOWN")

        topics_out.append(entry)

    # ── Dataset-level summary ─────────────────────────────────────────────────
    total_volume = int(agg["volume"].sum())
    top_topic    = topics_out[0]["topic"] if topics_out else None

    return {
        "summary": {
            "top_topic_by_volume":      top_topic,
            "total_records_in_result":  total_volume,
            "sort_by":                  sort_by,
            "filters": {
                "business_unit":       business_unit or "ALL",
                "application_channel": application_channel or "ALL",
            },
        },
        "topics": topics_out,
    }


# ── LangChain Tool ────────────────────────────────────────────────────────────

class TopicTool(BaseTool):
    """
    LangChain tool that returns trending topics and sentiment analysis
    from the AEP Customer Intelligence platform data.

    Provides volume, escalation rate, thumbs feedback, confidence scores,
    and optional sentiment and trend signals per topic.
    """

    name: str = "aep_topics"
    description: str = (
        "Analyse trending topics and customer sentiment from AEP chatbot data. "
        "Returns the top N topics with volume, escalation rate, thumbs feedback, "
        "confidence score, and optional sentiment/trend breakdown. "
        "Use this when asked about: top issues, common topics, trending problems, "
        "which topics are escalating most, or sentiment by topic."
    )
    args_schema: Type[BaseModel] = TopicInput

    def _run(
        self,
        top_n: int = 5,
        sort_by: str = "volume",
        business_unit: Optional[str] = None,
        application_channel: Optional[str] = None,
        include_sentiment: bool = True,
        include_trend: bool = False,
    ) -> str:
        try:
            result = get_trending_topics(
                top_n=top_n,
                sort_by=sort_by,
                business_unit=business_unit,
                application_channel=application_channel,
                include_sentiment=include_sentiment,
                include_trend=include_trend,
            )
            return json.dumps(result, indent=2)
        except FileNotFoundError as exc:
            return json.dumps({"error": f"Data file not found: {exc}. Run generate_mock_data.py first."})
        except Exception as exc:
            log.error("TopicTool error: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Async not supported — use _run.")


if __name__ == "__main__":
    tool = TopicTool()
    print(tool.run({"top_n": 5, "include_trend": True}))
