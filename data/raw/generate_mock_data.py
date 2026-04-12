"""
generate_mock_data.py
---------------------
Generates three sources of mock customer interaction data simulating an
Adobe Experience Platform (AEP) deployment:

  1. chatbot_interactions  – turn-level chatbot conversation data
  2. clickstream_events    – web/app clickstream events per session
  3. conversation_metadata – session-level conversation metadata

Output: CSV files written to data/raw/
Run:    python data/raw/generate_mock_data.py
"""

import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_DAYS = 30
RECORDS_PER_SOURCE = 3000
START_DATE = datetime(2026, 3, 7)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

BUSINESS_UNITS = ["Retail", "Financial Services", "Healthcare", "Technology", "Travel"]
APPLICATION_CHANNELS = ["Web", "Mobile App", "Kiosk", "IVR", "Partner API"]
TOPICS = [
    "Account Management",
    "Billing Inquiry",
    "Product Support",
    "Order Status",
    "Returns & Refunds",
    "Technical Troubleshooting",
    "Subscription Management",
    "Password Reset",
    "Loyalty Program",
    "General Inquiry",
]
RESPONSE_TYPES = ["FAQ", "Guided Flow", "Free Text", "Escalation", "Handoff", "Self-Service"]


def _random_date(start: datetime, days: int) -> datetime:
    """Return a random datetime within `days` days of `start`."""
    delta_seconds = int(timedelta(days=days).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta_seconds))


def _contact_ids(n: int) -> list[str]:
    """Generate a pool of contact IDs (some shared across sources to enable joins)."""
    return [f"CONT-{str(uuid.uuid4())[:8].upper()}" for _ in range(n)]


# ── Source 1: Chatbot Interactions ───────────────────────────────────────────

def generate_chatbot_interactions(
    n: int = RECORDS_PER_SOURCE,
    contact_pool: list[str] | None = None,
) -> pd.DataFrame:
    """
    Turn-level chatbot interaction records.
    Each row represents one exchange (user message + bot response) within a session.
    """
    if contact_pool is None:
        contact_pool = _contact_ids(n)

    records = []
    for _ in range(n):
        contact_id = random.choice(contact_pool)
        session_id = f"SESS-{str(uuid.uuid4())[:8].upper()}"
        topic_name = random.choice(TOPICS)
        confidence_score = round(np.clip(np.random.normal(0.72, 0.18), 0.0, 1.0), 4)
        response_type = random.choice(RESPONSE_TYPES)

        # Thumbs feedback: sparse – only ~15 % of turns get feedback
        has_feedback = random.random() < 0.15
        thumbs_up = int(has_feedback and random.random() > 0.35)
        thumbs_down = int(has_feedback and thumbs_up == 0)

        # Escalation more likely on low confidence or troubleshooting topics
        escalation_probability = 0.08
        if confidence_score < 0.50:
            escalation_probability += 0.20
        if topic_name in {"Technical Troubleshooting", "Returns & Refunds"}:
            escalation_probability += 0.10
        escalated_to_agent = int(random.random() < escalation_probability)

        business_unit = random.choice(BUSINESS_UNITS)
        application_channel = random.choice(APPLICATION_CHANNELS)

        # Handle duration: lognormal, skewed longer for escalations
        base_seconds = np.random.lognormal(mean=4.2, sigma=0.8)
        handle_seconds_dur = int(base_seconds * (2.5 if escalated_to_agent else 1.0))

        # Derived resolution score: composite heuristic
        derived_resolution_score = round(
            confidence_score * 0.5
            + (thumbs_up * 0.3)
            - (thumbs_down * 0.2)
            - (escalated_to_agent * 0.15)
            + random.uniform(-0.05, 0.05),
            4,
        )
        derived_resolution_score = round(np.clip(derived_resolution_score, 0.0, 1.0), 4)

        created_at = _random_date(START_DATE, NUM_DAYS)

        records.append(
            {
                "contact_id": contact_id,
                "session_id": session_id,
                "topic_name": topic_name,
                "confidence_score": confidence_score,
                "response_type": response_type,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "escalated_to_agent": escalated_to_agent,
                "business_unit": business_unit,
                "application_channel": application_channel,
                "handle_seconds_dur": handle_seconds_dur,
                "derived_resolution_score": derived_resolution_score,
                "created_at": created_at.isoformat(),
            }
        )

    return pd.DataFrame(records)


# ── Source 2: Clickstream Events ─────────────────────────────────────────────

def generate_clickstream_events(
    n: int = RECORDS_PER_SOURCE,
    contact_pool: list[str] | None = None,
) -> pd.DataFrame:
    """
    Web/app clickstream events.
    Each row is one event (page view, button click, form submit, etc.) tied to a session.
    """
    if contact_pool is None:
        contact_pool = _contact_ids(n)

    EVENT_TYPES = ["page_view", "button_click", "form_submit", "search", "chat_open", "chat_close"]
    PAGE_NAMES = [
        "/home", "/account", "/billing", "/support", "/products",
        "/checkout", "/orders", "/faq", "/contact", "/settings",
    ]

    records = []
    for _ in range(n):
        contact_id = random.choice(contact_pool)
        session_id = f"SESS-{str(uuid.uuid4())[:8].upper()}"
        event_type = random.choice(EVENT_TYPES)
        page_name = random.choice(PAGE_NAMES)
        application_channel = random.choice(APPLICATION_CHANNELS)
        business_unit = random.choice(BUSINESS_UNITS)
        time_on_page_seconds = int(np.random.lognormal(mean=3.5, sigma=1.0))
        scroll_depth_pct = round(random.uniform(0.0, 100.0), 1)
        is_bounce = int(time_on_page_seconds < 5 and event_type == "page_view")
        event_timestamp = _random_date(START_DATE, NUM_DAYS)

        records.append(
            {
                "contact_id": contact_id,
                "session_id": session_id,
                "event_type": event_type,
                "page_name": page_name,
                "application_channel": application_channel,
                "business_unit": business_unit,
                "time_on_page_seconds": time_on_page_seconds,
                "scroll_depth_pct": scroll_depth_pct,
                "is_bounce": is_bounce,
                "event_timestamp": event_timestamp.isoformat(),
            }
        )

    return pd.DataFrame(records)


# ── Source 3: Conversation Metadata ──────────────────────────────────────────

def generate_conversation_metadata(
    n: int = RECORDS_PER_SOURCE,
    contact_pool: list[str] | None = None,
) -> pd.DataFrame:
    """
    Session-level conversation metadata.
    Aggregates session-wide signals: CSAT, containment, number of turns, etc.
    """
    if contact_pool is None:
        contact_pool = _contact_ids(n)

    LANGUAGES = ["en", "es", "fr", "de", "ja", "pt", "zh"]
    QUEUE_NAMES = ["Tier1-Support", "Billing-Ops", "Tech-Escalation", "General-Service"]

    records = []
    for _ in range(n):
        contact_id = random.choice(contact_pool)
        session_id = f"SESS-{str(uuid.uuid4())[:8].upper()}"
        topic_name = random.choice(TOPICS)
        business_unit = random.choice(BUSINESS_UNITS)
        application_channel = random.choice(APPLICATION_CHANNELS)
        language = random.choices(LANGUAGES, weights=[60, 15, 8, 5, 4, 5, 3])[0]

        num_turns = random.randint(1, 20)
        contained = int(random.random() > 0.25)  # 75 % containment rate
        escalated_to_agent = int(not contained and random.random() > 0.4)
        csat_score = (
            round(random.uniform(1.0, 5.0), 1) if random.random() > 0.45 else None
        )  # ~55 % CSAT response rate
        queue_name = random.choice(QUEUE_NAMES) if escalated_to_agent else None
        session_duration_seconds = int(
            num_turns * np.random.lognormal(mean=3.8, sigma=0.6)
        )
        confidence_score = round(np.clip(np.random.normal(0.70, 0.20), 0.0, 1.0), 4)
        derived_resolution_score = round(
            confidence_score * 0.4
            + (contained * 0.3)
            + ((csat_score or 3.0) / 5.0) * 0.3,
            4,
        )
        session_start = _random_date(START_DATE, NUM_DAYS)

        records.append(
            {
                "contact_id": contact_id,
                "session_id": session_id,
                "topic_name": topic_name,
                "business_unit": business_unit,
                "application_channel": application_channel,
                "language": language,
                "num_turns": num_turns,
                "contained": contained,
                "escalated_to_agent": escalated_to_agent,
                "csat_score": csat_score,
                "queue_name": queue_name,
                "session_duration_seconds": session_duration_seconds,
                "confidence_score": confidence_score,
                "derived_resolution_score": derived_resolution_score,
                "session_start": session_start.isoformat(),
            }
        )

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Shared contact pool so join keys overlap realistically (~60 % overlap)
    shared_pool = _contact_ids(int(RECORDS_PER_SOURCE * 0.6))

    print("Generating chatbot_interactions ...")
    df_chatbot = generate_chatbot_interactions(contact_pool=shared_pool)
    path_chatbot = os.path.join(OUTPUT_DIR, "chatbot_interactions.csv")
    df_chatbot.to_csv(path_chatbot, index=False)
    print(f"  Written {len(df_chatbot):,} rows -> {path_chatbot}")

    print("Generating clickstream_events ...")
    df_click = generate_clickstream_events(contact_pool=shared_pool)
    path_click = os.path.join(OUTPUT_DIR, "clickstream_events.csv")
    df_click.to_csv(path_click, index=False)
    print(f"  Written {len(df_click):,} rows -> {path_click}")

    print("Generating conversation_metadata ...")
    df_meta = generate_conversation_metadata(contact_pool=shared_pool)
    path_meta = os.path.join(OUTPUT_DIR, "conversation_metadata.csv")
    df_meta.to_csv(path_meta, index=False)
    print(f"  Written {len(df_meta):,} rows -> {path_meta}")

    print("\nDone. Summary:")
    for name, df in [
        ("chatbot_interactions", df_chatbot),
        ("clickstream_events", df_click),
        ("conversation_metadata", df_meta),
    ]:
        print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
