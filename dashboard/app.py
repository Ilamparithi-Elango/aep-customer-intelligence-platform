"""
app.py
------
Streamlit dashboard for the AEP Customer Intelligence Platform.

Sections:
  1. Sidebar  – filters (business unit, channel, date range)
  2. KPI Cards – containment rate, escalation rate, CSAT, avg confidence
  3. Charts    – escalation by topic, volume trend, thumbs feedback,
                 resolution score distribution, channel breakdown
  4. Topic Intelligence – top topics table with sentiment indicators
  5. AI Chat   – natural-language Q&A powered by the LangChain query agent

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path when launched from any directory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv()

# Lazy imports (only needed in the AI Chat section)
def _get_agent():
    from agent.query_agent import ask, build_agent
    return ask, build_agent

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AEP Customer Intelligence Platform",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data Paths ────────────────────────────────────────────────────────────────

RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
CHATBOT_CSV  = os.path.join(RAW_DIR, "chatbot_interactions.csv")
METADATA_CSV = os.path.join(RAW_DIR, "conversation_metadata.csv")

# ── Load Data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_chat = pd.read_csv(CHATBOT_CSV,  parse_dates=["created_at"])
    df_meta = pd.read_csv(METADATA_CSV, parse_dates=["session_start"])
    return df_chat, df_meta


def apply_filters(
    df_chat: pd.DataFrame,
    df_meta: pd.DataFrame,
    business_unit: str,
    channel: str,
    start_date,
    end_date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if business_unit != "All":
        df_chat = df_chat[df_chat["business_unit"] == business_unit]
        df_meta = df_meta[df_meta["business_unit"] == business_unit]
    if channel != "All":
        df_chat = df_chat[df_chat["application_channel"] == channel]
        df_meta = df_meta[df_meta["application_channel"] == channel]
    df_chat = df_chat[
        (df_chat["created_at"].dt.date >= start_date)
        & (df_chat["created_at"].dt.date <= end_date)
    ]
    df_meta = df_meta[
        (df_meta["session_start"].dt.date >= start_date)
        & (df_meta["session_start"].dt.date <= end_date)
    ]
    return df_chat, df_meta


# ── Colour Palette ────────────────────────────────────────────────────────────

ADOBE_RED   = "#FA0F00"
ADOBE_DARK  = "#1F1F1F"
ADOBE_BLUE  = "#1473E6"
ADOBE_GRAY  = "#6E6E6E"
PALETTE     = [ADOBE_BLUE, ADOBE_RED, "#2D9D78", "#E68619", "#8E4EC6"]


# ── CSS Overrides ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .kpi-card {
        background: #f8f9fa;
        border-left: 4px solid #1473E6;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 0.78rem; color: #6E6E6E; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1F1F1F; line-height: 1.1; }
    .kpi-delta { font-size: 0.8rem; color: #2D9D78; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #1F1F1F;
                     border-bottom: 2px solid #1473E6; padding-bottom: 4px;
                     margin: 24px 0 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, delta: str = "") -> None:
    st.metric(label=label, value=value, delta=delta if delta else None)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("AEP Customer Intelligence Platform")
    

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df_chat_raw, df_meta_raw = load_data()
    except FileNotFoundError:
        st.error(
            "Data files not found. Run `python data/raw/generate_mock_data.py` first."
        )
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    business_units = ["All"] + sorted(df_chat_raw["business_unit"].dropna().unique().tolist())
    channels       = ["All"] + sorted(df_chat_raw["application_channel"].dropna().unique().tolist())

    selected_bu      = st.sidebar.selectbox("Business Unit", business_units)
    selected_channel = st.sidebar.selectbox("Channel", channels)

    min_date = df_chat_raw["created_at"].dt.date.min()
    max_date = df_chat_raw["created_at"].dt.date.max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    # Handle single-date selection (user is mid-pick)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]

    
    # ── Apply filters ─────────────────────────────────────────────────────────
    df_chat, df_meta = apply_filters(
        df_chat_raw, df_meta_raw,
        selected_bu, selected_channel, start_date, end_date,
    )

    if df_chat.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    section("Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)

    containment     = df_meta["contained"].mean() * 100 if not df_meta.empty else 0
    escalation      = df_chat["escalated_to_agent"].mean() * 100
    avg_confidence  = df_chat["confidence_score"].mean()
    avg_resolution  = df_chat["derived_resolution_score"].mean()
    csat            = df_meta["csat_score"].dropna().mean() if not df_meta.empty else None

    with c1:
        kpi_card("Containment Rate", f"{containment:.1f}%")
    with c2:
        kpi_card("Escalation Rate", f"{escalation:.1f}%")
    with c3:
        kpi_card("Avg Confidence", f"{avg_confidence:.3f}")
    with c4:
        kpi_card("Avg Resolution", f"{avg_resolution:.3f}")
    with c5:
        kpi_card("CSAT Score", f"{csat:.2f} / 5" if csat else "N/A")

    st.markdown("")

    # ── Row 1: Volume trend + Escalation by topic ─────────────────────────────
    section("Volume & Escalation")
    col_left, col_right = st.columns(2)

    with col_left:
        df_daily = (
            df_chat.set_index("created_at")
            .resample("D")["session_id"]
            .count()
            .reset_index()
            .rename(columns={"session_id": "interactions", "created_at": "date"})
        )
        fig_trend = px.area(
            df_daily, x="date", y="interactions",
            title="Daily Interaction Volume",
            color_discrete_sequence=[ADOBE_BLUE],
            template="plotly_white",
        )
        fig_trend.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        topic_esc = (
            df_chat.groupby("topic_name")
            .agg(
                volume           =("topic_name",         "count"),
                escalations      =("escalated_to_agent", "sum"),
            )
            .assign(escalation_rate=lambda x: x["escalations"] / x["volume"] * 100)
            .sort_values("escalation_rate", ascending=False)
            .head(8)
            .reset_index()
        )
        fig_esc = px.bar(
            topic_esc, x="escalation_rate", y="topic_name",
            orientation="h",
            title="Escalation Rate by Topic (%)",
            color="escalation_rate",
            color_continuous_scale=["#2D9D78", "#E68619", ADOBE_RED],
            template="plotly_white",
        )
        fig_esc.update_layout(margin=dict(t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_esc, use_container_width=True)

    # ── Row 2: Thumbs feedback + Resolution distribution ─────────────────────
    section("Feedback & Quality")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        rated = df_chat[(df_chat["thumbs_up"] == 1) | (df_chat["thumbs_down"] == 1)]
        if len(rated) > 0:
            up_pct   = rated["thumbs_up"].sum()   / len(rated) * 100
            down_pct = rated["thumbs_down"].sum() / len(rated) * 100
            fig_thumb = go.Figure(data=[go.Pie(
                labels=["Thumbs Up", "Thumbs Down"],
                values=[up_pct, down_pct],
                hole=0.55,
                marker_colors=[ADOBE_BLUE, ADOBE_RED],
            )])
            fig_thumb.update_layout(
                title="Thumbs Feedback Split",
                margin=dict(t=40, b=10),
                template="plotly_white",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_thumb, use_container_width=True)
        else:
            st.info("No thumbs feedback data for this filter.")

    with col_b:
        fig_res = px.histogram(
            df_chat, x="derived_resolution_score", nbins=20,
            title="Resolution Score Distribution",
            color_discrete_sequence=[ADOBE_BLUE],
            template="plotly_white",
        )
        fig_res.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_res, use_container_width=True)

    with col_c:
        ch_vol = (
            df_chat.groupby("application_channel")
            .size()
            .reset_index(name="volume")
            .sort_values("volume", ascending=False)
        )
        fig_ch = px.bar(
            ch_vol, x="application_channel", y="volume",
            title="Volume by Channel",
            color="application_channel",
            color_discrete_sequence=PALETTE,
            template="plotly_white",
        )
        fig_ch.update_layout(margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_ch, use_container_width=True)

    # ── Row 3: Top topics table ───────────────────────────────────────────────
    section("Topic Intelligence")

    topic_summary = (
        df_chat.groupby("topic_name")
        .agg(
            Volume           =("topic_name",          "count"),
            Escalations      =("escalated_to_agent",  "sum"),
            Thumbs_Up        =("thumbs_up",            "sum"),
            Thumbs_Down      =("thumbs_down",          "sum"),
            Avg_Confidence   =("confidence_score",     "mean"),
            Avg_Resolution   =("derived_resolution_score", "mean"),
            Avg_Handle_Secs  =("handle_seconds_dur",   "mean"),
        )
        .assign(
            Escalation_Rate_Pct=lambda x: (x["Escalations"] / x["Volume"] * 100).round(1),
            Thumbs_Up_Rate_Pct =lambda x: (x["Thumbs_Up"]   / x["Volume"] * 100).round(1),
            Avg_Confidence     =lambda x: x["Avg_Confidence"].round(3),
            Avg_Resolution     =lambda x: x["Avg_Resolution"].round(3),
            Avg_Handle_Secs    =lambda x: x["Avg_Handle_Secs"].round(0).astype(int),
        )
        .sort_values("Volume", ascending=False)
        .reset_index()
        [["topic_name", "Volume", "Escalation_Rate_Pct", "Thumbs_Up_Rate_Pct",
          "Avg_Confidence", "Avg_Resolution", "Avg_Handle_Secs"]]
        .rename(columns={
            "topic_name":          "Topic",
            "Escalation_Rate_Pct": "Escalation %",
            "Thumbs_Up_Rate_Pct":  "Thumbs Up %",
            "Avg_Confidence":      "Avg Confidence",
            "Avg_Resolution":      "Avg Resolution",
            "Avg_Handle_Secs":     "Avg Handle (s)",
        })
    )

    st.dataframe(
        topic_summary.style
            .background_gradient(subset=["Escalation %"], cmap="RdYlGn_r")
            .background_gradient(subset=["Avg Resolution"], cmap="RdYlGn")
            .format({"Escalation %": "{:.1f}%", "Thumbs Up %": "{:.1f}%"}),
        use_container_width=True,
        height=380,
    )

    # ── Row 4: AI Chat Interface ──────────────────────────────────────────────
    section("AI Agent Chat")
    st.caption(
        "Ask natural-language questions about the data. "
        "Powered by LangChain + GPT-4o-mini."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input("Ask about metrics, topics, or trends …")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                openai_key = os.getenv("OPENAI_API_KEY", "")
                if not openai_key:
                    answer = (
                        "OPENAI_API_KEY is not configured. "
                        "Add it to your .env file to enable the AI agent."
                    )
                else:
                    try:
                        ask_fn, build_agent_fn = _get_agent()
                        # Build agent once per session
                        if st.session_state.agent_executor is None:
                            st.session_state.agent_executor = build_agent_fn()
                        answer = ask_fn(
                            user_input,
                            executor=st.session_state.agent_executor,
                        )
                    except Exception as exc:
                        answer = f"Agent error: {exc}"

            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Example queries helper
    with st.expander("Example questions"):
        examples = [
            "What is the overall containment rate?",
            "Which topic has the highest escalation rate?",
            "What is the thumbs-up rate on Mobile App?",
            "Show me the top 5 topics for Healthcare sorted by escalation.",
            "What is the average CSAT score for Financial Services?",
            "Which channel has the most interactions?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "AEP Customer Intelligence Platform · "
        "Built with Streamlit, PySpark, LangChain, OpenAI & AWS"
    )


if __name__ == "__main__":
    main()
