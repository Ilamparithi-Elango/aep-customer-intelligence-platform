"""
query_agent.py
--------------
LangChain ReAct agent that answers natural-language questions about the
AEP Customer Intelligence Platform data.

The agent is equipped with two tools:
  - MetricsTool  – KPI metrics (containment, escalation, CSAT, etc.)
  - TopicTool    – trending topics and sentiment analysis

Example queries:
  "What is the current containment rate for Retail?"
  "Which topics have the highest escalation rate?"
  "What is the thumbs-up rate on Mobile App?"
  "Show me the top 3 topics trending this week for Healthcare."

Usage:
  python agent/query_agent.py --question "What is the containment rate?"
  python agent/query_agent.py  # interactive mode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.tools.metrics_tool import MetricsTool
from agent.tools.topic_tool import TopicTool
from observability.logger import get_logger
from security.input_guard import sanitize_agent_input, validate_question_length

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AGENT_MODEL    = os.getenv("AGENT_MODEL", "gpt-4o-mini")
AGENT_TEMP     = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))
VERBOSE        = os.getenv("AGENT_VERBOSE", "false").lower() == "true"

log = get_logger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the AEP Customer Intelligence AI Assistant, an expert analytics agent for an
Adobe Experience Platform deployment.

You have access to two tools:
  1. aep_metrics  – for KPI metrics like containment rate, escalation rate, CSAT,
                    handle time, thumbs feedback, confidence scores.
  2. aep_topics   – for trending topics, topic-level escalation, sentiment analysis,
                    week-over-week trends.

RULES:
- Always use a tool to retrieve data before answering quantitative questions.
- If the user's question is ambiguous, make a reasonable assumption and state it.
- Format numerical answers clearly (percentages, rounded to 2 decimal places).
- If a filter (business unit, channel, date) is mentioned, pass it to the tool.
- Do not fabricate numbers. If data is unavailable, say so.
- Keep answers concise — 3–5 sentences maximum unless the user asks for detail.
- When listing topics or metrics, use a numbered or bulleted list."""


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_agent():
    """Construct and return the LangGraph ReAct agent."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatOpenAI(
        model=AGENT_MODEL,
        temperature=AGENT_TEMP,
        api_key=OPENAI_API_KEY,
    )

    tools = [MetricsTool(), TopicTool()]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )


# ── Public Query Interface ────────────────────────────────────────────────────

def ask(question: str, executor=None) -> str:
    """
    Ask the agent a natural-language question about AEP data.

    Args:
        question:  The user's question string.
        executor:  Optionally pass a pre-built agent graph (avoids rebuilding).

    Returns:
        The agent's answer as a string.
    """
    # Input validation / sanitization
    try:
        validate_question_length(question)
        clean_question = sanitize_agent_input(question)
    except ValueError as exc:
        return f"Invalid input: {exc}"

    if executor is None:
        executor = build_agent()

    log.info("Agent query: %s", clean_question)

    try:
        result = executor.invoke(
            {"messages": [HumanMessage(content=clean_question)]}
        )
        # Last message in the graph output is the final answer
        answer = result["messages"][-1].content
        log.info("Agent answer: %s", answer[:200])
        return answer
    except Exception as exc:
        log.error("Agent error: %s", exc, exc_info=True)
        return f"An error occurred while processing your question: {exc}"


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AEP Customer Intelligence Query Agent.")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Question to ask the agent. If omitted, enters interactive mode.")
    args = parser.parse_args()

    print("=== AEP Customer Intelligence Agent ===")
    print(f"Model: {AGENT_MODEL}  |  Max iterations: {MAX_ITERATIONS}")
    print()

    try:
        executor = build_agent()
    except EnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        answer = ask(args.question, executor=executor)
        print(f"Q: {args.question}")
        print(f"A: {answer}")
    else:
        # Interactive loop
        print("Type your question and press Enter. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            answer = ask(question, executor=executor)
            print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
