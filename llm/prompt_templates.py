"""
prompt_templates.py
-------------------
Versioned prompt templates for all LLM tasks in the AEP Customer Intelligence
Platform.  Templates are stored as typed dataclasses so callers get IDE
completion and version tracking in git.

Template catalogue:
  - SummarizationTemplate     v1 / v2  – compress a chatbot transcript
  - TopicClassificationTemplate v1     – classify a conversation into a topic
  - AgentScoringTemplate       v1      – score agent quality from a transcript
  - SentimentTemplate          v1      – detect sentiment from conversation text
  - InsightTemplate            v1      – generate business insights from metrics

Usage:
  from llm.prompt_templates import get_template, TemplateKey

  tmpl = get_template(TemplateKey.SUMMARIZATION, version=2)
  prompt = tmpl.render(transcript="User: Hi …\nBot: Hello …", max_words=80)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any


# ── Template Registry Key ─────────────────────────────────────────────────────

class TemplateKey(str, Enum):
    SUMMARIZATION        = "summarization"
    TOPIC_CLASSIFICATION = "topic_classification"
    AGENT_SCORING        = "agent_scoring"
    SENTIMENT            = "sentiment"
    INSIGHT              = "insight"


# ── Base Template ─────────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """
    A versioned, named prompt template.

    `system` and `user` use Python's string.Template syntax: ${variable}.
    Call `.render(**kwargs)` to produce the final strings.
    """
    key: TemplateKey
    version: int
    description: str
    system: str
    user: str
    output_format: str = "text"          # "text" | "json"
    max_tokens: int = 512
    temperature: float = 0.2
    tags: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> dict[str, str]:
        """
        Substitute template variables and return {"system": ..., "user": ...}.

        Raises KeyError if a required variable is missing.
        """
        try:
            system_rendered = Template(self.system).substitute(**kwargs)
            user_rendered   = Template(self.user).substitute(**kwargs)
        except KeyError as exc:
            raise KeyError(f"[{self.key}@v{self.version}] Missing variable: {exc}") from exc
        return {"system": system_rendered, "user": user_rendered}

    def to_openai_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """Return an OpenAI-compatible messages list."""
        rendered = self.render(**kwargs)
        return [
            {"role": "system",  "content": rendered["system"]},
            {"role": "user",    "content": rendered["user"]},
        ]


# ── Template Definitions ──────────────────────────────────────────────────────

# ---------- Summarization v1 ----------
SUMMARIZATION_V1 = PromptTemplate(
    key=TemplateKey.SUMMARIZATION,
    version=1,
    description="Summarize a chatbot conversation transcript in ≤ max_words words.",
    system=(
        "You are a concise summarization assistant for enterprise customer service data. "
        "Produce factual, neutral summaries suitable for business analytics."
    ),
    user=(
        "Summarize the following customer–chatbot conversation in at most ${max_words} words.\n"
        "Focus on: the customer's core issue, the bot's resolution (if any), and the outcome.\n\n"
        "TRANSCRIPT:\n${transcript}\n\nSUMMARY:"
    ),
    output_format="text",
    max_tokens=200,
    temperature=0.1,
    tags=["summarization", "v1", "analytics"],
)

# ---------- Summarization v2 (structured JSON output) ----------
SUMMARIZATION_V2 = PromptTemplate(
    key=TemplateKey.SUMMARIZATION,
    version=2,
    description=(
        "Summarize a transcript and return structured JSON with summary, issue, and outcome."
    ),
    system=(
        "You are an enterprise customer-service analytics assistant. "
        "Always respond with valid JSON only — no markdown, no explanation."
    ),
    user=(
        "Analyze the following chatbot transcript and return a JSON object with these fields:\n"
        '  "summary"      : string  – ≤ ${max_words}-word neutral summary\n'
        '  "core_issue"   : string  – the customer\'s primary problem (≤ 10 words)\n'
        '  "outcome"      : string  – one of: RESOLVED, ESCALATED, UNRESOLVED, PARTIAL\n'
        '  "sentiment"    : string  – one of: POSITIVE, NEUTRAL, NEGATIVE, MIXED\n'
        '  "key_entities" : list    – up to 5 named entities (products, features, etc.)\n\n'
        "TRANSCRIPT:\n${transcript}\n\nJSON:"
    ),
    output_format="json",
    max_tokens=300,
    temperature=0.0,
    tags=["summarization", "v2", "structured", "analytics"],
)

# ---------- Topic Classification v1 ----------
TOPIC_CLASSIFICATION_V1 = PromptTemplate(
    key=TemplateKey.TOPIC_CLASSIFICATION,
    version=1,
    description="Classify a conversation into one of the known AEP topic categories.",
    system=(
        "You are a topic classification assistant for an enterprise customer service platform. "
        "Classify conversations accurately into one of the provided categories. "
        "Respond with JSON only."
    ),
    user=(
        "Classify the following conversation excerpt into EXACTLY ONE topic from this list:\n"
        "${topic_list}\n\n"
        "Return a JSON object:\n"
        '  "topic"       : string  – the chosen topic (must match list exactly)\n'
        '  "confidence"  : float   – your confidence 0.0–1.0\n'
        '  "reasoning"   : string  – one sentence explaining the classification\n\n'
        "CONVERSATION:\n${conversation}\n\nJSON:"
    ),
    output_format="json",
    max_tokens=150,
    temperature=0.0,
    tags=["classification", "v1", "topic"],
)

# ---------- Agent Scoring v1 ----------
AGENT_SCORING_V1 = PromptTemplate(
    key=TemplateKey.AGENT_SCORING,
    version=1,
    description="Score a human agent's performance after escalation based on transcript.",
    system=(
        "You are a quality-assurance evaluator for a customer service team. "
        "Score agent performance objectively on a 1–5 scale across four dimensions. "
        "Respond with JSON only."
    ),
    user=(
        "Evaluate the human agent's performance in the following escalated conversation.\n\n"
        "Return a JSON object with scores (1 = poor, 5 = excellent):\n"
        '  "empathy_score"       : int   – did the agent acknowledge the customer\'s frustration?\n'
        '  "resolution_score"    : int   – did the agent resolve the issue effectively?\n'
        '  "efficiency_score"    : int   – was the handle time reasonable for the complexity?\n'
        '  "professionalism_score": int  – was the language professional and brand-aligned?\n'
        '  "overall_score"       : float – weighted average (empathy 25%, rest 25% each)\n'
        '  "coaching_notes"      : string – one actionable coaching suggestion\n\n'
        "TRANSCRIPT:\n${transcript}\n\n"
        "Context – handle time: ${handle_seconds} seconds, business unit: ${business_unit}\n\n"
        "JSON:"
    ),
    output_format="json",
    max_tokens=250,
    temperature=0.1,
    tags=["agent-scoring", "v1", "qa"],
)

# ---------- Sentiment v1 ----------
SENTIMENT_V1 = PromptTemplate(
    key=TemplateKey.SENTIMENT,
    version=1,
    description="Detect customer sentiment and emotional signals from conversation text.",
    system=(
        "You are a sentiment analysis specialist for customer service conversations. "
        "Be precise and conservative – only flag strong signals. Respond with JSON only."
    ),
    user=(
        "Analyse the customer's sentiment in the following conversation excerpt.\n\n"
        "Return a JSON object:\n"
        '  "overall_sentiment"  : string – POSITIVE | NEUTRAL | NEGATIVE | MIXED\n'
        '  "sentiment_score"    : float  – −1.0 (very negative) to +1.0 (very positive)\n'
        '  "frustration_signal" : bool   – true if the customer expressed frustration\n'
        '  "urgency_signal"     : bool   – true if the customer indicated urgency\n'
        '  "churn_risk"         : string – LOW | MEDIUM | HIGH\n\n'
        "CONVERSATION:\n${conversation}\n\nJSON:"
    ),
    output_format="json",
    max_tokens=150,
    temperature=0.0,
    tags=["sentiment", "v1", "analytics"],
)

# ---------- Insight v1 ----------
INSIGHT_V1 = PromptTemplate(
    key=TemplateKey.INSIGHT,
    version=1,
    description="Generate executive business insights from aggregated KPI metrics.",
    system=(
        "You are a senior customer experience analyst at an enterprise software company. "
        "Interpret KPI data and provide clear, actionable insights for business leaders. "
        "Be specific and evidence-based."
    ),
    user=(
        "Based on the following AEP platform metrics for ${time_period}, "
        "provide 3–5 concise business insights and recommendations.\n\n"
        "METRICS:\n${metrics_json}\n\n"
        "Format as a numbered list. Each insight should:\n"
        "  1. State the observation (what the data shows)\n"
        "  2. Explain the business impact\n"
        "  3. Suggest one concrete action\n\n"
        "INSIGHTS:"
    ),
    output_format="text",
    max_tokens=600,
    temperature=0.3,
    tags=["insight", "v1", "executive"],
)


# ── Registry ──────────────────────────────────────────────────────────────────

# Maps (TemplateKey, version) → PromptTemplate
_REGISTRY: dict[tuple[TemplateKey, int], PromptTemplate] = {
    (TemplateKey.SUMMARIZATION,        1): SUMMARIZATION_V1,
    (TemplateKey.SUMMARIZATION,        2): SUMMARIZATION_V2,
    (TemplateKey.TOPIC_CLASSIFICATION, 1): TOPIC_CLASSIFICATION_V1,
    (TemplateKey.AGENT_SCORING,        1): AGENT_SCORING_V1,
    (TemplateKey.SENTIMENT,            1): SENTIMENT_V1,
    (TemplateKey.INSIGHT,              1): INSIGHT_V1,
}

# Latest version for each key
_LATEST: dict[TemplateKey, int] = {
    TemplateKey.SUMMARIZATION:        2,
    TemplateKey.TOPIC_CLASSIFICATION: 1,
    TemplateKey.AGENT_SCORING:        1,
    TemplateKey.SENTIMENT:            1,
    TemplateKey.INSIGHT:              1,
}


def get_template(key: TemplateKey, version: int | None = None) -> PromptTemplate:
    """
    Retrieve a prompt template by key and optional version.
    If version is None, returns the latest version.
    """
    v = version if version is not None else _LATEST[key]
    tmpl = _REGISTRY.get((key, v))
    if tmpl is None:
        available = [k for k in _REGISTRY if k[0] == key]
        raise ValueError(
            f"Template '{key}' v{v} not found. "
            f"Available versions: {[k[1] for k in available]}"
        )
    return tmpl


def list_templates() -> list[dict]:
    """Return a summary of all registered templates."""
    return [
        {
            "key":         tmpl.key.value,
            "version":     tmpl.version,
            "description": tmpl.description,
            "output_format": tmpl.output_format,
            "max_tokens":  tmpl.max_tokens,
            "tags":        tmpl.tags,
        }
        for tmpl in _REGISTRY.values()
    ]


if __name__ == "__main__":
    import json
    print("Registered prompt templates:")
    print(json.dumps(list_templates(), indent=2))
