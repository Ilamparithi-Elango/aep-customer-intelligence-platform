"""
input_guard.py
--------------
Input validation and sanitization for the AEP Customer Intelligence Platform.

Protects against:
  - Prompt injection attacks (jailbreak patterns, system-override attempts)
  - Excessive input length (token budget protection)
  - Disallowed characters and control codes
  - SQL/shell injection patterns (defence-in-depth for tool inputs)

All functions raise ValueError with a user-safe message on validation failure.
They never raise with internal details that could aid an attacker.

Usage:
  from security.input_guard import sanitize_agent_input, validate_question_length
  clean = sanitize_agent_input(raw_user_input)
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_QUESTION_LENGTH  = 500    # characters
MAX_FIELD_LENGTH     = 100    # generic field cap
MIN_QUESTION_LENGTH  = 2

# Regex patterns for known injection vectors
_PROMPT_INJECTION_PATTERNS = [
    # System-override attempts
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+instructions",
    r"forget\s+(all\s+)?(previous|prior)\s+(instructions|rules|context)",
    r"new\s+(system\s+)?prompt",
    r"you\s+are\s+now\s+(?!an?\s+AEP)",   # "you are now [something else]"
    r"\bDAN\b",                             # "Do Anything Now" jailbreak
    r"jailbreak",
    r"<\s*system\s*>",                      # fake system tags
    r"\[INST\]",                            # Llama instruction tags
    r"###\s*instruction",
]

_SQL_INJECTION_PATTERNS = [
    r";\s*(drop|delete|truncate|alter|create|insert|update)\s+",
    r"\bunion\s+select\b",
    r"\bor\s+1\s*=\s*1\b",
    r"--\s+",                               # SQL comment
    r"/\*.*?\*/",                           # block comment
]

_SHELL_INJECTION_PATTERNS = [
    r"[|;&`$]",                             # shell metacharacters
    r"\$\(",                                # command substitution
    r"\.\./",                               # path traversal
    r"(?:rm|del|format|shutdown|reboot)\s",
]

# Pre-compiled combined patterns
_INJECTION_RE = re.compile(
    "|".join(_PROMPT_INJECTION_PATTERNS + _SQL_INJECTION_PATTERNS + _SHELL_INJECTION_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)

# Allowlist for categorical filter fields (business_unit, channel)
_ALLOWED_BUSINESS_UNITS = frozenset({
    "retail", "financial services", "healthcare", "technology", "travel"
})
_ALLOWED_CHANNELS = frozenset({
    "web", "mobile app", "kiosk", "ivr", "partner api"
})

# Characters that should never appear in user text inputs
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip_control_chars(text: str) -> str:
    """Remove ASCII control characters (keep \t, \n, \r)."""
    return _CONTROL_CHAR_RE.sub("", text)


def _normalize_unicode(text: str) -> str:
    """
    NFC-normalize Unicode and remove zero-width / invisible characters
    that can be used to hide injection payloads.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width spaces, zero-width joiners, etc.
    invisible = {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"}
    return "".join(ch for ch in text if ch not in invisible)


def _contains_injection(text: str) -> bool:
    """Return True if the text matches any injection pattern."""
    return bool(_INJECTION_RE.search(text))


# ── Public Validators ─────────────────────────────────────────────────────────

def validate_question_length(question: str) -> None:
    """
    Raise ValueError if the question is too short or too long.
    Should be called before sanitization to fail fast.
    """
    if not isinstance(question, str):
        raise ValueError("Input must be a string.")
    stripped = question.strip()
    if len(stripped) < MIN_QUESTION_LENGTH:
        raise ValueError("Question is too short. Please enter a complete question.")
    if len(stripped) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question is too long ({len(stripped)} chars). "
            f"Please keep it under {MAX_QUESTION_LENGTH} characters."
        )


def sanitize_agent_input(text: str) -> str:
    """
    Sanitize a user question before passing it to the LangChain agent.

    Steps:
      1. Strip leading/trailing whitespace
      2. Remove control characters
      3. Normalize Unicode
      4. Detect injection patterns → raise ValueError if found
      5. Collapse excessive whitespace
      6. Return cleaned string

    Raises:
      ValueError: if the input contains injection patterns.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    cleaned = text.strip()
    cleaned = _strip_control_chars(cleaned)
    cleaned = _normalize_unicode(cleaned)

    if _contains_injection(cleaned):
        # Log the attempt (with original text) but never reveal pattern details to caller
        log.warning("Potential injection attempt blocked. Length=%d", len(text))
        raise ValueError(
            "Your input contains patterns that are not allowed. "
            "Please ask a straightforward question about the data."
        )

    # Collapse runs of whitespace (but preserve single newlines for multi-line context)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned


def sanitize_filter_value(value: str, field_name: str = "field") -> str:
    """
    Sanitize a categorical filter value (e.g. business_unit, channel).
    Returns the stripped string if safe, raises ValueError otherwise.
    """
    if not isinstance(value, str):
        raise ValueError(f"Filter value for '{field_name}' must be a string.")
    cleaned = value.strip()
    if len(cleaned) > MAX_FIELD_LENGTH:
        raise ValueError(f"Filter value for '{field_name}' is too long.")
    if _contains_injection(cleaned):
        log.warning("Injection pattern in filter '%s': %s", field_name, cleaned[:50])
        raise ValueError(f"Invalid value for filter '{field_name}'.")
    return cleaned


def validate_business_unit(value: str) -> str:
    """
    Validate and normalise a business_unit filter value.
    Raises ValueError if not in the allowed set.
    """
    cleaned = sanitize_filter_value(value, field_name="business_unit")
    if cleaned.lower() not in _ALLOWED_BUSINESS_UNITS:
        raise ValueError(
            f"Unknown business unit '{cleaned}'. "
            f"Valid values: {sorted(_ALLOWED_BUSINESS_UNITS)}"
        )
    return cleaned


def validate_channel(value: str) -> str:
    """
    Validate and normalise an application_channel filter value.
    Raises ValueError if not in the allowed set.
    """
    cleaned = sanitize_filter_value(value, field_name="application_channel")
    if cleaned.lower() not in _ALLOWED_CHANNELS:
        raise ValueError(
            f"Unknown channel '{cleaned}'. "
            f"Valid values: {sorted(_ALLOWED_CHANNELS)}"
        )
    return cleaned


def sanitize_html_output(text: str) -> str:
    """
    Escape HTML special characters in text destined for web display.
    Prevents stored/reflected XSS if output is ever rendered in a browser.
    """
    return html.escape(str(text), quote=True)


def validate_date_string(value: str) -> str:
    """
    Validate that a date string is in YYYY-MM-DD format.
    Raises ValueError on malformed input.
    """
    import re as _re
    cleaned = sanitize_filter_value(value, field_name="date")
    if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        raise ValueError(
            f"Invalid date format '{cleaned}'. Expected YYYY-MM-DD."
        )
    # Basic range check
    year = int(cleaned[:4])
    if year < 2000 or year > 2100:
        raise ValueError(f"Date year {year} is out of expected range [2000, 2100].")
    return cleaned
