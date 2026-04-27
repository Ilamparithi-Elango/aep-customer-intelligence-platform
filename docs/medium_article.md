# Agentic AI Project: Build an AWS-Native Customer Intelligence Platform with LLM Enrichment and a Conversational AI Agent

## From raw chatbot logs to live business insights — a complete end-to-end pipeline using PySpark, GPT-4o-mini, LangGraph, and Streamlit [Code Included]

---

*Every large enterprise generates millions of customer interaction records. Most of it sits in raw logs, never analysed. Product managers ask the same three questions every week — "What is our containment rate?", "Which topics are escalating most?", "What's driving churn risk?" — and someone has to manually pull the data each time.*

*This project automates all of it.*

---

## What We're Building

The **AEP Customer Intelligence Platform** is a fully automated AWS-native data pipeline that:

1. **Ingests** three sources of raw customer interaction data (chatbot logs, clickstream events, session metadata)
2. **Transforms** them into a unified, enriched Parquet dataset using PySpark
3. **Enriches** every record with GPT-4o-mini — extracting sentiment, conversation summaries, and topic classifications concurrently
4. **Serves** a conversational AI agent and a live Streamlit dashboard so any stakeholder can query the data in plain English

The entire stack runs on AWS. Orchestration is handled by an Airflow DAG that kicks off daily at 02:00 UTC. Everything is observable through structured JSON logs compatible with CloudWatch Logs Insights, and LLM cost is tracked to the cent.

Here's the architecture:

![AEP Customer Intelligence Platform AWS Architecture](AEP_Customer_Intelligence_platform.png)

---

## Why This Problem Matters

Customer service analytics is one of the most underserved areas in enterprise data engineering. Teams invest heavily in chatbots and contact centre platforms, but the insight layer is usually an afterthought — a few static dashboards, a weekly CSV export, and a lot of Slack messages asking "can you just pull that number for me?"

The result:
- **Product managers** can't self-serve. They wait on data teams.
- **Operations leaders** see aggregate metrics but can't drill down into *why* escalation rates spiked on Tuesday.
- **LLM outputs are wasted** — most companies run transcripts through models, get unstructured text back, and store it in a column nobody queries.

This platform solves all three. Let's walk through how it's built.

---

## The Tech Stack

| Layer | Technology |
|---|---|
| Cloud Storage | AWS S3 (AES-256 encryption + versioning) |
| Orchestration | Apache Airflow 2.x (daily DAG, 7 tasks) |
| Data Processing | PySpark 3.5 (data quality + Glue-style transform) |
| Data Format | Apache Parquet (snappy compressed, date-partitioned) |
| LLM | OpenAI GPT-4o-mini (concurrent batch processing) |
| Agent Framework | LangGraph `create_react_agent` + custom tools |
| Dashboard | Streamlit + Plotly |
| Security | Custom prompt injection guard + PII masking |
| Observability | Structured JSON logging + LLM cost tracker |
| Testing | pytest (unit tests, zero external calls) |

---

## Project Structure

```
aep-customer-intelligence-platform/
├── data/raw/
│   └── generate_mock_data.py      # 3 sources × 3,000 rows each
├── infrastructure/
│   └── aws_setup.py               # S3 provisioning via boto3
├── pipeline/
│   ├── data_quality.py            # PySpark DQ checks
│   ├── glue_transform.py          # PySpark join & enrich → Parquet
│   └── airflow_dag.py             # Daily orchestration DAG
├── llm/
│   ├── prompt_templates.py        # Versioned prompt registry
│   └── batch_processor.py         # Concurrent OpenAI batch processor
├── agent/
│   ├── tools/metrics_tool.py      # LangChain KPI metrics tool
│   ├── tools/topic_tool.py        # LangChain trending topics tool
│   └── query_agent.py             # LangGraph ReAct agent
├── security/
│   └── input_guard.py             # Injection detection + sanitization
├── observability/
│   └── logger.py                  # JSON logging + cost tracking
├── dashboard/
│   └── app.py                     # Streamlit dashboard + AI chat
└── tests/
    └── test_pipeline.py           # pytest unit tests
```

---

## Part 1 — Data Generation: Realistic Mock Data Without Hardcoding

Before building any pipeline, we need data. The mock data generator produces three CSV files that simulate a real AEP deployment.

The key design decision: **no fixed random seed, no hardcoded outcomes**. Each run produces genuinely different data — different containment rates, different escalating topics, different confidence distributions — so the downstream pipeline has to actually work rather than just matching a pre-determined answer.

```python
# Per-run parameters — set ONCE, applied consistently to all 3,000 records
high_escalation_topics = set(random.sample(TOPICS, 2))  # 2 random topics per run
confidence_mean        = random.uniform(0.60, 0.82)     # shifts the NLU confidence curve
feedback_rate          = random.uniform(0.10, 0.25)     # how often customers rate the session
base_escalation        = random.uniform(0.05, 0.15)     # baseline escalation probability
escalation_multiplier  = random.uniform(1.8, 3.2)       # handle-time boost for escalated calls

# Per-record: escalation probability built from signals
escalation_probability = base_escalation
if confidence_score < 0.50:
    escalation_probability += random.uniform(0.10, 0.25)
if topic_name in high_escalation_topics:
    escalation_probability += random.uniform(0.08, 0.18)
escalated_to_agent = int(random.random() < min(escalation_probability, 1.0))
```

This matters more than it sounds. Many portfolio projects generate data with `random.seed(42)` and then build "analysis" that just reproduces the same numbers every time. Here, you could run the pipeline on Monday and see 72% containment; run it again on Thursday and see 79%, with completely different topics leading escalations. The code has to handle both — and it does.

The three sources are joined on `contact_id` + `session_id` with a shared contact pool to create realistic overlap (~60% join rate, matching production behaviour).

---

## Part 2 — AWS Infrastructure: S3 with Security Defaults

Infrastructure is provisioned in Python via boto3 — no ClickOps.

```python
s3.create_bucket(Bucket=bucket_name, ...)

# Encryption: AES-256 on every object by default
s3.put_bucket_encryption(
    Bucket=bucket_name,
    ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
    },
)

# Versioning: protect against accidental overwrites
s3.put_bucket_versioning(
    Bucket=bucket_name,
    VersioningConfiguration={"Status": "Enabled"},
)

# Block all public access — no accidental data exposure
s3.put_public_access_block(
    Bucket=bucket_name,
    PublicAccessBlockConfiguration={
        "BlockPublicAcls": True,
        "IgnorePublicAcls": True,
        "BlockPublicPolicy": True,
        "RestrictPublicBuckets": True,
    },
)
```

A `--dry-run` flag lets you preview every action before it executes — useful for CI/CD environments or when reviewing changes before production deploys.

---

## Part 3 — PySpark Pipeline: Data Quality + Glue-Style Transform

### Data Quality Checks

Before transforming anything, a PySpark data quality job validates all three sources:

- **Null checks** on primary keys (`contact_id`, `session_id`)
- **Range validation** on numeric fields (confidence scores must be 0–1)
- **Referential completeness** — what percentage of chatbot sessions have matching metadata?
- **Recency check** — are we receiving fresh data or stale backfill?

All results are written as structured log output that CloudWatch Logs Insights can query directly.

### The Glue-Style Transform

The transform joins all three sources and computes four derived columns:

```python
def enrich(df: DataFrame) -> DataFrame:

    # Coalesce confidence from two sources
    df = df.withColumn(
        "effective_confidence_score",
        F.coalesce(F.col("confidence_score"), F.col("meta_confidence_score")),
    )

    # Resolution tier: business rule, not a magic number
    df = df.withColumn(
        "resolution_tier",
        F.when(F.col("derived_resolution_score") >= 0.7, "HIGH")
         .when(F.col("derived_resolution_score") >= 0.4, "MEDIUM")
         .otherwise("LOW"),
    )

    # Session quality score: weighted composite KPI
    # Nulls contribute 0.0 — missing data doesn't inflate scores
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

    # Date partitions for Hive-style Parquet partitioning
    df = df.withColumn("year",  F.year("created_at_ts").cast("string"))
    df = df.withColumn("month", F.lpad(F.month("created_at_ts").cast("string"), 2, "0"))
    df = df.withColumn("day",   F.lpad(F.dayofmonth("created_at_ts").cast("string"), 2, "0"))

    return df
```

The output is snappy-compressed Parquet, partitioned by `year/month/day` — ready for Athena queries or downstream Spark jobs.

One subtle but important decision: nulls in `coalesce(..., F.lit(0.0))`, **not** `F.lit(0.5)`. A missing confidence score means we don't know — defaulting to 0.5 would artificially inflate the session quality score for records with incomplete data. Defaulting to 0.0 is honest: missing data pulls the score down rather than propping it up.

---

## Part 4 — LLM Batch Processing: Three Enrichments Per Record

Every record gets three GPT-4o-mini enrichments in a single batch run:

1. **Summarisation** — structured JSON with `summary`, `core_issue`, `outcome`, `sentiment`, `key_entities`
2. **Sentiment analysis** — `overall_sentiment`, `sentiment_score`, `frustration_signal`, `churn_risk`
3. **Topic classification** — classify into one of 10 business topics with a confidence score

### The Prompt Template System

Prompts are versioned, typed dataclasses — not f-strings scattered across the codebase:

```python
SUMMARIZATION_V2 = PromptTemplate(
    key=TemplateKey.SUMMARIZATION,
    version=2,
    description="Summarize a transcript and return structured JSON.",
    system=(
        "You are an enterprise customer-service analytics assistant. "
        "Always respond with valid JSON only — no markdown, no explanation."
    ),
    user=(
        'Analyze the following chatbot transcript and return a JSON object:\n'
        '  "summary"    : string  – ≤ ${max_words}-word neutral summary\n'
        '  "core_issue" : string  – the customer\'s primary problem (≤ 10 words)\n'
        '  "outcome"    : string  – RESOLVED | ESCALATED | UNRESOLVED | PARTIAL\n'
        '  "churn_risk" : string  – LOW | MEDIUM | HIGH\n\n'
        "TRANSCRIPT:\n${transcript}\n\nJSON:"
    ),
    output_format="json",
    max_tokens=300,
    temperature=0.0,
)
```

New prompt versions are added to the registry without touching call sites — callers always use `get_template(TemplateKey.SUMMARIZATION, version=2)`.

### Concurrent Processing with ThreadPoolExecutor

Processing 3,000 records one-by-one would take hours. The batch processor runs 10 workers concurrently, each making independent OpenAI API calls:

```python
def run_batch(records, model=DEFAULT_MODEL, max_workers=10):
    client  = OpenAI(api_key=OPENAI_API_KEY)
    tracker = UsageTracker()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, row, client, model, tracker): i
            for i, row in enumerate(records)
        }
        for done_future in as_completed(futures):
            try:
                results.append(done_future.result())
            except Exception as exc:
                log.error("Record %d failed: %s", futures[done_future], exc)

    log.info("Batch complete. %s", tracker.summary())
    return results, tracker
```

Rate limit errors are retried with exponential backoff. The tracker accumulates token usage and computes exact USD cost using a single source of truth:

```python
_MODEL_PRICING = {
    "gpt-4o":      {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}
```

One important design detail: **topic is intentionally excluded from the synthetic transcript**. If you include the known topic in the text you send to the LLM for classification, you're measuring whether the model can read — not whether it can classify. Removing it means the topic classification is genuinely independent.

---

## Part 5 — The AI Agent: Natural Language Queries Over Live Data

The agent is built with **LangGraph's `create_react_agent`** — the modern replacement for the deprecated LangChain `AgentExecutor`. It uses a ReAct (Reason + Act) loop to decide which tool to call, call it, observe the result, and formulate an answer.

```python
def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    tools = [MetricsTool(), TopicTool()]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

def ask(question: str, executor=None) -> str:
    validate_question_length(question)
    clean_question = sanitize_agent_input(question)

    result = executor.invoke(
        {"messages": [HumanMessage(content=clean_question)]}
    )
    return result["messages"][-1].content
```

The agent has two tools:

**`aep_metrics`** — answers KPI questions. Given a question like "What is the containment rate for Healthcare on Mobile App?", it filters the dataset, computes the metric, and returns a structured dict.

**`aep_topics`** — answers topic trend questions. Supports sorting by volume, escalation rate, resolution score, or thumbs-up rate. Includes week-over-week trend detection (TRENDING_UP / STABLE / TRENDING_DOWN) and resolution quality breakdowns per topic.

The system prompt enforces strict behaviour:

```
RULES:
- Always use a tool to retrieve data before answering quantitative questions.
- If the user's question is ambiguous, make a reasonable assumption and state it.
- Format numerical answers clearly (percentages, rounded to 2 decimal places).
- Do not fabricate numbers. If data is unavailable, say so.
- Keep answers concise — 3–5 sentences unless the user asks for detail.
```

The "do not fabricate numbers" rule matters. Without it, GPT-4o-mini will confidently invent plausible-looking statistics from training data. With tools and a strict system prompt, it either retrieves the real number or says it can't find it.

---

## Part 6 — Security: Prompt Injection Protection

Any time you expose an LLM agent to user input, you need an injection guard. The `input_guard.py` module checks for:

- **System-override attempts**: "ignore all previous instructions", "you are now DAN", fake `<system>` tags
- **SQL injection patterns**: `UNION SELECT`, `DROP TABLE`, SQL comment markers
- **Shell metacharacters**: `|`, `;`, `$()`, `../` path traversal
- **Unicode smuggling**: zero-width characters and invisible codepoints used to hide payloads

```python
_INJECTION_RE = re.compile(
    "|".join(
        _PROMPT_INJECTION_PATTERNS +
        _SQL_INJECTION_PATTERNS +
        _SHELL_INJECTION_PATTERNS
    ),
    flags=re.IGNORECASE | re.DOTALL,
)

def sanitize_agent_input(text: str) -> str:
    cleaned = text.strip()
    cleaned = _strip_control_chars(cleaned)
    cleaned = _normalize_unicode(cleaned)   # removes zero-width chars

    if _contains_injection(cleaned):
        log.warning("Potential injection attempt blocked. Length=%d", len(text))
        raise ValueError(
            "Your input contains patterns that are not allowed. "
            "Please ask a straightforward question about the data."
        )
    return cleaned
```

Notice the error message deliberately says nothing about *which* pattern was matched. Revealing that detail would help an attacker refine their payload.

The observability layer adds a second layer: API keys (`sk-...`), AWS access keys, SSNs, and email addresses are regex-masked before they can appear in any log output — even if someone accidentally logs a raw request object.

---

## Part 7 — Observability: Structured Logging + Cost Tracking

Every component uses the same structured JSON logger, which emits CloudWatch-compatible records:

```json
{
  "timestamp": "2026-04-26 14:23:01",
  "level": "INFO",
  "logger": "batch_processor",
  "message": "Batch complete in 47.2s.",
  "service": "aep-customer-intelligence-platform",
  "prompt_tokens": 184320,
  "completion_tokens": 61440,
  "total_cost_usd": 0.0646
}
```

The `CostTracker` accumulates token usage across a batch run and breaks it down by model:

```python
with CostTracker() as tracker:
    tracker.record_llm_call(
        model="gpt-4o-mini",
        prompt_tokens=150,
        completion_tokens=80,
        task="summarization",
    )
# Automatically logs summary on __exit__
```

The `@timed_step` decorator wraps any pipeline function and logs start, completion, and elapsed time — giving you a full timing trace without adding boilerplate to every function.

---

## Part 8 — The Dashboard: Interactive Analytics + AI Chat in One UI

The Streamlit dashboard has two panels:

**Analytics panel:**
- KPI cards: containment rate, escalation rate, thumbs-up rate, avg confidence — all computed live from the filtered dataset
- Escalation trend chart (line chart, daily)
- Topic distribution bar chart
- Filterable session table with all raw + derived fields
- Channel and business unit breakdowns

**AI Chat panel:**
- Text input connected directly to the LangGraph agent
- Responses appear inline, with no page reload
- The agent's tool calls are invisible to the user — they just ask a question and get an answer

```python
# Agent is built once per session, not per question
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

if prompt := st.chat_input("Ask about your data..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(prompt, executor=st.session_state.agent)
        st.markdown(response)
```

---

## Part 9 — Airflow Orchestration: Daily Pipeline in 7 Tasks

The full pipeline runs as a single Airflow DAG:

```
check_env
    → generate_mock_data
    → provision_aws
    → data_quality_checks
    → glue_transform
    → llm_batch_process
    → notify_success
```

Each task is a `PythonOperator` wrapping the functions we've already described. The DAG runs at `0 2 * * *` (02:00 UTC daily), and each step logs its output to CloudWatch. If `data_quality_checks` fails, the downstream transform never runs — no corrupted data in production.

---

## Running It Yourself

```bash
# 1. Clone and install
git clone https://github.com/Ilamparithi-Elango/aep-customer-intelligence-platform.git
cd aep-customer-intelligence-platform
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env   # Mac/Linux
# copy .env.example .env  (Windows)
# Fill in OPENAI_API_KEY, AWS_REGION, AWS_S3_BUCKET

# 3. Generate mock data
python data/raw/generate_mock_data.py

# 4. Provision AWS
python infrastructure/aws_setup.py

# 5. Run PySpark transform locally
spark-submit --master local[*] pipeline/glue_transform.py --local

# 6. Run LLM batch processor (50 records to test)
python -m llm.batch_processor --local --limit 50

# 7. Launch dashboard
streamlit run dashboard/app.py

# 8. Query the agent directly
python -m agent.query_agent --question "Which topics have the highest escalation rate?"
```

---

## Key Design Decisions Worth Noting

**1. No hardcoded results in mock data.** Every interesting metric — containment rate, escalation leaders, confidence distribution — shifts with each run. This forces the entire pipeline to be genuinely data-driven, not just playing back a pre-determined answer.

**2. Single source of truth for LLM pricing.** Model pricing lives in exactly one place (`observability/logger.py`). The batch processor imports it rather than duplicating constants. When OpenAI changes prices, one file changes.

**3. Topic excluded from the transcript it's asked to classify.** Including the ground-truth topic in the LLM input would make the classification circular — the model would just copy it. Removing it means the classification is independently derived from the conversation content.

**4. Null handling in composite scores.** `F.lit(0.0)` not `F.lit(0.5)` as the null default. Missing data should pull scores down, not prop them up to the middle of the range.

**5. LangGraph over AgentExecutor.** LangChain's `AgentExecutor` was removed in LangChain 1.x. `langgraph.prebuilt.create_react_agent` is the current standard — cleaner invocation, better observability into the reasoning loop.

---

## What's Next

The natural extensions to this project:

- **Athena integration** — replace the pandas CSV reads in the LangChain tools with Athena queries over the Parquet data lake, enabling time-range filtering over the full dataset
- **LLM result feedback loop** — use the LLM-enriched `llm_churn_risk` and `llm_sentiment` fields as features in a customer churn prediction model
- **Real-time streaming** — replace the daily batch with a Kinesis stream processor for near-real-time enrichment of live chatbot sessions
- **Multi-turn agent memory** — add `MemorySaver` to the LangGraph agent so users can ask follow-up questions within a session context

---

## Final Thoughts

What I wanted to demonstrate with this project isn't any single technology — it's that modern data engineering and AI can be combined into a coherent production architecture without sacrificing rigour.

The pipeline has honest data (no fixed seeds), honest metrics (no artificially inflated null defaults), honest LLM outputs (no circular classification), and honest cost tracking (to the cent). Security and observability are built-in, not bolted on.

Most importantly: a product manager can open the dashboard, type a question in plain English, and get a real answer backed by real data — without waiting for a data team.

That's the goal.

---

**GitHub:** [github.com/Ilamparithi-Elango/aep-customer-intelligence-platform](https://github.com/Ilamparithi-Elango/aep-customer-intelligence-platform)

---

*If this was useful, follow for more end-to-end AI and data engineering projects. Happy to answer questions in the comments.*
