# LinkedIn Post — AEP Customer Intelligence Platform

---

Adobe Summit last week declared we've entered the **Agentic Enterprise** era.

I decided to build one.

---

Inspired by Adobe's vision of AI agents orchestrating real customer experiences, I spent the past week building the **AEP Customer Intelligence Platform** — an end-to-end AWS-native pipeline that turns raw chatbot logs into live business intelligence, served through a conversational AI agent.

Here's what it does:

🔗 **3 data sources unified** — chatbot interactions, clickstream events, session metadata joined with PySpark into a single Parquet data lake on AWS S3

🤖 **LLM enrichment at scale** — GPT-4o-mini processes every record concurrently, extracting sentiment, conversation summaries, topic classifications, and churn risk signals

🧠 **Agentic AI layer** — a LangGraph ReAct agent answers plain-English business questions over live data using custom tools. No SQL. No dashboards. Just ask.

📊 **Streamlit dashboard** — KPI cards, trend charts, topic breakdowns, and an AI chat panel — all filterable in real time

☁️ **Production-grade AWS stack** — Airflow DAG orchestrates 7 pipeline steps daily, S3 with AES-256 encryption, structured JSON logs compatible with CloudWatch, LLM cost tracked to the cent

🔒 **Security built in** — prompt injection detection, PII masking in all log output, input sanitization before every agent call

A product manager can open the dashboard, type:

*"Which topics have the highest escalation rate for Healthcare this week?"*

And get a real, data-backed answer in seconds — no data team, no ticket, no waiting.

---

Adobe Summit also shared that **AI traffic to US retail sites grew 269% year-over-year in March 2026.**

The demand for engineers who understand the full agentic AI stack — not just the model layer, but data pipelines, orchestration, security, and observability — is only going one direction.

---

Full code on GitHub 👉 https://github.com/Ilamparithi-Elango/aep-customer-intelligence-platform
Full walkthrough article 👉 [Medium link]

If you're building in agentic AI, hiring for it, or just curious about how the pieces fit together — let's connect.

#AgenticAI #AdobeSummit #DataEngineering #LLMEngineering #LangGraph #AWS #OpenAI #Python #ArtificialIntelligence #MachineLearning #CustomerExperience #AdobeExperienceCloud

---

## 💡 Tips before posting:

1. **Add the architecture diagram as the image** — upload `docs/AEP_Customer_Intelligence_platform.png` directly to the LinkedIn post (not a link). Visuals dramatically increase reach.
2. **Replace [Medium link]** with the actual Medium article URL once it's properly formatted.
3. **Post on a Tuesday or Wednesday morning** (8–10am your timezone) — highest engagement window on LinkedIn.
4. **First comment** — immediately after posting, add a comment with the GitHub link again. LinkedIn's algorithm treats links in the first comment better than links in the post body.
5. **Tag Adobe** (@Adobe) in the post — increases visibility to people following Adobe content.
