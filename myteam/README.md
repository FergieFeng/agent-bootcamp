           ┌──────────────────────┐
           │ User Question         │
           └───────────────┬──────┘
                           ▼
                  ┌──────────────────┐
                  │ Trigger Agent     │
                  │ (intent JSON)     │
                  └───────┬──────────┘
        ┌───────────────────────────────┬──────────────────────────┐
        ▼                               ▼                          ▼
┌──────────────────┐         ┌──────────────────┐       ┌────────────────────┐
│ MJ_BI Engine      │         │ MJ_SCOTIA Engine │       │ Orchestrator Agent │
│ (fast insights)   │         │ (deep analysis)  │       │ (fallback LLM)     │
└───────────────────┘         └──────────────────┘       └────────────────────┘
        ▼                               ▼                          ▼
                                (result returned to UI)

---
trigger_connect.py — Main app entry; routes questions → MJ_BI / MJ_SCOTIA / Orchestrator; runs Gradio UI.
info_retrieval_prompt_intent.py — from Jessie; Trigger Agent prompt; classifies intent (data_insight, deep_analysis, etc.).
MJ_BI.py — from MJ; Fast BI engine using Pandas; simple insights (segment revenue, averages, group-by).
MJ_SCOTIA.py — from MJ; Deep multi-metric analysis engine; executes generated code; heavy insights.
knowledgebase.py — Placeholder RAG/KB interface (future expansion).
dataset_tools.py — Utility functions for loading, summarizing, and analyzing banking dataset.
dataset_definitions.py — Metadata definitions for dataset schema and column descriptions.
Retrival_UI_test.py — from Jessie; use to test UI for experimenting with retrieval agent separately; not used. 
trigger_connect copy.py — Backup of old main file; not used.
---

# System Architecture – File Responsibilities and Connections

## 1. `trigger_connect.py` – Main Orchestrator
- Entry point for the entire system.
- Routes each question to the correct engine (Trigger Agent → MJ_BI or MJ_SCOTIA).
- Manages stateless Gradio chat interface.
- Connects all agents, tools, and prompts.

**Depends on:**
- `info_retrieval_prompt_intent.py`
- `MJ_BI.py`
- `MJ_SCOTIA.py`
- `dataset_tools.py`
- `knowledgebase.py` (optional)

---

## 2. `info_retrieval_prompt_intent.py` – Trigger Agent
- Classifies user questions into intents.
- Extracts slots (metric, time_range, segment, product).
- Decides whether the question should go to BI mode or Deep Analysis.

**Used by:**
- `trigger_connect.py`

---

## 3. `dataset_definitions.py` – Column Metadata
- Stores column descriptions and metadata for all dataset fields.
- Provides lookup functions to retrieve column definitions.

**Used by:**
- `dataset_tools.py`
- `MJ_BI.py`
- `MJ_SCOTIA.py`

---

## 4. `dataset_tools.py` – Dataset Loader & Utilities
- Loads the synthetic banking dataset from HuggingFace.
- Provides helper utilities:
  - `get_dataset_columns()`
  - `count_active_clients_last_month()`
  - `build_dataset_summary()`

**Used by:**
- `trigger_connect.py`
- `MJ_BI.py`
- `MJ_SCOTIA.py`

---

## 5. `MJ_BI.py` – Fast BI Analysis Engine
- Handles simple and mid-level analytical questions.
- Uses prebuilt pandas operations (groupby, aggregate, summary).
- Returns quick business insights without code generation.

**Triggered when:**
- Intent is `quick_metric_lookup`, `data_insight`, or `segment_behavior`.

**Used by:**
- `trigger_connect.py`

---

## 6. `MJ_SCOTIA.py` – Deep Reasoning Analysis Engine
- Full 5-layer intelligent reasoning system:
  1. Understand
  2. Plan
  3. Generate Python code
  4. Execute safely
  5. Interpret insights
- Produces detailed multi-step analytical answers.

**Triggered when:**
- User says “deep analysis”
- or Trigger Agent marks question as complex.

**Used by:**
- `trigger_connect.py`

---

## 7. `knowledgebase.py` – Knowledge Retrieval Layer (Optional)
- Semantic search and retrieval helper functions.
- Supports contextual lookup and external reference reasoning.

**Used by:**
- `trigger_connect.py` (optional)

---

## 8. `Retrival_UI_test.py` – Standalone Retrieval UI
- Interface for independently testing the retrieval agent.
- Not part of the main system pipeline.

---

## 9. `trigger_connect copy.py` – Backup Version
- Previous version of the orchestrator file.
- Stored for reference; not used by the system.

---
# 🔌 Tool Integration

All analytical tools and teammate modules are wired together inside  
**`trigger_connect.py`**, where the orchestrator loads and connects every component.

### Imported Tools
```python
from dataset_definitions import get_column_definition
from dataset_tools import get_dataset_columns, count_active_clients_last_month
from MJ_BI import analyze_business_insight
from MJ_SCOTIA import run_mj_scotia
```

### Registered Tool List
```python
tools = [
    search_agent.as_tool(...),     # Knowledgebase / RAG search
    dataset_columns_tool,          # Returns dataset column names
    active_clients_tool,           # Metric: active clients (last month)
    column_definition_tool,        # Lookup column meaning from metadata
]
```

### Purpose
- Central place where **all agents and utilities are linked**.
- Ensures Trigger Agent → BI Engine → Deep Analysis Engine all share the same toolset.
- Makes the orchestrator modular: new tools can be added by a single line import + registration.
