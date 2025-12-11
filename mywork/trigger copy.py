"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py

# ============================================================
# 0. My Note
# ============================================================
original code refer to 2_multi_agent, efficient.py

Multi-agent example with a Trigger Agent + Orchestrator + Search worker.

Flow:
User → Trigger Agent (classify & extract intent) → Orchestrator (Main Agent)
     → SearchAgent (and future agents like DataMiner / Librarian / Analyst)
"""

import asyncio
import contextlib
import json
import signal
import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

import agents
import gradio as gr

from datasets import load_dataset
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client

# ============================================================
#  dataset summary (mj44442022/dataset_synthetic_v2)
# ============================================================

BANK_DATASET_ID = "mj44442022/dataset_synthetic_v2"


def build_dataset_summary(dataset_id: str, max_rows: int = 100) -> str:
    """
    Load a small slice of the HF dataset and turn it into a text summary
    we can feed into the agent as context.

    We only keep up to `max_rows` rows and only string columns to save tokens.
    """
    ds = load_dataset(dataset_id)

    # choose a split
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split = ds[split_name]

    num_rows_total = len(split)
    columns = list(split.features.keys())

    # build context from first N rows
    lines: list[str] = []
    for i in range(min(max_rows, num_rows_total)):
        row = split[i]
        parts = []
        for k, v in row.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
        if parts:
            lines.append(" | ".join(parts))

    context = "\n".join(lines)

    summary = (
        f"DATASET_ID: {dataset_id}\n"
        f"TOTAL_ROWS: {num_rows_total}\n"
        f"COLUMNS: {', '.join(columns)}\n"
        f"SAMPLED_ROWS (truncated):\n{context}"
    )
    return summary


BANK_DATASET_SUMMARY = build_dataset_summary(BANK_DATASET_ID, max_rows=200)



load_dotenv(verbose=True)

set_up_logging()

AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}

configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# ============================================================
# 1. Trigger Agent 
# ============================================================

TRIGGER_SYSTEM_PROMPT = """
You are the Trigger Agent for a banking analytics assistant.

Your job:
- Read the user's question about bank dashboards / clients / products.
- Classify the question into an INTENT type.
- Extract key SLOTS so downstream agents can use them.

You MUST respond with a single JSON object, no extra text.

JSON schema (keys):

- "intent": one of
    ["quick_metric_lookup", "definition", "trend",
     "anomaly_investigation", "segment_behavior",
     "deep_research", "other"]

- "metric": short string for the main metric if mentioned
    (e.g. "unique_clients", "active_clients", "new_credit_card_accounts",
     "client_balance", "segment_performance", "churn_rate").
    If unclear, set to null.

- "time_range": description of time period, e.g.
    "this_month", "last_month", "2024-09_to_2024-11",
    "since_2023", or null if not mentioned.

- "product_or_area": product / domain if mentioned, e.g.
    "credit_card", "savings", "mortgage", "xyz_dashboard", or null.

- "segment": client segment if mentioned, e.g.
    "Segment A", "newcomer", "affluent", "active_cc_client", or null.

- "needs_context": boolean.
    true  -> business user needs extra explanation / context
    false -> a single fact or metric answer is enough.

- "complexity": "simple" or "complex".
    "simple"  -> like Regular Mode quick questions
    "complex" -> like Research Mode multi-step questions.

If the question is not related to banking analytics, set:
- "intent": "other"
and you may leave other fields null.
""".strip()


async def run_trigger_agent(question: str) -> dict:
    """
    Call the LLM once to get structured intent JSON for the user's question.
    This is your Trigger Agent.
    """
    resp = await async_openai_client.chat.completions.create(
        model=AGENT_LLM_NAMES["planner"],   # you can switch to worker if needed
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": TRIGGER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    content = resp.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap raw content
        parsed = {
            "intent": "other",
            "metric": None,
            "time_range": None,
            "product_or_area": None,
            "segment": None,
            "needs_context": True,
            "complexity": "simple",
            "raw_trigger_output": content,
        }
    return parsed

# ============================================================
# 2. Worker Agent: Search / Data Miner
# ============================================================

# Worker Agent: handles long context efficiently
search_agent = agents.Agent(
    name="SearchAgent",
    instructions=(
        "You are a search agent. You receive a single search query as input. "
        "Use the search tool to perform a search, then produce a concise "
        "'search summary' of the key findings. Do NOT return raw search results."
    ),
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)


# ============================================================
# 3. Orchestrator (Main Agent) – uses Trigger Agent output
# ============================================================
MAIN_AGENT_INSTRUCTIONS = f"""
SYSTEM INSTRUCTIONS (MAIN ORCHESTRATOR):

You are the Orchestrator Agent in a multi-agent banking-analytics system.

You MUST:
- Use the parsed intent from the Trigger Agent to decide if a question
  is a quick lookup or a deeper investigation.
- When appropriate, call tools (like SearchAgent) to gather information.
- Answer in clear, business-friendly language.
IMPORTANT:
- Answer only the current user question.
- Do NOT restate previous questions.
- Do NOT repeat the full list of columns unless the user asks for them.

You have access to a synthetic banking-like dataset. Use it as
background context when answering questions, but do not pretend it is
live production data.

If the user asks for specific metrics like "active clients", "balances",
"segment A", etc.:
- First check whether those concepts exist in the dataset summary.
- If they do not exist, say clearly that the synthetic dataset does not
  contain that metric or column.
- Do NOT say you are a large language model.
- Do NOT say you cannot access internal systems.
- Do NOT talk about your general training data.

---------------------------------------
SYNTHETIC DATASET SUMMARY
(from Hugging Face: {BANK_DATASET_ID})
---------------------------------------

{BANK_DATASET_SUMMARY}

---------------------------------------
END OF DATASET
---------------------------------------

Now follow the REACT-style reasoning instructions below to plan your
steps and, if useful, call tools such as the SearchAgent:

{REACT_INSTRUCTIONS}
""".strip()



# Main Agent: more expensive and slower, but better at complex planning
main_agent = agents.Agent(
    name="MainAgent",
    instructions=MAIN_AGENT_INSTRUCTIONS,
    # Allow the planner agent to invoke the worker agent.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        search_agent.as_tool(
            tool_name="search",
            tool_description="Perform a web search for a query and return a concise summary.",
        )
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
    ),
)

# ============================================================
# 4. Gradio entry point (stateless / no accumulated history)
# ============================================================

async def _main(question: str, history: list[ChatMessage]):
    """
    Gradio ChatInterface handler (stateless version).

    For every new user question, we:
      - run Trigger Agent
      - run Main Agent (Orchestrator)
      - return ONLY this turn's user question + assistant answer

    Old turns are not kept in the UI, so answers will not look
    "accumulated" or repeated across questions.
    """

    setup_langfuse_tracer()

    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        # -------- Trigger Agent step --------
        trigger_result = await run_trigger_agent(question)
        span.update(trigger_result=trigger_result)

        orchestrator_input = {
            "original_question": question,
            "parsed_intent": trigger_result,
        }
        orchestrator_input_str = json.dumps(orchestrator_input, indent=2)

        # -------- Orchestrator (Main Agent) step --------
        result_stream = agents.Runner.run_streamed(
            main_agent,
            input=orchestrator_input_str,
        )

        # Consume the stream, don't push partials to UI
        async for _item in result_stream.stream_events():
            pass

        final_answer = result_stream.final_output
        span.update(output=final_answer)

        # ❗ Stateless: build a brand-new history each time
        new_history: list[ChatMessage] = [
            ChatMessage(role="user", content=question),
            ChatMessage(role="assistant", content=final_answer),
        ]

        # Return only this Q&A pair
        yield new_history


demo = gr.ChatInterface(
    _main,
    title="Scotiabank-Two: Actionable Insights Engine (Trigger + Orchestrator)",
    type="messages",
    examples=[
        "How many active clients made at least one transaction this month?",
        "Segment A's balances dropped 30% since October. What might be going on?",
        "What columns are available in the dataset?",
        "How many active clients did we have last month?",
        "Why did Segment A balances drop in the synthetic dataset?",
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
