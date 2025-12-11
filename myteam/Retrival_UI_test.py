"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

import asyncio
import contextlib
import logging
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.info_retrieval_prompt_copy import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
)

# Load environment variables (Weaviate, OpenAI keys, etc.)
load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO)

# LLM to use for the agent (keep in sync with your bootcamp setup)
AGENT_LLM_NAME = "gemini-2.5-flash"

# -----------------------------------------------------------------------------
# Clients & Knowledge Base
# -----------------------------------------------------------------------------

configs = Configs.from_env_var()

# NOTE: using the *_co_* host/keys that point to your competitor-offers Weaviate
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
    collection_name="bns",
)

# -----------------------------------------------------------------------------
# Graceful shutdown helpers
# -----------------------------------------------------------------------------

async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shut down."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# -----------------------------------------------------------------------------
# Main Gradio handler
# -----------------------------------------------------------------------------

async def _main(
    question: str,
    gr_messages: list[ChatMessage],
    intent: str,  # from dropdown
):
    """Gradio callback: run the ReAct-RAG competitor offer agent."""
    # Define the Information Retrieval Agent
    main_agent = agents.Agent(
        name="Competitor Offers Information Retrieval Agent",
        instructions=REACT_INSTRUCTIONS,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

    # ⬇️ Encode intent into the text so input stays a string
    tagged_question = f"[INTENT: {intent}] {question}"

    # Stream the result to Gradio UI
    # IMPORTANT: input must be str or list, NOT dict
    result_stream = agents.Runner.run_streamed(main_agent, input=tagged_question)

    async for _item in result_stream.stream_events():
        gr_messages += oai_agent_stream_to_gradio_messages(_item)
        if len(gr_messages) > 0:
            yield gr_messages

# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

intent_dropdown = gr.Dropdown(
    choices=["data point", "metric definition", "insights"],
    label="What are you looking for?",
    value="data point",  # default
)

demo = gr.ChatInterface(
    _main,
    title="Column & Metric Selector – ReAct RAG Agent",
    type="messages",
    # When additional_inputs are used, examples must be list-of-lists:
    # [ user_question, intent_value ]
    examples=[
        ["How is Peru performing in Dec 2024?", "insights"],
        [
            "Compare monthly revenue performance between Canada and Brazil from Dec 2024 to Mar 2025",
            "insights",
        ],
        ["Which region performed best last quarter?", "insights"],
        ["Show the average revenue per client across segment", "metric definition"],
    ],
    additional_inputs=[intent_dropdown],
)

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
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
    async_knowledgebase = AsyncWeaviateKnowledgeBase(
        async_weaviate_client,
        collection_name="bns",
    )

    async_openai_client = AsyncOpenAI()
    agents.set_tracing_disabled(disabled=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())