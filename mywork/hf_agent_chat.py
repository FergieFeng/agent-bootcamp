"""
Simple HF-dataset-backed chatbot using an OpenAI-compatible API and Gradio.

- Reads OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL from .env
- Loads Hugging Face dataset: mj44442022/dataset_synthetic_v2
- Uses Gradio ChatInterface for UI
"""

import os
import traceback

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

# -------------------------------------------------------
# 1. Load environment variables
# -------------------------------------------------------

load_dotenv(verbose=True)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # e.g. Gemini proxy
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")  # default if not set

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

print(f"[INFO] Using base_url={OPENAI_BASE_URL!r}, model={OPENAI_MODEL!r}")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,  # can be None if using normal OpenAI endpoint
)

# -------------------------------------------------------
# 2. Build context from your HF dataset
# -------------------------------------------------------

HF_DATASET_ID = "mj44442022/dataset_synthetic_v2"


def load_dataset_and_summary(dataset_id: str, max_rows: int = 100):
    print(f"[INFO] Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id)

    # Pick a split
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split = ds[split_name]
    print(f"[INFO] Using split: {split_name}, rows: {len(split)}")

    num_rows_total = len(split)
    columns = list(split.features.keys())

    # Build context from first max_rows rows
    num_rows_context = min(max_rows, num_rows_total)
    lines = []
    for i in range(num_rows_context):
        row = split[i]
        parts = []
        for k, v in row.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
        if parts:
            lines.append(" | ".join(parts))

    context = "\n".join(lines)
    print(f"[INFO] Context built from {num_rows_context} rows, length={len(context)} chars")

    return context, num_rows_total, columns


DATASET_CONTEXT, DATASET_N_ROWS, DATASET_COLUMNS = load_dataset_and_summary(
    HF_DATASET_ID, max_rows=200
)


# -------------------------------------------------------
# 3. Chat logic
# -------------------------------------------------------


def chat_fn(message, history):
    """
    Gradio ChatInterface handler.

    message: latest user message (string)
    history: list of [user, bot] pairs
    """

    system_prompt = (
    "You are a helpful assistant that uses the following dataset as background "
    "context. Use it when it's relevant. If the dataset does not clearly support "
    "an answer, say you are not sure instead of making things up.\n\n"
    f"DATASET SUMMARY:\n"
    f"- Total rows (records): {DATASET_N_ROWS}\n"
    f"- Columns: {', '.join(DATASET_COLUMNS)}\n\n"
    f"SAMPLED DATA (truncated):\n{DATASET_CONTEXT[:6000]}"
    )


    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    # Add current user question
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        reply = response.choices[0].message.content
        return reply
    except Exception as e:
        # Print full traceback in the terminal
        print("[ERROR] API call failed:")
        traceback.print_exc()

        # Also show the error message in the UI so you see what's wrong
        return f"Backend error: {repr(e)}"


# -------------------------------------------------------
# 4. Gradio UI
# -------------------------------------------------------

demo = gr.ChatInterface(
    fn=chat_fn,
    title="HF Dataset Chatbot (OpenAI-compatible + Gradio)",
    description="Asks questions using mj44442022/dataset_synthetic_v2 as background context.",
)

if __name__ == "__main__":
    demo.launch(share=True)
