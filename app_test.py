import os
import gradio as gr
from openai import OpenAI
from datasets import load_dataset

# ---------------------
# 1. SET YOUR OPENAI KEY
# ---------------------
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

client = OpenAI()

# ---------------------
# 2. LOAD YOUR HF DATASET
# ---------------------
ds = load_dataset("mj44442022/dataset_synthetic_v2")
data = ds["train"]

# ---------------------
# 3. SIMPLE HELPER TO GET A RANDOM ROW (example)
# ---------------------
def get_random_client():
    row = data.shuffle(seed=42)[0]
    return row

# ---------------------
# 4. CHAT FUNCTION
# ---------------------
def chat_with_model(user_msg, history):

    # Optionally inject one row from dataset
    sample = get_random_client()

    system_message = (
        "You are a Customer Insight Assistant. "
        "Use the dataset row provided only as context, not as ground truth. "
        f"Dataset sample: {sample}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or any OpenAI model
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_msg},
        ]
    )


    return response.choices[0].message.content

# ---------------------
# 5. GRADIO INTERFACE
# ---------------------
demo = gr.ChatInterface(
    chat_with_model,
    title="Customer Insight Chatbot (Simple Test)",
    description="Powered by OpenAI + Hugging Face Dataset",
)

# ---------------------
# 6. LAUNCH
# ---------------------
if __name__ == "__main__":
    demo.launch(share=True)
