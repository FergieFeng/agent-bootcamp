"""
Dataset utilities for the synthetic banking dataset
mj44442022/dataset_synthetic_v2.

Provides:
- Cached loading of the HF dataset
- Column list
- Segment list
- Simple "active client" definition and counts
- Text summary used in the Orchestrator prompt
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from datasets import load_dataset
import pandas as pd


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

BANK_DATASET_ID = "mj44442022/dataset_synthetic_v2"


# ------------------------------------------------------------
# Loading helpers (cached)
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_hf_split():
    """
    Load the dataset split once and cache it.

    Returns:
        A HuggingFace Dataset object for the main split.
    """
    ds_dict = load_dataset(BANK_DATASET_ID)
    split_name = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
    return ds_dict[split_name]


@lru_cache(maxsize=1)
def _load_pandas() -> pd.DataFrame:
    """
    Convert the HF split to a pandas DataFrame (also cached).
    """
    split = _load_hf_split()
    return split.to_pandas()


# ------------------------------------------------------------
# Public utilities (called by tools / prompts)
# ------------------------------------------------------------

def get_dataset_columns() -> List[str]:
    """
    Return the list of columns in the synthetic dataset.
    """
    split = _load_hf_split()
    return list(split.features.keys())


def get_segments() -> List[str]:
    """
    Return the unique segment values (e.g., 'High Value Peru').
    """
    df = _load_pandas()
    return sorted(df["segment"].dropna().unique().tolist())


def count_active_clients_last_month() -> Dict[str, object]:
    """
    Example definition of 'active client':
        purchase_count_last_month > 0 AND is_churned == 0.

    Returns:
        A dictionary with counts and share.
    """
    df = _load_pandas()

    active_mask = (df["purchase_count_last_month"] > 0) & (df["is_churned"] == 0)
    active_count = int(active_mask.sum())
    total = int(len(df))

    return {
        "definition": (
            "active client = purchase_count_last_month > 0 "
            "AND is_churned == 0"
        ),
        "active_clients_last_month": active_count,
        "total_clients": total,
        "share_active": active_count / total if total > 0 else 0.0,
    }


def build_dataset_summary(max_rows: int = 100) -> str:
    """
    Build a text summary of the dataset for the Orchestrator prompt.

    We only keep up to `max_rows` rows and only string columns to save tokens.
    """
    split = _load_hf_split()

    num_rows_total = len(split)
    columns = list(split.features.keys())

    # build context from first N rows
    lines: List[str] = []
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
        f"DATASET_ID: {BANK_DATASET_ID}\n"
        f"TOTAL_ROWS: {num_rows_total}\n"
        f"COLUMNS: {', '.join(columns)}\n"
        f"SAMPLED_ROWS (truncated):\n{context}"
    )
    return summary
