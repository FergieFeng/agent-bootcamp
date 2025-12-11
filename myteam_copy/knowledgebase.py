"""
Knowledge Base utilities for retrieval and search.

This module provides:
- async_knowledgebase: shared AsyncWeaviateKnowledgeBase instance
- clean_query: small helper to normalize queries
- rag_search: simple RAG-style helper that searches Weaviate and returns a summary
"""

from __future__ import annotations

from typing import Any, Dict, List

import re

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
)

# ================================================================
# 1. Utility: clean the query text
# ================================================================


def clean_query(query: str) -> str:
    """
    Normalize query strings before sending to Weaviate.
    Removes repeated spaces, trims, etc.
    """
    if not query:
        return ""
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    return q


# ================================================================
# 2. Build shared async Weaviate knowledge base
# ================================================================

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

# This is the same collection name used elsewhere in the bootcamp code.
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)


# ================================================================
# 3. Simple RAG-style helper
# ================================================================


async def rag_search(query: str, limit: int = 5) -> str:
    """
    Simple RAG-style helper: search the internal knowledge base
    and return a short, readable summary string.

    This is what your Main Agent will call via the `rag_search` tool.
    """
    cleaned = clean_query(query)

    result_dict: Dict[str, Any] = await async_knowledgebase.search_knowledgebase(
        query=cleaned,
        limit=limit,
    )

    results: List[Dict[str, Any]] = result_dict.get("results", [])

    if not results:
        return (
            f"No relevant documents found in the bank knowledge base for query: {cleaned!r}."
        )

    bullets: List[str] = []
    for item in results:
        props = item.get("properties", {}) or {}
        # Try common property names used in the Weaviate KB
        text = (
            props.get("text")
            or props.get("content")
            or props.get("body")
            or str(props)
        )
        text = str(text)
        # Truncate to avoid huge outputs
        bullets.append(f"- {text[:300]}")

    joined = "\n".join(bullets)

    return (
        f"Here are key points from the internal bank knowledge base related to '{cleaned}':\n\n"
        f"{joined}"
    )
