"""
Knowledge Base utilities for retrieval and search.

This module provides:
- AsyncWeaviateKnowledgeBase: a wrapper around Weaviate for async search
- clean_query: removes noise from queries
- prepare_weaviate_schema: ensures schema exists (optional)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import re
import weaviate
import weaviate.classes as wvc


# ================================================================
# 1. Utility: clean the query text
# ================================================================

def clean_query(query: str) -> str:
    """
    Normalize query strings before sending to Weaviate.
    Removes repeated spaces, lowercases, strips noise.
    """
    if not query:
        return ""
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    return q


# ================================================================
# 2. Async Knowledge Base Wrapper
# ================================================================

class AsyncWeaviateKnowledgeBase:
    """
    Light wrapper around an async Weaviate client.
    Handles:
        - vector search
        - BM25 search
        - hybrid search
    """

    def __init__(
        self,
        async_client: weaviate.WeaviateAsyncClient,
        collection_name: str,
    ):
        self.client = async_client
        self.collection_name = collection_name

    # ------------------------------------------------------------

    async def search_knowledgebase(
        self,
        query: str,
        limit: int = 5,
        with_vectors: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point called by the SearchAgent.

        Parameters:
            query       : user query string
            limit       : number of retrieved objects
            with_vectors: include vector data or not

        Returns:
            A dictionary with:
                - "query": cleaned query
                - "results": list of retrieved objects
        """
        cleaned = clean_query(query)

        try:
            resp = await self.client.collections.get(self.collection_name).query.hybrid(
                query=cleaned,
                limit=limit,
            )
        except Exception as e:
            return {
                "query": cleaned,
                "results": [],
                "error": f"Weaviate hybrid search failed: {e}",
            }

        results = []
        for obj in resp.objects:
            entry = {
                "properties": obj.properties,
            }
            if with_vectors and obj.vector:
                entry["vector"] = obj.vector
            results.append(entry)

        return {
            "query": cleaned,
            "results": results,
        }

    # ------------------------------------------------------------

    async def insert_batch(self, items: List[Dict[str, Any]]):
        """
        Optional utility to load dataset examples into Weaviate.

        items: list of dictionaries mapping to schema properties.
        """
        coll = self.client.collections.get(self.collection_name)

        try:
            async with coll.batch.dynamic() as batch:
                for item in items:
                    await batch.add_object(properties=item)
        except Exception as e:
            raise RuntimeError(f"Batch insert failed: {e}")

    # ------------------------------------------------------------

    async def ensure_schema(self, properties: Optional[Dict[str, Any]] = None):
        """
        Optional helper: create the schema for a dataset collection if missing.

        properties example:
            {
                "text": wvc.DataType.TEXT,
                "title": wvc.DataType.TEXT,
            }
        """
        try:
            existing_collections = await self.client.collections.list_all()
            names = [c.name for c in existing_collections]

            if self.collection_name in names:
                return  # already exists

            props = []
            if properties:
                for name, dtype in properties.items():
                    props.append(wvc.Property(name=name, data_type=dtype))

            await self.client.collections.create(
                name=self.collection_name,
                properties=props,
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
            )

        except Exception as e:
            raise RuntimeError(f"Schema creation failed: {e}")
