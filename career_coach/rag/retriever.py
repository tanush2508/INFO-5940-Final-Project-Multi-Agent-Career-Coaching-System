from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

_CHROMA_PATH = Path("chroma_store")
_COLLECTION_NAME = "jobs"


def _get_collection():
    """
    Get or create the Chroma collection used for job postings.
    """
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    return client.get_or_create_collection(_COLLECTION_NAME)


def get_top_jobs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search over the job postings.

    Returns a list of job dicts like:
    {
        "job_id": str,
        "score": float,
        "title": str,
        "company": str,
        "location": str,
        "description": str,
    }
    """
    collection = _get_collection()
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    query_embedding = embedder.embed_query(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    jobs: List[Dict[str, Any]] = []
    # If nothing found, result["ids"] might be empty
    if not result["ids"]:
        return jobs

    ids = result["ids"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]
    documents = result["documents"][0]

    for job_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
        jobs.append(
            {
                "job_id": job_id,
                "score": float(1.0 - dist),  # crude similarity score
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "description": doc,
            }
        )

    return jobs


def get_job_description_by_id(job_id: str) -> Optional[str]:
    """
    Fetch the full description/requirements text for a job id from Chroma.
    """
    collection = _get_collection()
    result = collection.get(ids=[job_id])

    if not result["ids"]:
        return None

    # documents is a list; we stored description+requirements as a single text
    return result["documents"][0]
