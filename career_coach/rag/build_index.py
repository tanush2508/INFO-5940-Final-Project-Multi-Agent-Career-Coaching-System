from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from .load_jobs import load_and_clean_jobs

load_dotenv()  # load OPENAI_API_KEY


def build_index():
    jobs = load_and_clean_jobs()
    if not jobs:
        raise ValueError("No jobs found to index. Fill data/jobs_raw.csv first.")

    client = chromadb.PersistentClient(path=str(Path("chroma_store")))
    collection = client.get_or_create_collection("jobs")

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [
        job.description + "\n\n" + job.requirements
        for job in jobs
    ]
    ids = [job.id for job in jobs]
    metadatas = [
        {
            "title": job.title,
            "company": job.company,
            "location": job.location,
        }
        for job in jobs
    ]

    print(f"Embedding {len(jobs)} job postings...")
    embeddings = embedder.embed_documents(texts)

    collection.delete(where={})
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
    )
    print("Index built successfully.")


if __name__ == "__main__":
    build_index()
