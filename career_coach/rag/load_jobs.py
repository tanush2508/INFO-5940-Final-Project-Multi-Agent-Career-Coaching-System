# career_coach/rag/load_jobs.py

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
BASE_URL = "https://api.openwebninja.com/jsearch/search"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "jobs_raw.json"
CLEAN_PATH = DATA_DIR / "jobs_clean.json"


def _fetch_jobs_from_jsearch(
    queries: List[str],
    pages_per_query: int = 1,
) -> List[Dict[str, Any]]:
    """
    Call JSearch API and return a flat list of job objects.

    We keep this simple:
      - use your x-api-key
      - send query, page, country, etc.
      - DO NOT pass 'fields' so we get full objects including job_id.
    """
    if not JSEARCH_API_KEY:
        raise RuntimeError("JSEARCH_API_KEY is not set in .env")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_jobs: List[Dict[str, Any]] = []

    headers = {
        "x-api-key": JSEARCH_API_KEY,
        "Accept": "*/*",
    }

    for q in queries:
        for page in range(1, pages_per_query + 1):
            params = {
                "query": q,
                "page": page,
                "num_pages": 1,
                "country": "us",
                "language": "en",
                "date_posted": "month",     # keep it reasonably fresh
                "work_from_home": "false",  # "true" / "false" as strings
                "employment_types": "FULLTIME",
                "job_requirements": "no_experience",
                "radius": 25,
                # NO 'fields' param so we get full job objects (including job_id)
            }

            print(f"[JSEARCH] Fetching page {page} for query='{q}'")
            resp = requests.get(
                BASE_URL,
                headers=headers,
                params=params,
                timeout=30,
            )

            # If something goes wrong, show it loudly
            try:
                resp.raise_for_status()
            except Exception as e:
                print(f"[JSEARCH] Error for query='{q}', page={page}: {e}")
                print("Response text:", resp.text[:500])
                continue

            data = resp.json()
            items = data.get("data")
            if items is None and isinstance(data, list):
                items = data

            count = len(items or [])
            print(f"[JSEARCH] Got {count} items for query='{q}', page={page}")

            if not items:
                continue

            all_jobs.extend(items)

    print(f"[JSEARCH] Total raw jobs fetched: {len(all_jobs)}")
    return all_jobs


def _clean_jobs(raw_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw JSearch jobs into a simpler schema for our app and RAG.
    """
    cleaned: List[Dict[str, Any]] = []

    for j in raw_jobs:
        job_id = j.get("job_id")
        if not job_id:
            # If there is no job_id, skip it (we need stable IDs)
            continue

        title = j.get("job_title") or ""
        company = j.get("employer_name") or ""
        city = j.get("job_city") or ""
        state = j.get("job_state") or ""
        country = j.get("job_country") or ""

        location_parts = [p for p in [city, state, country] if p]
        location = ", ".join(location_parts) if location_parts else ""

        description = j.get("job_description") or ""
        employment_type = j.get("job_employment_type") or ""
        publisher = j.get("job_publisher") or ""

        cleaned.append(
            {
                "job_id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "employment_type": employment_type,
                "publisher": publisher,
                "description": description,
            }
        )

    print(f"[JSEARCH] Cleaned jobs: {len(cleaned)}")
    return cleaned


def load_and_clean_jobs(
    force_refresh: bool = False,
    queries: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Main entry point used by the rest of the app.

    - If jobs_clean.json exists and force_refresh=False -> load and return it.
    - Otherwise:
        * fetch from JSearch (using given `queries` if provided,
          or default student/early-career tech queries)
        * write jobs_raw.json and jobs_clean.json
        * return cleaned list

    This is what `job_matcher_node` now calls with resume-driven queries.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_PATH.exists() and not force_refresh:
        try:
            data = json.loads(CLEAN_PATH.read_text(encoding="utf-8"))
            print(f"[load_jobs] Loaded {len(data)} jobs from existing jobs_clean.json")
            return data
        except Exception as e:
            print("[load_jobs] Failed to read existing jobs_clean.json:", e)

    if queries is None or len(queries) == 0:
        # Fallback queries tuned for students / early-career tech roles
        queries = [
            "software engineer internship in New York",
            "data science internship in New York",
            "machine learning engineer entry level in New York",
            "backend developer internship remote",
            "ai engineer entry level remote",
        ]
        print("[load_jobs] Using default queries:", queries)
    else:
        print("[load_jobs] Using custom queries from resume:", queries)

    raw_jobs = _fetch_jobs_from_jsearch(queries, pages_per_query=1)

    # Save raw
    RAW_PATH.write_text(json.dumps(raw_jobs, indent=2), encoding="utf-8")
    print(f"[load_jobs] Saved raw jobs to {RAW_PATH}")

    # Clean and save
    cleaned = _clean_jobs(raw_jobs)
    CLEAN_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    print(f"[load_jobs] Saved cleaned jobs to {CLEAN_PATH}")

    return cleaned


# Optional CLI usage: still works if you run `python -m rag.load_jobs`
if __name__ == "__main__":
    jobs = load_and_clean_jobs(force_refresh=True)
    print(f"[load_jobs] Final count: {len(jobs)} cleaned jobs")
