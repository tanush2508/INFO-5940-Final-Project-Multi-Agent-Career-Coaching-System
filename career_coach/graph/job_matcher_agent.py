from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .state import SharedState, JobMatch
from rag.retriever import get_top_jobs

load_dotenv()

_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "job_matcher.md"
    return prompt_path.read_text(encoding="utf-8")


_JOB_MATCH_PROMPT = _load_prompt()


def job_matcher_node(state: SharedState, k: int = 5) -> SharedState:
    """
    Uses resume profile to query RAG and populate state.job_matches.
    Then calls LLM to generate a short rationale for each match.
    """
    if not state.resume_profile:
        return state

    rp = state.resume_profile
    # Simple query: skills + summary
    query = " ".join(rp.skills) + " " + rp.experience_summary

    raw_jobs = get_top_jobs(query, k=k)
    matches: List[JobMatch] = []

    for job in raw_jobs:
        desc_snippet = job["description"][:600]

        messages = [
            {"role": "system", "content": _JOB_MATCH_PROMPT},
            {
                "role": "user",
                "content": (
                    f"RESUME SUMMARY:\n{rp.experience_summary}\n\n"
                    f"SKILLS: {', '.join(rp.skills)}\n\n"
                    f"JOB:\nTitle: {job['title']}\n"
                    f"Company: {job['company']}\n"
                    f"Location: {job['location']}\n"
                    f"Description:\n{desc_snippet}"
                ),
            },
        ]

        rationale_resp = _llm.invoke(messages)
        rationale_text = (
            rationale_resp.content
            if isinstance(rationale_resp.content, str)
            else str(rationale_resp.content)
        )

        matches.append(
            JobMatch(
                job_id=job["job_id"],
                score=job["score"],
                rationale=rationale_text.strip(),
                title=job.get("title"),
                company=job.get("company"),
                location=job.get("location"),
            )
        )

    state.job_matches = matches
    return state
