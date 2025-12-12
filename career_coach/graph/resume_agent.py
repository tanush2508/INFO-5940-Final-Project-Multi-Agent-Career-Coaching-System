import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .state import SharedState, ResumeProfile

load_dotenv()

_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "resume_analyzer.md"
    return prompt_path.read_text(encoding="utf-8")


_RESUME_PROMPT = _load_prompt()


def resume_analyzer_node(state: SharedState) -> SharedState:
    """
    Takes state.resume_profile.raw_text and fills in skills, summary, etc.
    """
    if not state.resume_profile or not state.resume_profile.raw_text.strip():
        return state

    resume_text = state.resume_profile.raw_text

    messages = [
        {"role": "system", "content": _RESUME_PROMPT},
        {"role": "user", "content": resume_text},
    ]

    resp = _llm.invoke(messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try crude JSON extraction if the model wraps it in text
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])

    state.resume_profile = ResumeProfile(
        raw_text=resume_text,
        skills=data.get("skills", []),
        experience_summary=data.get("experience_summary", ""),
        years_experience=data.get("years_experience"),
        suggestions=data.get("suggestions", []),
    )

    return state
