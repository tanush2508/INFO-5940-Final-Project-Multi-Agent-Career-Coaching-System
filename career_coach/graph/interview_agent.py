import json
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .state import (
    SharedState,
    InterviewQuestion,
    InterviewFeedback,
)

from rag.retriever import get_job_description_by_id

load_dotenv()

_llm_questions = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
_llm_feedback = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)


def _load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / name
    return prompt_path.read_text(encoding="utf-8")


_QUESTION_PROMPT = _load_prompt("interview_questions.md")
_FEEDBACK_PROMPT = _load_prompt("interview_feedback.md")


def generate_questions_node(state: SharedState) -> SharedState:
    """
    Use selected_job_id and resume_profile to create tailored interview questions.
    """
    if not state.resume_profile or not state.selected_job_id:
        return state

    job_desc = get_job_description_by_id(state.selected_job_id)
    if not job_desc:
        return state

    rp = state.resume_profile

    messages = [
        {"role": "system", "content": _QUESTION_PROMPT},
        {
            "role": "user",
            "content": (
                f"JOB DESCRIPTION:\n{job_desc}\n\n"
                f"CANDIDATE SUMMARY:\n{rp.experience_summary}\n\n"
                f"SKILLS: {', '.join(rp.skills)}"
            ),
        },
    ]

    resp = _llm_questions.invoke(messages)
    content = resp.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])

    qs: List[InterviewQuestion] = []
    for q in data.get("questions", []):
        qs.append(
            InterviewQuestion(
                question=q.get("question", ""),
                dimension=q.get("dimension", ""),
                ideal_answer_notes=q.get("ideal_answer_notes", ""),
            )
        )

    state.interview_questions = qs
    return state


def evaluate_answer_node(state: SharedState, question_index: int, user_answer: str) -> SharedState:
    """
    Evaluate a single answer for question at question_index and append to feedback_history.
    """
    if question_index < 0 or question_index >= len(state.interview_questions):
        return state

    q = state.interview_questions[question_index]

    messages = [
        {"role": "system", "content": _FEEDBACK_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION: {q.question}\n\n"
                f"IDEAL ANSWER NOTES: {q.ideal_answer_notes}\n\n"
                f"CANDIDATE ANSWER:\n{user_answer}"
            ),
        },
    ]

    resp = _llm_feedback.invoke(messages)
    content = resp.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])

    fb = InterviewFeedback(
        question=q.question,
        user_answer=user_answer,
        score=int(data.get("score", 3)),
        strengths=data.get("strengths", []),
        improvements=data.get("improvements", []),
        summary=data.get("summary", ""),
    )

    state.feedback_history.append(fb)
    return state
