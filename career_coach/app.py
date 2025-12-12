import io

import streamlit as st

st.set_page_config(page_title="Multi-Agent Career Coach", layout="wide")

st.title("🎓 Multi-Agent Career Coaching System")
st.write("If you see this title, `app.py` is loading correctly.")

# ---------------------------------------------------------------------
# Try imports and show errors in the UI if they fail
# ---------------------------------------------------------------------

IMPORT_OK = True
IMPORT_ERROR = None

try:
    from dotenv import load_dotenv
    from PyPDF2 import PdfReader

    from graph.state import SharedState, ResumeProfile
    from graph.graph_resume import build_resume_graph
    from graph.interview_agent import (
        generate_questions_node,
        evaluate_answer_node,
    )
    from rag.retriever import get_job_description_by_id
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = e

if not IMPORT_OK:
    st.error("❌ Import error in `app.py`.")
    st.code(repr(IMPORT_ERROR), language="python")
    st.stop()

# If we reach here, imports worked
load_dotenv()

st.success("✅ Imports loaded correctly.")

# ---------------------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------------------

if "app_state" not in st.session_state:
    st.session_state.app_state = SharedState()

if "resume_graph" not in st.session_state:
    st.session_state.resume_graph = build_resume_graph()

app_state: SharedState = st.session_state.app_state


# ---------------------------------------------------------------------
# Helper: Extract text from uploaded file
# ---------------------------------------------------------------------

def _extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name: str = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        # Basic PDF text extraction
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    st.warning("Unsupported file type. Please upload a .txt or .pdf file.")
    return ""


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("Steps")
st.sidebar.markdown("1. Upload resume\n2. Review matches\n3. Practice interviews")


# ---------------------------------------------------------------------
# Step 1: Resume upload & analysis
# ---------------------------------------------------------------------

st.header("1. Upload and Analyze Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume (.txt or .pdf)", type=["txt", "pdf"]
)

if st.button("Analyze Resume", type="primary"):
    text = _extract_text_from_upload(uploaded_file)

    if not text.strip():
        st.error("Could not read any text from the file. Please check the format.")
    else:
        app_state.resume_profile = ResumeProfile(raw_text=text)
        graph = st.session_state.resume_graph
        app_state = graph.invoke(app_state)
        st.session_state.app_state = app_state
        st.success("Resume analyzed and job matches generated!")

# Show analysis if available
if app_state.resume_profile:
    rp = app_state.resume_profile

    st.subheader("Resume Summary")
    if rp.experience_summary:
        st.write(rp.experience_summary)
    else:
        st.write("_Summary not computed yet. Click 'Analyze Resume' above._")

    st.subheader("Key Skills")
    if rp.skills:
        st.write(", ".join(rp.skills))
    else:
        st.write("_No skills extracted yet._")

    st.subheader("Suggestions to Improve Your Resume")
    if rp.suggestions:
        for s in rp.suggestions:
            st.markdown(f"- {s}")
    else:
        st.write("_No suggestions yet._")

st.markdown("---")


# ---------------------------------------------------------------------
# Step 2: Job matches
# ---------------------------------------------------------------------

st.header("2. Job Matches")

if app_state.job_matches:
    for i, jm in enumerate(app_state.job_matches):
        label = (
            f"{jm.title or 'Job'} at {jm.company or 'Company'} "
            f"({jm.location or 'Location'}) — Score: {jm.score:.2f}"
        )
        with st.expander(label):
            st.markdown(f"**Match rationale:** {jm.rationale}")
            if st.button("Select this job for interview practice", key=f"select_{i}"):
                app_state.selected_job_id = jm.job_id
                app_state.interview_questions = []
                app_state.feedback_history = []
                st.session_state.app_state = app_state
                st.success("Selected job for interview practice.")
else:
    st.write("_No job matches yet. Upload and analyze a resume first._")

st.markdown("---")


# ---------------------------------------------------------------------
# Step 3: Interview practice
# ---------------------------------------------------------------------

st.header("3. Interview Practice")

if app_state.selected_job_id:
    job_desc = get_job_description_by_id(app_state.selected_job_id)

    if job_desc:
        with st.expander("View selected job description"):
            preview = job_desc[:2500]
            if len(job_desc) > 2500:
                preview += "..."
            st.write(preview)

    if not app_state.interview_questions:
        if st.button("Generate Interview Questions", type="primary"):
            app_state = generate_questions_node(app_state)
            st.session_state.app_state = app_state
            if app_state.interview_questions:
                st.success(f"Generated {len(app_state.interview_questions)} questions.")
            else:
                st.error("Failed to generate questions. Try again.")
    else:
        st.subheader("Questions and Feedback")

        for idx, q in enumerate(app_state.interview_questions):
            st.markdown(f"**Q{idx + 1}. {q.question}**  _({q.dimension})_")

            answer_key = f"answer_{idx}"
            user_answer = st.text_area(
                "Your answer",
                key=answer_key,
                placeholder="Type your answer here...",
            )

            if st.button("Get Feedback", key=f"feedback_{idx}") and user_answer.strip():
                app_state = evaluate_answer_node(app_state, idx, user_answer)
                st.session_state.app_state = app_state

                fb = app_state.feedback_history[-1]
                st.markdown(f"**Score:** {fb.score}/5")
                st.markdown("**Strengths:**")
                for s in fb.strengths:
                    st.markdown(f"- {s}")
                st.markdown("**Areas for improvement:**")
                for im in fb.improvements:
                    st.markdown(f"- {im}")
                st.markdown(f"**Summary:** {fb.summary}")
else:
    st.write("_Select a job in Step 2 to start interview practice._")
