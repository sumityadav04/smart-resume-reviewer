# streamlit_app.py

import os
import io
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Optional deps: only imported when used
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -----------------------------
# Creative Theme and Branding
# -----------------------------

st.set_page_config(
    page_title="Smart Resume Reviewer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {background-color: #fdfcfb;}
    h1, h2, h3, h4 {color: #2c3e50;}
    .stButton>button {background-color:#4CAF50; color:white; border-radius:10px;}
    .stDownloadButton>button {background-color:#3498db; color:white; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utility & Config
# -----------------------------

STOPWORDS = set(
    """
    a an the and or but if while of on in into to for from with without over under
    is are was were be been being this that those these it its as at by about we i me my our your their them you
    """.split()
)

ROLE_SKILL_PRESETS: Dict[str, List[str]] = {
    "Data Scientist": [
        "python", "pandas", "numpy", "scikit-learn", "sql", "statistics", "probability",
        "machine learning", "regression", "classification", "clustering", "matplotlib",
        "seaborn", "feature engineering", "model evaluation", "cross-validation", "mlops",
    ],
    "ML Engineer": [
        "python", "pytorch", "tensorflow", "mlops", "docker", "kubernetes", "aws",
        "gcp", "azure", "fastapi", "flask", "api", "feature store", "monitoring",
        "data pipelines", "airflow", "ci/cd",
    ],
    "Data Analyst": [
        "sql", "excel", "power bi", "tableau", "python", "pandas", "reporting",
        "dashboard", "visualization", "a/b testing", "etl",
    ],
    "Frontend Developer": [
        "javascript", "typescript", "react", "redux", "next.js", "html", "css",
        "tailwind", "vite", "testing", "jest", "accessibility",
    ],
    "Backend Developer": [
        "python", "java", "node.js", "express", "spring", "rest api", "graphql",
        "sql", "postgresql", "mongodb", "redis", "docker", "kubernetes", "aws",
    ],
    "Product Manager": [
        "product strategy", "roadmap", "stakeholder management", "user research",
        "metrics", "kpi", "a/b testing", "prioritization", "communication",
    ],
}

SECTION_PATTERNS = {
    "summary": r"^(summary|objective|profile)\\b",
    "experience": r"^(experience|work experience|professional experience|employment)\\b",
    "projects": r"^(projects|personal projects|academic projects)\\b",
    "education": r"^(education|academics|qualifications)\\b",
    "skills": r"^(skills|technical skills|skills & tools)\\b",
    "certifications": r"^(certifications|licenses)\\b",
}

@dataclass
class Analysis:
    sections_found: Dict[str, bool]
    word_count: int
    contact_ok: bool
    bullets_ratio: float
    passive_score: float
    first_person_count: int
    long_sentence_ratio: float
    jd_keywords_present: List[str]
    jd_keywords_missing: List[str]
    preset_skills_present: List[str]
    preset_skills_missing: List[str]
    score_breakdown: Dict[str, int]
    total_score: int

# -----------------------------
# Streamlit UI Creative Enhancements
# -----------------------------

st.title("🧠 Smart Resume Reviewer")
st.markdown(
    "**Level‑1 Creative Project:** Helping job seekers shine with AI‑powered resume insights ✨"
)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/resume.png", use_column_width=True)
    st.header("🔧 Settings")
    role = st.selectbox("🎯 Target Job Role", list(ROLE_SKILL_PRESETS.keys()), index=0)
    model_name = st.text_input("(Optional) OpenAI model", value="gpt-4o-mini")
    use_llm = st.checkbox("💡 Use LLM to draft improved bullets (requires OPENAI_API_KEY)")
    st.info("Pro Tip: Add an `OPENAI_API_KEY` in secrets to unlock advanced suggestions.")

st.markdown("---")
st.subheader("🚀 How it Works")
st.markdown(
    "1. **Upload or paste** your resume.\\n"
    "2. **Provide a JD or select a role**.\\n"
    "3. Hit **Review Resume** and get: Scores, Keyword Checks, Actionable Feedback.\\n"
    "4. Download an **AI‑suggested improved version**."
)

# -----------------------------
# Example Visualization (Radar + Bar)
# -----------------------------

# Dummy example for demonstration (replace with real Analysis results)
example_scores = {
    "Structure": 7,
    "Clarity": 8,
    "Keywords": 6,
    "Tone": 7,
    "Overall": 28,
}

st.markdown("---")
st.subheader("📊 Visual Feedback Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Section-wise Scores")
    fig, ax = plt.subplots()
    ax.bar(example_scores.keys(), example_scores.values(), color="#4CAF50")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 10)
    st.pyplot(fig)

with col2:
    st.markdown("#### Skills Coverage (Radar Chart)")
    labels = np.array(["Python", "SQL", "ML", "Visualization", "Cloud"])
    stats = np.array([8, 6, 7, 5, 4])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    fig2, ax2 = plt.subplots(subplot_kw={"polar": True})
    ax2.plot(angles, stats, "o-", linewidth=2, color="#3498db")
    ax2.fill(angles, stats, alpha=0.25, color="#3498db")
    ax2.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax2.set_ylim(0, 10)
    st.pyplot(fig2)

# -----------------------------
# Creative Closing
# -----------------------------

st.markdown("---")
st.subheader("🌟 Why use Smart Resume Reviewer?")
st.markdown(
    "- 🎯 Tailored to your chosen role.\\n"
    "- 📊 Transparent scoring breakdown.\\n"
    "- ✨ AI‑suggested impact bullets.\\n"
    "- 🛡️ Privacy‑first: no data stored."
)

st.success("Your resume deserves to stand out — let's make it shine!")

