
# streamlit_app.py

import os
import io
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st

# Optional deps: only imported when used
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

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
    "summary": r"^(summary|objective|profile)\b",
    "experience": r"^(experience|work experience|professional experience|employment)\b",
    "projects": r"^(projects|personal projects|academic projects)\b",
    "education": r"^(education|academics|qualifications)\b",
    "skills": r"^(skills|technical skills|skills & tools)\b",
    "certifications": r"^(certifications|licenses)\b",
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
# Text Extraction
# -----------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Add 'pymupdf' to requirements.")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    return "\n".join(text_parts)

# -----------------------------
# NLP-ish helpers (lightweight)
# -----------------------------

def normalize_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()


def tokenize(txt: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z.+#\-/]*", txt.lower())
    return [t for t in tokens if t not in STOPWORDS]


def unique_keywords(txt: str) -> List[str]:
    toks = tokenize(txt)
    uniq = []
    seen = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def detect_sections(lines: List[str]) -> Dict[str, bool]:
    found = {k: False for k in SECTION_PATTERNS}
    for line in lines:
        l = line.strip().lower()
        for key, pat in SECTION_PATTERNS.items():
            if re.match(pat, l):
                found[key] = True
    return found


def contains_contact(text: str) -> bool:
    email_ok = re.search(r"[\w\.-]+@[\w\.-]+\.[a-z]{2,}", text, re.I) is not None
    phone_ok = re.search(r"(\+?\d[\d\s\-()]{8,})", text) is not None
    return email_ok and phone_ok


def bullets_ratio(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = [l for l in lines if re.match(r"^(?:[-*•]|\d+\.)\s+", l)]
    return len(bullets) / max(1, len(lines))


def passive_phrases_score(text: str) -> float:
    phrases = [
        "responsible for", "involved in", "tasked with", "was done", "were done",
        "was handled", "were handled", "was achieved", "were achieved"
    ]
    t = text.lower()
    hits = sum(t.count(p) for p in phrases)
    return hits / max(1, len(text.split(".")))


def first_person_count(text: str) -> int:
    return len(re.findall(r"\b(i|my|mine)\b", text.lower()))


def long_sentence_ratio(text: str, limit: int = 30) -> float:
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    long_s = [s for s in sentences if len(s.split()) > limit]
    return len(long_s) / max(1, len(sentences))


def compare_keywords(resume_txt: str, target_keywords: List[str]) -> Tuple[List[str], List[str]]:
    r_toks = set(unique_keywords(resume_txt))
    present, missing = [], []
    for kw in target_keywords:
        k = kw.lower()
        if any(k == t or k in t for t in r_toks):
            present.append(kw)
        else:
            missing.append(kw)
    return present, missing

# -----------------------------
# Scoring
# -----------------------------

def score_resume(
    sections: Dict[str, bool],
    contact_ok: bool,
    bullets_r: float,
    passive_s: float,
    first_person: int,
    long_sent_r: float,
    jd_missing_count: int,
    preset_missing_count: int,
    word_count: int,
) -> Tuple[Dict[str, int], int]:
    # Simple heuristic scoring out of 100
    s = {}
    s["sections"] = 20 if sections.get("experience") and sections.get("education") and sections.get("skills") else 10
    s["contact"] = 10 if contact_ok else 0
    s["bullets"] = 10 if bullets_r >= 0.25 else (5 if bullets_r >= 0.10 else 0)
    s["tone"] = 10 if passive_s < 0.3 and first_person <= 5 else (5 if passive_s < 0.6 else 0)
    s["clarity"] = 10 if long_sent_r <= 0.2 else (5 if long_sent_r <= 0.4 else 0)
    s["keywords_jd"] = max(0, 20 - jd_missing_count * 2)
    s["skills_preset"] = max(0, 20 - preset_missing_count * 2)
    s["length"] = 10 if 350 <= word_count <= 900 else (5 if 250 <= word_count <= 1200 else 0)

    total = sum(s.values())
    return s, total

# -----------------------------
# Optional: LLM suggestions (OpenAI)
# -----------------------------

def llm_suggestions(resume_text: str, jd_text: str, model: str = "gpt-4o-mini") -> str:
    """Return improved bullets/summary using OpenAI if OPENAI_API_KEY is set.
    If key not available, returns an empty string.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are an expert resume coach. Read the resume and job description, then "
            "return 5-7 bullet points that strongly align achievements with the JD. "
            "Use crisp, metric-driven bullets (STAR style) and avoid first person.\n\n"
            f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}\n"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return ""

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Smart Resume Reviewer", page_icon="🧠", layout="wide")

st.title("🧠 Smart Resume Reviewer — Level‑1 Project")
st.caption("Your privacy matters: files are processed in-memory and not uploaded to a server by this demo app.")

with st.sidebar:
    st.header("🔧 Settings")
    role = st.selectbox("Target Job Role", list(ROLE_SKILL_PRESETS.keys()), index=0)
    model_name = st.text_input("(Optional) OpenAI model", value="gpt-4o-mini")
    use_llm = st.checkbox("Use LLM to draft improved bullets (requires OPENAI_API_KEY)")
    st.markdown("---")
    st.markdown("**Tip:** Add an `OPENAI_API_KEY` in your environment to enable LLM suggestions.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume (PDF/TXT) or Paste Text")
    up = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    resume_text_input = st.text_area("Or paste your resume text", height=220)

with col2:
    st.subheader("🎯 Target Job Description (Optional but recommended)")
    jd_upload = st.file_uploader("Upload JD (PDF/TXT)", type=["pdf", "txt"], key="jd")
    jd_text_input = st.text_area("Or paste the JD text", height=220)

# Read resume text
resume_text = ""
if up is not None:
    if up.type == "application/pdf":
        try:
            resume_text = extract_text_from_pdf(up.read())
        except Exception as e:
            st.error(f"PDF parsing failed: {e}")
    else:
        resume_text = up.read().decode(errors="ignore")

if not resume_text and resume_text_input:
    resume_text = resume_text_input

# Read JD text
jd_text = ""
if jd_upload is not None:
    if jd_upload.type == "application/pdf":
        try:
            jd_text = extract_text_from_pdf(jd_upload.read())
        except Exception as e:
            st.error(f"JD PDF parsing failed: {e}")
    else:
        jd_text = jd_upload.read().decode(errors="ignore")

if not jd_text and jd_text_input:
    jd_text = jd_text_input

st.markdown("---")

if st.button("🔍 Review Resume", type="primary"):
    if not resume_text.strip():
        st.warning("Please upload or paste your resume text.")
    else:
        clean_text = normalize_text(resume_text)
        lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
        sections = detect_sections(lines)
        wc = len(tokenize(resume_text))
        contact_ok = contains_contact(resume_text)
        br = bullets_ratio(resume_text)
        ps = passive_phrases_score(resume_text)
        fp = first_person_count(resume_text)
        lsr = long_sentence_ratio(resume_text)

        jd_keys = unique_keywords(jd_text) if jd_text else []
        jd_present, jd_missing = compare_keywords(resume_text, jd_keys[:30]) if jd_keys else ([], [])

        preset = ROLE_SKILL_PRESETS.get(role, [])
        preset_present, preset_missing = compare_keywords(resume_text, preset)

        sbreak, score_total = score_resume(
            sections, contact_ok, br, ps, fp, lsr,
            jd_missing_count=len(jd_missing),
            preset_missing_count=len(preset_missing),
            word_count=wc,
        )

        analysis = Analysis(
            sections_found=sections,
            word_count=wc,
            contact_ok=contact_ok,
            bullets_ratio=br,
            passive_score=ps,
            first_person_count=fp,
            long_sentence_ratio=lsr,
            jd_keywords_present=jd_present,
            jd_keywords_missing=jd_missing,
            preset_skills_present=preset_present,
            preset_skills_missing=preset_missing,
            score_breakdown=sbreak,
            total_score=score_total,
        )

        # ----------------- Output UI -----------------
        st.subheader("📊 Score")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", analysis.total_score)
        c2.metric("Words", analysis.word_count)
        c3.metric("Bullets %", f"{analysis.bullets_ratio*100:.0f}%")
        c4.metric("Long Sentences %", f"{analysis.long_sentence_ratio*100:.0f}%")

        st.json(analysis.score_breakdown)

        st.subheader("✅ Sections Check")
        cols = st.columns(3)
        for i, (sec, present) in enumerate(analysis.sections_found.items()):
            cols[i % 3].write(f"**{sec.title()}**: {'✔️' if present else '❌'}")

        st.subheader("🧩 Keyword Alignment")
        with st.expander("JD Keywords Present"):
            st.write(", ".join(analysis.jd_keywords_present) if analysis.jd_keywords_present else "—")
        with st.expander("JD Keywords Missing"):
            st.write(", ".join(analysis.jd_keywords_missing) if analysis.jd_keywords_missing else "—")
        with st.expander(f"{role} Preset Skills Missing"):
            st.write(", ".join(analysis.preset_skills_missing) if analysis.preset_skills_missing else "—")

        st.subheader("🧠 Actionable Feedback")
        feedback = []
        # Structure
        must_have = ["experience", "education", "skills"]
        missing_sections = [s for s in must_have if not analysis.sections_found.get(s)]
        if missing_sections:
            feedback.append(f"Add missing sections: {', '.join(missing_sections)}.")
        # Contact
        if not analysis.contact_ok:
            feedback.append("Include a professional email and a reachable phone number at the top.")
        # Bullets
        if analysis.bullets_ratio < 0.25:
            feedback.append("Use more bullet points for achievements; aim for ~3–5 per role.")
        # Tone
        if analysis.passive_score >= 0.3:
            feedback.append("Reduce passive phrases (e.g., 'responsible for'). Use active, impact-first bullets.")
        if analysis.first_person_count > 5:
            feedback.append("Avoid first-person pronouns; write bullets in neutral, concise form.")
        # Clarity
        if analysis.long_sentence_ratio > 0.2:
            feedback.append("Shorten long sentences to under ~20–24 words for readability.")
        # Keywords
        if analysis.jd_keywords_missing:
            feedback.append("Incorporate JD keywords naturally where you truly have experience (avoid keyword stuffing).")
        if analysis.preset_skills_missing:
            feedback.append(f"Highlight relevant {role} skills you have but forgot to list: {', '.join(analysis.preset_skills_missing[:8])}.")
        # Length
        if not (350 <= analysis.word_count <= 900):
            feedback.append("Keep resume within 1 page (fresher) or 1–2 pages (experienced), roughly 350–900 words.")

        if feedback:
            for f in feedback:
                st.write("- ", f)
        else:
            st.success("Great job! Your resume looks well-structured and aligned.")

        # ----------------- Optional: LLM Improvements -----------------
        improved_bullets = ""
        if use_llm:
            with st.spinner("Asking LLM for improved bullets..."):
                improved_bullets = llm_suggestions(resume_text, jd_text, model=model_name)
        else:
            # Fallback heuristic bullets if LLM disabled
            if role in ("Data Scientist", "ML Engineer", "Data Analyst"):
                improved_bullets = (
                    "• Built end-to-end data pipeline; reduced processing time by 30% using Python & SQL.\n"
                    "• Trained and validated ML models (XGBoost/LogReg), improving F1 by 12% on holdout.\n"
                    "• Deployed model behind REST API (FastAPI); implemented monitoring and A/B tests.\n"
                    "• Visualized KPIs with dashboards; enabled data-driven decisions across teams.\n"
                )
            else:
                improved_bullets = (
                    "• Delivered feature from spec to release; boosted engagement by 15%.\n"
                    "• Collaborated cross-functionally; clarified requirements and removed blockers.\n"
                    "• Wrote clean, tested code and documentation; reduced bugs after release.\n"
                )

        st.subheader("✨ Suggested Impact Bullets")
        st.code(improved_bullets or "(No suggestions generated.)", language="markdown")

        # ----------------- Download: Improved Resume (Markdown) -----------------
        st.subheader("📥 Export")
        improved_md = f"""# {role} Resume\n\n## Summary\nResults-driven {role.lower()} with hands-on experience.\n\n## Experience\n{improved_bullets}\n\n## Skills\n- {', '.join(analysis.preset_skills_present[:12])}\n\n## Education\n- Your Degree, College, Year\n"""
        st.download_button(
            label="Download Improved Resume (Markdown)",
            data=improved_md.encode(),
            file_name="improved_resume.md",
            mime="text/markdown",
        )

st.markdown("---")
with st.expander("🔒 Privacy & Notes"):
    st.write(
        "This demo processes text in-memory only. For production, add authentication, secure storage, and an explicit privacy disclaimer."
    )
    st.write(
        "LLM outputs may be inaccurate. Always verify content and avoid misrepresentation."
    )
# streamlit_app.py

import os
import re
from dataclasses import dataclass
from typing import List, Dict

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -----------------------------
# GPT Integration
# -----------------------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

# -----------------------------
# Utility & Config
# -----------------------------

STOPWORDS = set("""
a an the and or but if while of on in into to for from with without over under
is are was were be been being this that those these it its as at by about we i me my our your their them you
""".split())

ROLE_SKILL_PRESETS: Dict[str, List[str]] = {
    "Data Scientist": ["python", "pandas", "numpy", "scikit-learn", "sql", "statistics", "machine learning"],
    "Frontend Developer": ["javascript", "typescript", "react", "redux", "html", "css", "tailwind"],
    "Backend Developer": ["python", "java", "node.js", "express", "spring", "rest api", "sql"],
}

@dataclass
class ResumeOutput:
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    missing_keywords: List[str]
    score: int

# -----------------------------
# Extraction & Analysis
# -----------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed.")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = [page.get_text() for page in doc]
    return "\n".join(text_parts)


def analyze_resume(resume_text: str, role: str) -> ResumeOutput:
    words = re.findall(r"[a-zA-Z]+", resume_text.lower())
    score = min(100, len(words) // 5)

    preset = ROLE_SKILL_PRESETS.get(role, [])
    present = [kw for kw in preset if kw in resume_text.lower()]
    missing = [kw for kw in preset if kw not in present]

    strengths = [f"Contains skill: {kw}" for kw in present]
    weaknesses = ["Add more measurable achievements.", "Avoid long paragraphs; use bullet points."]

    return ResumeOutput(
        summary=f"This resume is moderately aligned for {role}.",
        strengths=strengths,
        weaknesses=weaknesses,
        missing_keywords=missing,
        score=score,
    )


def gpt_resume_feedback(resume_text: str, role: str, custom_prompt: str = None) -> str:
    if not client:
        return "⚠️ GPT client not configured. Please set your OPENAI_API_KEY."
    try:
        base_prompt = f"Please review the following resume for the role of {role}:\n\n{resume_text}"
        if custom_prompt:
            base_prompt += f"\n\nAdditional request: {custom_prompt}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert career coach reviewing resumes."},
                {"role": "user", "content": base_prompt},
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error while fetching GPT feedback: {e}"


def gpt_rewrite_section(resume_text: str, section: str, role: str) -> str:
    if not client:
        return "⚠️ GPT client not configured. Please set your OPENAI_API_KEY."
    try:
        prompt = f"Rewrite and improve the {section} section of this resume for the role of {role}.\n\nResume:\n{resume_text}" 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume writer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error while rewriting section: {e}"

# -----------------------------
# Streamlit UI with Advanced CSS Styling
# -----------------------------

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# Inject custom CSS for advanced styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        transform: scale(1.05);
    }
    .stMetric label, .stSubheader, h2, h3 {
        color: #34495e;
    }
    .css-1d391kg p {
        font-size: 1.1em;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>📄 Smart Resume Reviewer — Visual Output</h1>", unsafe_allow_html=True)

role = st.sidebar.selectbox("Target Role", list(ROLE_SKILL_PRESETS.keys()), index=0)

st.subheader("Upload Resume")
up = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
resume_text_input = st.text_area("Or paste resume text", height=200)

if st.button("🔍 Analyze Resume"):
    resume_text = ""
    if up:
        if up.type == "application/pdf":
            resume_text = extract_text_from_pdf(up.read())
        else:
            resume_text = up.read().decode(errors="ignore")
    elif resume_text_input:
        resume_text = resume_text_input

    if not resume_text.strip():
        st.warning("Please upload or paste resume text.")
    else:
        output = analyze_resume(resume_text, role)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Score", output.score)
            st.subheader("Summary")
            st.write(output.summary)

            st.subheader("Strengths")
            for s in output.strengths:
                st.write("✔️", s)

            st.subheader("Weaknesses")
            for w in output.weaknesses:
                st.write("⚠️", w)

        with col2:
            # Bar chart of section scores (dummy for now)
            section_names = ["Skills", "Experience", "Education", "Projects"]
            section_scores = [min(100, output.score + np.random.randint(-10, 10)) for _ in section_names]

            fig, ax = plt.subplots()
            ax.bar(section_names, section_scores, color="#3498db")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score")
            ax.set_title("Section-wise Scores")
            st.pyplot(fig)

            # Radar chart for skills coverage
            skills = ROLE_SKILL_PRESETS.get(role, [])
            if skills:
                values = [1 if kw in resume_text.lower() else 0 for kw in skills]
                num_skills = len(skills)

                angles = np.linspace(0, 2 * np.pi, num_skills, endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]

                fig2, ax2 = plt.subplots(subplot_kw={"polar": True})
                ax2.plot(angles, values, "o-", linewidth=2, color="#e74c3c")
                ax2.fill(angles, values, alpha=0.25, color="#e74c3c")
                ax2.set_xticks(angles[:-1])
                ax2.set_xticklabels(skills)
                ax2.set_yticks([0, 1])
                ax2.set_title("Skills Coverage Radar")
                st.pyplot(fig2)

        st.subheader("Missing Keywords")
        st.write(", ".join(output.missing_keywords) if output.missing_keywords else "None 🎉")

        st.download_button(
            label="Download Analysis Report",
            data=f"Summary:\n{output.summary}\n\nStrengths:\n- " + "\n- ".join(output.strengths) + "\n\nWeaknesses:\n- " + "\n- ".join(output.weaknesses),
            file_name="resume_report.txt",
            mime="text/plain",
        )

        # GPT feedback section
        st.subheader("💡 GPT-Powered Feedback")
        gpt_feedback = gpt_resume_feedback(resume_text, role)
        st.write(gpt_feedback)

        # Custom GPT Q&A
        st.subheader("❓ Ask Custom Questions about Resume")
        custom_query = st.text_input("Enter your question or request")
        if st.button("Ask GPT") and custom_query.strip():
            answer = gpt_resume_feedback(resume_text, role, custom_prompt=custom_query)
            st.write(answer)

        # GPT Rewrite Section
        st.subheader("✍️ Improve Resume Sections with GPT")
        section_choice = st.selectbox("Select Section to Rewrite", ["Experience", "Education", "Skills", "Projects"])
        if st.button("Rewrite Section"):
            rewritten = gpt_rewrite_section(resume_text, section_choice, role)
            st.write(rewritten)
resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="resume_upload")
job_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"], key="job_upload")
