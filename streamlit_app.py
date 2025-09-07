import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import openai

# -----------------------------
# Enhanced Smart Resume Reviewer
# -----------------------------
st.set_page_config(page_title="Smart Resume Reviewer", layout="wide")

# Sidebar: API key input and settings
st.sidebar.title("Settings 🔧")
api_key_input = st.sidebar.text_input("OpenAI API key (optional)", type="password")
if api_key_input:
    openai.api_key = api_key_input
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.sidebar.warning("OpenAI API key not found — GPT features will be disabled.")

role = st.sidebar.selectbox("Target Role", ["Data Scientist", "Frontend Developer", "Backend Developer", "Product Manager"], index=0)
use_gpt = st.sidebar.checkbox("Enable GPT suggestions", value=bool(openai.api_key))
max_gpt_tokens = st.sidebar.slider("GPT max tokens", 200, 1200, 600)

st.title("📄 Smart Resume Reviewer — Enhanced")
st.markdown("AI-powered resume analysis with visual insights and optional GPT improvements.")

# -----------------------------
# Helpers
# -----------------------------

@dataclass
class Analysis:
    total_score: int
    word_count: int
    bullets_pct: float
    long_sentence_pct: float
    contact_ok: bool
    sections: Dict[str, bool]
    preset_present: List[str]
    preset_missing: List[str]
    jd_present: List[str]
    jd_missing: List[str]


ROLE_SKILLS = {
    "Data Scientist": ["python", "pandas", "numpy", "scikit-learn", "sql", "ml", "statistics"],
    "Frontend Developer": ["javascript", "react", "css", "html", "typescript"],
    "Backend Developer": ["python", "java", "node", "sql", "rest api"],
    "Product Manager": ["product strategy", "roadmap", "user research", "kpi"],
}

SECTION_PATTERNS = {
    "summary": r"^(summary|objective|profile)",
    "experience": r"^(experience|work experience|professional experience|employment)",
    "projects": r"^(projects|personal projects|academic projects)",
    "education": r"^(education|academics|qualifications)",
    "skills": r"^(skills|technical skills|skills & tools)",
}
# Demo GPT suggestion fallback
def get_gpt_suggestions(text: str, role: str) -> str:
    if not openai.api_key or openai.api_key.strip() == "":
        return (
            f"💡 **Demo Suggestions for {role}:**\n"
            "- Add more measurable achievements (use numbers/percentages).\n"
            "- Highlight projects relevant to the role.\n"
            "- Tailor your skills section with industry keywords.\n"
            "- Keep sentences concise and action-oriented.\n"
        )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer."},
                {"role": "user", "content": f"Review this resume for a {role} role and give concise suggestions:\n{text}"}
            ],
            max_tokens=max_gpt_tokens,
        )
        return response.choices[0].message["content"]
    except Exception:
        return "⚠️ GPT request failed, showing demo suggestions instead.\n- Emphasize role-specific keywords.\n- Improve clarity in work experience.\n- Add a strong summary section."


def extract_text_from_pdf_bytes(b: bytes) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(stream=b, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\+\#\-]+", text.lower())


def detect_sections(lines: List[str]) -> Dict[str, bool]:
    found = {k: False for k in SECTION_PATTERNS}
    for line in lines:
        l = line.strip().lower()
        for k, p in SECTION_PATTERNS.items():
            if re.match(p, l):
                found[k] = True
    return found


def contains_contact(text: str) -> bool:
    email = re.search(r"[\w\.-]+@[\w\.-]+\.[a-z]{2,}", text, re.I)
    phone = re.search(r"(\+?\d[\d\s\-()]{7,})", text)
    return bool(email and phone)


def bullets_ratio(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    bullets = [l for l in lines if re.match(r"^(?:[-*•]|\d+\.)\s+", l)]
    return len(bullets) / len(lines)


def long_sentence_ratio(text: str, limit: int = 28) -> float:
    sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if not sents:
        return 0.0
    long_s = [s for s in sents if len(s.split()) > limit]
    return len(long_s) / len(sents)


def compare_keywords(res_text: str, keywords: List[str]) -> (List[str], List[str]):
    toks = set(tokenize(res_text))
    present = [k for k in keywords if any(k.lower() in t for t in toks)]
    missing = [k for k in keywords if k not in present]
    return present, missing


def compute_score(sections: Dict[str, bool], contact_ok: bool, bullets_p: float, long_p: float, jd_missing: int, preset_missing: int, word_cnt: int) -> int:
    s = 0
    s += 20 if sections.get("experience") and sections.get("education") and sections.get("skills") else 10
    s += 10 if contact_ok else 0
    s += 10 if bullets_p >= 0.25 else (5 if bullets_p >= 0.10 else 0)
    s += 10 if long_p <= 0.2 else (5 if long_p <= 0.4 else 0)
    s += max(0, 20 - jd_missing * 2)
    s += max(0, 20 - preset_missing * 2)
    s += 10 if 350 <= word_cnt <= 900 else (5 if 250 <= word_cnt <= 1200 else 0)
    return min(100, s)


# -----------------------------
# Uploaders (unique keys)
# -----------------------------
st.markdown("---")
col_a, col_b = st.columns([2,1])
with col_a:
    st.header("1. Upload Resume & Job Description")
    resume_up = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt"], key="resume_up")
    manual_resume = st.text_area("Or paste your resume text", height=200, key="manual_resume")
    st.write("---")
    jd_up = st.file_uploader("Job Description (PDF or TXT) — optional", type=["pdf", "txt"], key="jd_up")
    manual_jd = st.text_area("Or paste job description (optional)", height=150, key="manual_jd")

with col_b:
    st.header("Quick Actions")
    st.write("Choose options and click **Analyze Resume** to run checks and (optionally) fetch GPT suggestions.")
    st.write("\n")
    st.write("**Role:** ", role)
    st.write("**GPT:** ", "On" if use_gpt and openai.api_key else "Off")

# Read texts
resume_text = ""
if resume_up is not None:
    if resume_up.type == "application/pdf":
        resume_text = extract_text_from_pdf_bytes(resume_up.read())
    else:
        resume_text = resume_up.read().decode(errors="ignore")

if not resume_text and manual_resume:
    resume_text = manual_resume

jd_text = ""
if jd_up is not None:
    if jd_up.type == "application/pdf":
        jd_text = extract_text_from_pdf_bytes(jd_up.read())
    else:
        jd_text = jd_up.read().decode(errors="ignore")

if not jd_text and manual_jd:
    jd_text = manual_jd

# Empty state
if not resume_text:
    st.info("Upload or paste a resume to get started — tips: include contact info, bullets, and role-aligned skills.")

# -----------------------------
# Analysis and Visualization
# -----------------------------
if resume_text and st.button("Analyze Resume", key="analyze_main"):
    cleaned = sanitize_text(resume_text)
    lines = [l for l in resume_text.splitlines() if l.strip()]
    sections = detect_sections(lines)
    wc = len(tokenize(resume_text))
    contact_ok = contains_contact(resume_text)
    bullets_p = bullets_ratio(resume_text)
    long_p = long_sentence_ratio(resume_text)

    preset = ROLE_SKILLS.get(role, [])
    preset_present, preset_missing = compare_keywords(resume_text, preset)

    jd_present, jd_missing = ([], [])
    if jd_text:
        jd_keys = list({k.lower() for k in tokenize(jd_text)})[:40]
        jd_present, jd_missing = compare_keywords(resume_text, jd_keys)

    total = compute_score(sections, contact_ok, bullets_p, long_p, len(jd_missing), len(preset_missing), wc)

    analysis = Analysis(
        total_score=total,
        word_count=wc,
        bullets_pct=bullets_p,
        long_sentence_pct=long_p,
        contact_ok=contact_ok,
        sections=sections,
        preset_present=preset_present,
        preset_missing=preset_missing,
        jd_present=jd_present,
        jd_missing=jd_missing,
    )

    # Results layout
    st.subheader("Results — Quick Summary")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Score", analysis.total_score)
    cc2.metric("Words", analysis.word_count)
    cc3.metric("Bullets %", f"{analysis.bullets_pct*100:.0f}%")
    cc4.metric("Long Sent %", f"{analysis.long_sentence_pct*100:.0f}%")

    st.markdown("---")
    st.subheader("Sections & Keywords")
    cols = st.columns(3)
    for i, (k, v) in enumerate(analysis.sections.items()):
        cols[i%3].write(f"**{k.title()}**: {'✔️' if v else '❌'}")

    st.markdown("---")
    st.subheader("Skills & JD alignment")
    st.write("**Skills found:**", ", ".join(analysis.preset_present) if analysis.preset_present else "None")
    st.write("**Skills missing:**", ", ".join(analysis.preset_missing) if analysis.preset_missing else "None")
    st.write("**JD keywords present (sample):**", ", ".join(analysis.jd_present[:20]) if analysis.jd_present else "—")
    st.write("**JD keywords missing (sample):**", ", ".join(analysis.jd_missing[:20]) if analysis.jd_missing else "—")

    st.markdown("---")
    # Charts
    st.subheader("Visual Insights")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Section presence bar
        labels = list(analysis.sections.keys())
        vals = [1 if v else 0 for v in analysis.sections.values()]
        fig1, ax1 = plt.subplots()
        ax1.bar(labels, vals, color="#4CAF50")
        ax1.set_ylim(0,1)
        ax1.set_ylabel("Present")
        st.pyplot(fig1)

    with chart_col2:
        # Skills radar
        skills = ROLE_SKILLS.get(role, [])
        resume_scores = np.array([1 if s.lower() in resume_text.lower() else 0 for s in skills]) * 100
        job_scores = np.array([100 for _ in skills]) if jd_text else None
        if skills:
            angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
            vals = np.concatenate((resume_scores, [resume_scores[0]]))
            angs = angles + angles[:1]
            fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))
            ax2.plot(angs, vals, "o-", linewidth=2, color="#e74c3c")
            ax2.fill(angs, vals, alpha=0.25, color="#e74c3c")
            if job_scores is not None:
                js = np.concatenate((job_scores, [job_scores[0]]))
                ax2.plot(angs, js, "o-", linewidth=1, color="#3498db")
                ax2.fill(angs, js, alpha=0.12, color="#3498db")
            ax2.set_thetagrids(np.degrees(angs[:-1]), skills)
            ax2.set_ylim(0,100)
            st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Actionable Feedback")
    heuristics = []
    if not analysis.contact_ok:
        heuristics.append("Add a professional email and phone number at the top.")
    if analysis.bullets_pct < 0.25:
        heuristics.append("Use more bullets for achievements; aim for 3–5 bullets per role.")
    if analysis.long_sentence_pct > 0.2:
        heuristics.append("Shorten long sentences; prefer concise impact-driven bullets.")
    if analysis.preset_missing:
        heuristics.append(f"Consider adding/emphasizing: {', '.join(analysis.preset_missing[:8])}.")

    for h in heuristics:
        st.write("- ", h)

    # GPT suggestions (concise + bullets)
    gpt_text = None
    if use_gpt and openai.api_key:
        with st.spinner("Fetching GPT suggestions..."):
            prompt = (
                f"You are an expert resume coach. The candidate's resume is below. Provide: (A) 6 concise, role-specific improvements; (B) 6 high-impact achievement bullets rewritten for the resume.\n\nResume:\n{resume_text}\n\nRole: {role}\n\nJob Description:\n{jd_text if jd_text else 'N/A'}"
            )
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=max_gpt_tokens,
                    temperature=0.2,
                )
                gpt_text = resp["choices"][0]["message"]["content"]
                st.markdown("**GPT Suggestions**")
                st.write(gpt_text)
            except Exception as e:
                st.error(f"GPT request failed: {e}")
    elif use_gpt and not openai.api_key:
        st.info("Enter your OpenAI API key in the sidebar to enable GPT suggestions.")

    st.markdown("---")
    # Export improved resume (markdown) using GPT text if available
    if st.button("Generate Improved Resume (Markdown)"):
        improved = "# Improved Resume\n\n"
        improved += "## Summary\nA results-driven professional with relevant experience and proven impact.\n\n"
        improved += "## Experience\n"
        if gpt_text:
            improved += gpt_text + "\n\n"
        else:
            improved += "- Delivered feature X; improved metric Y by Z%.\n- Collaborated with cross-functional teams to ship product improvements.\n\n"
        improved += "## Skills\n- " + ", ".join(ROLE_SKILLS.get(role, [])[:12]) + "\n\n"
        st.download_button("Download Markdown", improved, file_name="improved_resume.md", mime="text/markdown")

st.markdown("---")
st.info("Your resume is processed in-memory only. Set OPENAI_API_KEY env or enter it in the sidebar to enable GPT features.")

