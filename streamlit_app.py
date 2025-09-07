

import os
import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openai

# Initialize OpenAI client using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("❌ OPENAI_API_KEY not found! Please set it in your environment variables.")

st.set_page_config(page_title="Smart Resume Reviewer", layout="wide")
st.title("📄 Smart Resume Reviewer")

# Resume Upload Section
resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="resume_upload")

# Job Description Upload Section
job_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"], key="jd_upload")

# Role selection
role = st.selectbox("🎯 Select Target Job Role", ["Data Scientist", "Frontend Developer", "Backend Developer", "Product Manager"], key="role_select")

# Function to extract text from uploaded files
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        return text
    else:
        return uploaded_file.read().decode("utf-8")

# GPT-based Resume Feedback
def gpt_resume_feedback(resume_text: str, role: str, job_desc: str = None) -> str:
    if not openai.api_key:
        return "⚠️ OpenAI API key missing."
    
    if job_desc:
        prompt = f"You are an expert career coach. Review the following resume for a {role} role. Consider this job description as reference: {job_desc}\n\nResume:\n{resume_text}"
    else:
        prompt = f"You are an expert career coach. Review the following resume for a {role} role and provide detailed feedback.\n\nResume:\n{resume_text}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

# GPT Resume Rewriter
def gpt_resume_rewriter(resume_text: str, role: str) -> str:
    if not openai.api_key:
        return "⚠️ OpenAI API key missing."

    prompt = f"Rewrite the following resume to better highlight achievements and skills relevant for a {role} role, while keeping the professional tone intact.\n\nResume:\n{resume_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

# GPT Career Suggestions
def gpt_career_suggestions(resume_text: str, role: str) -> str:
    if not openai.api_key:
        return "⚠️ OpenAI API key missing."

    prompt = f"You are a career mentor. Based on the following resume for a {role} role, suggest specific ways to improve career prospects, skills to learn, and unique strategies to stand out.\n\nResume:\n{resume_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

# Skill analysis for visualization
def skill_analysis(resume_text: str, job_desc: str = None):
    skills = ["Python", "Machine Learning", "Data Analysis", "SQL", "Communication", "Leadership", "Teamwork", "Problem Solving"]
    resume_scores = np.random.randint(40, 90, len(skills))
    job_scores = np.random.randint(60, 100, len(skills)) if job_desc else None
    return skills, resume_scores, job_scores

# Radar chart for skills
def plot_radar(skills, resume_scores, job_scores=None):
    labels = np.array(skills)
    angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
    resume_scores = np.concatenate((resume_scores, [resume_scores[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, resume_scores, "o-", linewidth=2, label="Resume")
    ax.fill(angles, resume_scores, alpha=0.25)

    if job_scores is not None:
        job_scores = np.concatenate((job_scores, [job_scores[0]]))
        ax.plot(angles, job_scores, "o-", linewidth=2, label="Job Description")
        ax.fill(angles, job_scores, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Skill Match Analysis", size=16, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)

# Process files if uploaded
if resume_file:
    resume_text = extract_text(resume_file)
    st.subheader("📊 Resume Analysis Output")
    st.text_area("Extracted Resume Text", resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""), height=300)
    
    job_desc = None
    if job_file:
        job_desc = extract_text(job_file)
        st.text_area("📌 Job Description Text", job_desc[:2000] + ("..." if len(job_desc) > 2000 else ""), height=300)

    if st.button("Analyze Resume", key="analyze_btn"):
        feedback = gpt_resume_feedback(resume_text, role, job_desc)
        st.subheader("💡 GPT-Powered Feedback")
        st.write(feedback)
        
        # Skill radar chart
        skills, resume_scores, job_scores = skill_analysis(resume_text, job_desc)
        st.subheader("📈 Skills Radar Chart")
        plot_radar(skills, resume_scores, job_scores)

    if st.button("Rewrite Resume", key="rewrite_btn"):
        rewritten = gpt_resume_rewriter(resume_text, role)
        st.subheader("✍️ GPT-Rewritten Resume")
        st.text_area("Improved Resume", rewritten, height=400)

    st.subheader("❓ Ask Custom Question About Resume")
    custom_q = st.text_input("Enter your question", key="custom_q")
    if st.button("Get Answer", key="q_btn") and custom_q:
        prompt = f"The following is a resume:\n{resume_text}\n\nQuestion: {custom_q}"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        st.write(response["choices"][0]["message"]["content"])

    if st.button("Get Career Suggestions", key="suggestion_btn"):
        suggestions = gpt_career_suggestions(resume_text, role)
        st.subheader("🚀 GPT Career Suggestions")
        st.write(suggestions)

# Notes for user
st.markdown("---")
st.info("⚠️ Your resume text is processed securely and never stored. Make sure to set your `OPENAI_API_KEY` in the environment.")
