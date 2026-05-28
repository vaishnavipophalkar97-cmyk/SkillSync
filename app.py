import streamlit as st
import json
import re
import os
import pandas as pd
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# 1. Initialization and Client Configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Corporate AI Resume Shortlister", layout="wide")



# Initialize the modern Google GenAI Client
client = genai.Client(api_key="AIzaSyAxhnAZkC5TuXEUS2ecNjEPqigfcdA3l1U")

# ------------------------------------------------------------------
# 2. Core Helper Functions
# ------------------------------------------------------------------
def extract_text_from_pdf(file) -> str:
    """Extracts raw text content from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def evaluate_resume(resume_text: str, job_description: str) -> dict:
    """
    Sends the resume and job description to Gemini 2.5 Flash
    and returns a structured JSON evaluation.
    """
    prompt = f"""
    You are an expert HR data scientist and recruiter. Your task is to analyze the provided Candidate Resume 
    against the Job Description (JD). You must grade objectively and output strictly in JSON format.

    Job Description:
    \"\"\"{job_description}\"\"\"

    Candidate Resume:
    \"\"\"{resume_text}\"\"\"

    Analyze the alignment and provide your output matching this JSON schema exactly:
    {{
        "candidate_name": "Extract candidate name or use 'Unknown'",
        "match_score": 85,  // An integer percentage from 0 to 100
        "key_strengths": ["Strength 1", "Strength 2"],
        "skills_gap": ["Missing skill or weak area 1"],
        "experience_alignment": "Brief assessment of professional history alignment",
        "verdict": "Shortlist" // Must be either: "Shortlist", "Backup", or "Reject"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                # Enforce JSON structural response mirroring our requirements
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        # Parse the rigid JSON response back into a python dictionary
        return json.loads(response.text)
    except Exception as e:
        return {
            "candidate_name": "Error Parsing",
            "match_score": 0,
            "key_strengths": [f"Error generated: {str(e)}"],
            "skills_gap": [],
            "experience_alignment": "N/A",
            "verdict": "Reject"
        }

# ------------------------------------------------------------------
# 3. Streamlit UI Dashboard Interface
# ------------------------------------------------------------------
st.title("🎯 Enterprise AI Resume Shortlister")
st.caption("Automate candidate filtering using deep contextual semantic matching.")

# Layout: Two columns for input setup
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Step 1: Define Job Specification")
    job_desc = st.text_area(
        "Paste target Job Description here:", 
        height=300, 
        placeholder="Requirements, tech stack, roles and responsibilities..."
    )

with col2:
    st.subheader("📂 Step 2: Upload Target Resumes")
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (PDF format supported, multiple uploads allowed):", 
        type=["pdf"], 
        accept_multiple_files=True
    )

# ------------------------------------------------------------------
# 4. Screening Analytics & Orchestration Engine
# ------------------------------------------------------------------
if st.button("🚀 Run AI Screening Engine", type="primary"):
    if not job_desc.strip():
        st.warning("Please provide a valid Job Description before evaluating candidates.")
    elif not uploaded_files:
        st.warning("Please upload at least one candidate resume PDF file.")
    else:
        results_pool = []
        
        with st.spinner("Processing applicant database and running semantic evaluations..."):
            for file in uploaded_files:
                # 1. Parse content
                resume_raw_text = extract_text_from_pdf(file)
                
                # 2. Pass payload to Gemini model evaluation
                evaluation = evaluate_resume(resume_raw_text, job_desc)
                # Keep track of file origin references
                evaluation['filename'] = file.name
                
                results_pool.append(evaluation)
        
        st.success("🎉 Screening Pipeline execution successful!")
        st.balloons()
        
        # Convert aggregated metrics to a clean analytical dataframe
        df = pd.DataFrame(results_pool)
        
        # Reorder columns to place primary metrics cleanly in the view
        ordered_cols = ['candidate_name', 'match_score', 'verdict', 'experience_alignment', 'filename']
        # Fallback security structure handling for data inconsistencies
        display_df = df[[c for c in ordered_cols if c in df.columns]]
        
        # Sort values based on priority of the match score matrix
        display_df = display_df.sort_values(by="match_score", ascending=False)
        
        # --- UI Segment Displaying Structured Analytics ---
        st.subheader("📊 Candidate Ranking Matrix Dashboard")
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "match_score": st.column_config.ProgressColumn(
                    "Overall Match Score",
                    help="AI generated overall semantic compatibility metric",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
                "verdict": st.column_config.SelectboxColumn(
                    "Hiring Status Verdict",
                    options=["Shortlist", "Backup", "Reject"]
                )
            }
        )
        
        # Detailed Deep-Dive breakdown for granular HR verification
        st.markdown("---")
        st.subheader("🔍 Deep-Dive Candidate Diagnostics Profiles")
        
        for candidate in results_pool:
            # Dynamic badge configuration depending on verdict values
            v_color = "🟢" if candidate.get('verdict') == "Shortlist" else ("🟡" if candidate.get('verdict') == "Backup" else "🔴")
            
            with st.expander(f"{v_color} {candidate.get('candidate_name')} — Score: {candidate.get('match_score')}% ({candidate.get('verdict')})"):
                st.write(f"**Associated Application File:** `{candidate.get('filename')}`")
                st.write(f"**Experience Profile Alignment:** {candidate.get('experience_alignment')}")
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.success("🌟 Notable Candidate Core Strengths")
                    for strength in candidate.get('key_strengths', []):
                        st.write(f"- {strength}")
                with col_right:
                    st.warning("⚠️ Identified Skill & Domain Gaps")
                    for gap in candidate.get('skills_gap', []):
                        st.write(f"- {gap}")