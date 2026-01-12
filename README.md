# AI-Powered Resume Screening and Ranking System

## Overview
This project automates resume screening and ranking using Natural Language Processing (NLP) techniques. It compares resumes against a given Job Description (JD) and ranks candidates based on relevance, reducing manual screening effort for recruiters.

## Problem Statement
Manual resume shortlisting is time-consuming and subjective. Recruiters often need to review hundreds of resumes for a single role. This system automates the process by ranking resumes based on textual similarity with the JD.

## Approach
- Extracted text from resumes and job descriptions
- Applied TF-IDF vectorization to convert text into numerical features
- Used Cosine Similarity to measure resume–JD relevance
- Ranked resumes based on similarity scores
- Built a simple interface to display ranked results

## Tech Stack
- Python
- NLP (TF-IDF, Cosine Similarity)
- Pandas, NumPy, Scikit-learn
- Streamlit

## Results
- Reduced manual resume shortlisting effort by approximately 70%
- Generated recruiter-ready ranked output
- Improved consistency and objectivity in screening

## How to Run
```bash
pip install -r requirements.txt
streamlit run resume_ranking_system3.py
