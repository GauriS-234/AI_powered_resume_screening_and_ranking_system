import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + " "
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2)).fit_transform(documents)
    vectors = vectorizer.toarray()
    
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
    return cosine_similarities

# Streamlit UI enhancements with theme
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("Use this tool to rank resumes based on job descriptions.")

st.title("📄 AI Resume Screening & Candidate Ranking")
st.markdown("### Upload resumes and enter a job description to get ranked results.")

# Job description input
st.subheader("Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.subheader("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description.strip():
    st.subheader("Ranking Resumes")

    # Extract text from uploaded PDFs
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    
    # Create DataFrame and rank resumes
    results_df = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files],
        "Score": scores
    })
    results_df["Rank"] = results_df["Score"].rank(method="max", ascending=False).astype(int)
    results_df = results_df.sort_values(by="Score", ascending=False)

    # Display ranked results
    for _, row in results_df.iterrows():
        st.markdown(f"### 🏆 Rank #{row['Rank']}: {row['Resume']}")
        st.progress(float(row['Score']))
        st.write(f"🔹 **Match Score:** {row['Score']:.2f}")

    # Convert DataFrame to CSV for download
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Download button
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="resume_ranking_results.csv",
        mime="text/csv"
    )
else:
    st.warning("Please enter a job description and upload at least one resume.")
