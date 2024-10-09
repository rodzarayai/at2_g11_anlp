import streamlit as st
from job_descriptions import *
from aux_func import *
import PyPDF2
import os
import pickle

import plotly.express as px


#====================================================================PAGE CONFIG
apptitle = 'CV Job Matcher App'

st.set_page_config(page_title=apptitle, 
                   page_icon="💼")


st.title("CV Job Matcher")

st.subheader("Match your CV with the last published offers in the web!")


#=================================job_desc 
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, '../data')

#Read the bulk of job descriptions
with open(os.path.join(data_dir, 'job_desc_embeddings.pkl'), 'rb') as f:
    #with open('../data/job_desc_embeddings.pkl', 'rb') as f:
    #with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings.pkl', 'rb') as f:
    job_desc_embeddings_total = pickle.load(f)


df_path = os.path.join(data_dir, 'job_descriptions', 'training_data.csv')
print(df_path)
jobs_df = pd.read_csv(df_path)


# 1. Top Companies Hiring (by company_name)
# value_counts() returns a Series, so reset the index to make it a DataFrame
top_companies = jobs_df['company_name'].value_counts().reset_index()
top_companies.columns = ['company_name', 'job_count']  # Renaming for plotly

# Sort by job_count in descending order and get the top 10 (no 'by' argument for Series sorting)
top_companies = top_companies.sort_values('job_count', ascending=True).head(10)


# 2. Hottest Positions (by position_title)
hottest_positions = jobs_df['position_title'].value_counts().head(10).reset_index()
hottest_positions.columns = ['position_title', 'job_count']  # Renaming for plotly
hottest_positions = hottest_positions.sort_values('job_count', ascending=True).head(10)


# Create two columns in Streamlit
col1, col2 = st.columns(2)

with col1:
    # Bar Plot: Top Companies Hiring
    st.subheader("Top Companies Hiring")
    fig_companies = px.bar(
        top_companies, 
        x='job_count',  # X-axis is the count
        y='company_name',  # Y-axis is the company names
        title="Top Companies Hiring",
        labels={'company_name': 'Company Name', 'job_count': 'Number of Jobs'},
        orientation='h'  # Horizontal orientation
    )
    st.plotly_chart(fig_companies)

with col2:
# Bar Plot: Hottest Positions
    st.subheader("Hottest Positions")
    fig_positions = px.bar(
        hottest_positions, 
        x='job_count',  # X-axis is the count
        y='position_title',  # Y-axis is the position titles
        title="Hottest Positions",
        labels={'position_title': 'Position Title', 'job_count': 'Number of Jobs'},
        orientation='h'  # Horizontal orientation
    )
    st.plotly_chart(fig_positions)

#Process the job description
    # """ 
    # text_preprocessor = TextPreprocessor(processing_mode='none')
    # jobs_df = text_preprocessor.preprocess_dataframe(jobs_df, 'job_description')
    # jobs_df['skills_matched_cleaned'] = jobs_df['job_description_processed_cleaned'].apply(find_skills)

    # jobs_skills = [skill for sublist in jobs_df['skills_matched_cleaned'] for skill in sublist]

    # # Count the occurrences of each skill
    # skill_counts = Counter(jobs_skills)

    # # Convert the Counter object to a DataFrame for plotting
    # skills_df = pd.DataFrame(skill_counts.items(), columns=['skill', 'count'])

    # # Sort the DataFrame by count in descending order and get the top 10 most common skills
    # skills_df = skills_df.sort_values('count', ascending=False).head(10)

    # # Horizontal Bar Plot: Most Common Skills
    # st.subheader("Top 10 Most Common Skills")
    # fig_skills = px.bar(
    #     skills_df, 
    #     x='count',  # X-axis is the count of occurrences
    #     y='skill',  # Y-axis is the skill names
    #     title="Top 10 Most Common Skills",
    #     labels={'count': 'Number of Occurrences', 'skill': 'Skill'},
    #     orientation='h'  # Horizontal orientation
    # )
    # st.plotly_chart(fig_skills)

    # """

#==========================================TOTAL JOBS




#===========================================

# Default job description
example_job_desc = "We are looking for a data scientist with experience in machine learning and Python and SQL. Knowledge in database and version control and webscrapping."

# Create a text area input with the default job description
new_job_desc = st.text_area("Enter the job description:", example_job_desc, height=150)

# Display the job description (either modified or default)
st.write("Job Description:")
st.write(new_job_desc)


#Process the job description
text_preprocessor = TextPreprocessor(processing_mode='none')
new_job_cleaned = text_preprocessor.preprocess(new_job_desc)


total_skills, skills_matched, job_desc_embeddings = job_desc_emb(new_job_cleaned)


# Create a selection option
selection = st.radio(
    "Which skills list would you like to see?",
    ('Total List', 'Skills Matched')
)

# Create a button to display the selected list
if st.button("Show Skills List"):
    if selection == 'Total List':
        st.write("Total Skills List:")
        st.write(total_skills)
    else:
        st.write("Matched Skills List:")
        st.write(skills_matched)



# Function to extract text from an uploaded PDF file
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ' '.join(page.extract_text() for page in reader.pages)
    return text

# Streamlit app
st.title("PDF Text Extractor")

# File uploader widget in Streamlit
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # If a file is uploaded, extract and display the text
    cv_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", cv_text, height=300)

    #Process the job description
    text_preprocessor_cv = TextPreprocessor(processing_mode='none')
    cv_text_cleaned = text_preprocessor_cv.preprocess(cv_text)
    print(type(cv_text_cleaned))
    st.text_area("Extracted Text", cv_text_cleaned, height=300)


    total_skills_cv, skills_matched_cv, resume_emb = job_desc_emb(cv_text_cleaned)

    st.write(skills_matched_cv)

    result_df = pd.DataFrame(columns=['jobId', 'resumeId', 'similarity', 'domainResume', 'domainDesc'])

   # Loop through the job descriptions and calculate similarity with the CV
    for i, job_desc_emb in enumerate(job_desc_embeddings_total):
        job_desc_emb_2d = np.array(job_desc_emb).reshape(1, -1)  # Converts to 2D (1 row)
        resume_emb_2d = np.array(resume_emb).reshape(1, -1)  # Converts to 2D (1 row)
        print(job_desc_emb.shape)
        print(job_desc_emb_2d.shape)
        print(resume_emb.shape)
        print(resume_emb_2d.shape)
        # Calculate cosine similarity between the CV and the current job description
        similarity = cosine_similarity(job_desc_emb_2d, resume_emb_2d)[0][0]
        print(similarity)
        # Append the similarity result to the DataFrame
        # Create a new row to be added
        new_row = pd.DataFrame({
        'jobId': [i],  # Assuming i is the jobId, otherwise use actual jobId
        'resumeId': ['resume_01'],  # Replace with actual resume ID if available
        'similarity': [similarity],
        'domainResume': ['domain_placeholder'],  # Add appropriate domain info
        'domainDesc': ['domain_placeholder']  # Add appropriate domain info
        })
    
        # Concatenate the new row with the existing DataFrame
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    # Filter for jobs with similarity score above threshold
    result_df = result_df[result_df['similarity'] > 0.7]

    # Sort by similarity in descending order
    result_df = result_df.sort_values(by='similarity', ascending=False)

    # Get the top 10 jobs
    top_10_jobs = result_df.head(10)

    # Display the result
    st.write(top_10_jobs)