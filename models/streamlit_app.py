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


#st.title("CV Job Matcher")

st.markdown("""
<h1 style='text-align: center; font-size: 40px;'>CV Job Matcher App</h1>
            
<h2 style='text-align: center; font-size: 20px'> 
    Looking for the perfect opportunity to align your skills with the latest job offers? You're in the right place!
</h2>
            
<p style='text-align: center; font-size: 16px'>
    CV Job Matcher analyzes your resume and compares it with the latest job postings from top companies. Whether you're a seasoned professional or just starting out, we help you discover the best roles that match your unique skillset.

</p>
""", unsafe_allow_html=True)

st.header("", divider="gray")

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

st.markdown("""

            
<h2 style='text-align: center; font-size: 26px'> 
    🔥 Actual trendings in the job market 🔥
</h2>

""", unsafe_allow_html=True)



# # Create two columns in Streamlit
# col1, col2 = st.columns(2)

# with col1:
#     # Bar Plot: Top Companies Hiring
#     st.subheader("Top Companies Hiring")
#     fig_companies = px.bar(
#         top_companies, 
#         x='job_count',  # X-axis is the count
#         y='company_name',  # Y-axis is the company names
#         #title="Top Companies Hiring",
#         labels={'company_name': 'Company Name', 'job_count': 'Number of Jobs'},
#         orientation='h'  # Horizontal orientation
#     )
#     st.plotly_chart(fig_companies)

#with col2:
# Bar Plot: Hottest Positions
st.subheader("Top Positions")
fig_positions = px.bar(
    hottest_positions, 
    x='job_count',  # X-axis is the count
    y='position_title',  # Y-axis is the position titles
    #title="Hottest Positions",
    labels={'position_title': 'Position Title', 'job_count': 'Number of Jobs'},
    orientation='h'  # Horizontal orientation
)
st.plotly_chart(fig_positions)

#Process skills required
# This functions takes several minutes to load and process. It is changed by a preloaded csv.
#skills_df = get_skills_df(jobs_df)
skills_total_path = os.path.join(data_dir, 'processed_data', 'skills_jobs_df.csv')

skills_df = pd.read_csv(skills_total_path)

skills_df['skill'] = skills_df['skill'].str.title()
# Sort the DataFrame by count in descending order and get the top 10 most common skills
skills_df = skills_df.sort_values('count', ascending=False).head(10)
skills_df = skills_df.sort_values('count', ascending=True)
print(skills_df)
# # Horizontal Bar Plot: Most Common Skills
st.subheader("Top 10 Most Demanded Skills")
fig_skills = px.bar(
    skills_df, 
    x='count',  # X-axis is the count of occurrences
    y='skill',  # Y-axis is the skill names
    #title="Top 10 Most Common Skills",
    labels={'count': 'Number of Occurrences', 'skill': 'Skill'},
    orientation='h'  # Horizontal orientation
)
st.plotly_chart(fig_skills)

    # """

#==========================================TOTAL JOBS

st.header("", divider="gray")

st.markdown("""            
<h2 style='text-align: center; font-size: 35px'> 
     Match Your CV with the Best Industry Opportunities! 🚀
</h2>
            
<p style='text-align: center; font-size: 25px'>
    Ready to take your career to the next level? 
    Upload your CV and let our tool work its magic by matching your skills with the hottest job openings available right now! ✨
</p>
""", unsafe_allow_html=True)

st.header("", divider="gray")


#===========================================

skills_matched_cv_capitalized = []
# Function to extract text from an uploaded PDF file
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ' '.join(page.extract_text() for page in reader.pages)
    return text

# File uploader widget in Streamlit
uploaded_file = st.file_uploader("Upload your CV in PDF format", type="pdf")

if uploaded_file is not None:
    st.success("Your CV was successfully uploaded! It's being processed ...")
    # If a file is uploaded, extract and display the text
    cv_text = extract_text_from_pdf(uploaded_file)
    #st.text_area("Extracted Text", cv_text, height=300)
    #Process the job description
    text_preprocessor_cv = TextPreprocessor(processing_mode='none')
    cv_text_cleaned = text_preprocessor_cv.preprocess(cv_text)
    #print(type(cv_text_cleaned))
    #st.text_area("Extracted Text", cv_text_cleaned, height=300)


    total_skills_cv, skills_matched_cv, resume_emb = job_desc_emb(cv_text_cleaned)

    
    st.markdown("""
    <h2 style='text-align: center; font-size: 24px;'> 
        🌟 You are a Rockstar! 🌟
    </h2>
    <p style='text-align: center; font-size: 18px;'>
        These are the skills we found in your CV! 🎯
    </p>
    """, unsafe_allow_html=True)
    # Convert the list into a bullet point format
    

    
    # Capitalize each skill and format them for display
    skills_matched_cv_capitalized = [skill.title() for skill in skills_matched_cv]

    # Split skills into 3 columns for display
    cols = st.columns(2)  # Adjust the number of columns based on how you want to organize the skills

    # Display skills in the columns
    for i, skill in enumerate(skills_matched_cv_capitalized):
        cols[i % 2].write(f"🌟 {skill}")


    st.header("", divider="gray")
    result_df = pd.DataFrame(columns=['Position Title', 'Company Name', 'Similarity Score'])

   # Loop through the job descriptions and calculate similarity with the CV
    for i, job_desc_embs in enumerate(job_desc_embeddings_total):
        job_desc_emb_2d = np.array(job_desc_embs).reshape(1, -1)  # Converts to 2D (1 row)
        resume_emb_2d = np.array(resume_emb).reshape(1, -1)  # Converts to 2D (1 row)
        #print(job_desc_emb.shape)
        #print(job_desc_emb_2d.shape)
        #print(resume_emb.shape)
        #print(resume_emb_2d.shape)
        # Calculate cosine similarity between the CV and the current job description
        similarity = cosine_similarity(job_desc_emb_2d, resume_emb_2d)[0][0]
        #print(similarity)

        job_position = jobs_df.loc[i, 'position_title']  # Assuming 'jobID' is the index of jobs_df
        company_name = jobs_df.loc[i, 'company_name']
        # Append the similarity result to the DataFrame
        # Create a new row to be added
        new_row = pd.DataFrame({
        'Position Title': [job_position],  #
        'Company Name': [company_name],  #
        'Similarity Score': [similarity]

        })
    
        # Concatenate the new row with the existing DataFrame
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    # Filter for jobs with similarity score above threshold
    result_df = result_df[result_df['Similarity Score'] > 0.8]

    # Sort by similarity in descending order
    result_df = result_df.sort_values(by='Similarity Score', ascending=False)

    # Get the top 10 jobs
    top_10_jobs = result_df.head(10)

    st.markdown("""
        <h2 style='text-align: center; font-size: 24px;'> 
            Here are the Top Job Matches Just for You 🚀
        </h2>
        <p style='text-align: center; font-size: 25px'>
            We have analyzed your skills and matched them with the current opportunities in the market. <br> 

       </p>
        """, unsafe_allow_html=True)

    # Display the result
    st.write(top_10_jobs)

st.header("", divider="gray")

st.markdown("""
<h2 style='text-align: center; font-size: 35px'> 
    Do you have a specific job in mind?
</h2>
            
            
<p style='text-align: center; font-size: 25px'>
    Just add the job description below and watch as we compare your skillset with the exact requirements for the role. <br>

</p>            
<p style='text-align: center; font-size: 20px'>
    Let's see how you align with your next big opportunity! 🙌 
</p>
""", unsafe_allow_html=True)

#st.header("", divider="gray")

# Default job description
example_job_desc = "We are looking for a data scientist with experience in machine learning and Python and SQL. Knowledge in database and version control and webscrapping."

# Create a text area input with the default job description
new_job_desc = st.text_area("Enter the job description:", example_job_desc, height=150)

# Display the job description (either modified or default)

#st.write(new_job_desc)


#Process the job description
text_preprocessor = TextPreprocessor(processing_mode='none')
new_job_cleaned = text_preprocessor.preprocess(new_job_desc)


total_skills, skills_matched, job_desc_embeddings = job_desc_emb(new_job_cleaned)


# Capitalize each skill and format them for display
skills_matched_jd_capitalized = [skill.title() for skill in skills_matched]
st.markdown("""

<p style='text-align: center; font-size: 18px;'>
    These are the skills we found in the Job Description! 🎯
</p>
""", unsafe_allow_html=True)

# Split skills into 3 columns for display
cols = st.columns(2)  # Adjust the number of columns based on how you want to organize the skills

# Display skills in the columns
for i, skill in enumerate(skills_matched_jd_capitalized):
    cols[i % 2].write(f"🎯 {skill}")




if skills_matched_cv_capitalized:
    # Find common skills between CV and Job Description
    common_skills = set(skills_matched_cv_capitalized) & set(skills_matched_jd_capitalized)
    print(skills_matched_cv_capitalized)
    print(skills_matched_jd_capitalized)
    print(common_skills)
    # Find skills required by JD but not present in CV
    missing_skills = set(skills_matched_jd_capitalized) - set(skills_matched_cv_capitalized)
    print(missing_skills)
    # Display common skills
    if common_skills:
        st.markdown("""
        <h3 style='text-align: center; font-size: 20px;'> 
            🎯 You have these skills that match the job requirements!
        </h3>
        """, unsafe_allow_html=True)
        
        common_skills_list = "\n".join([f"- {skill}" for skill in common_skills])
        st.markdown(common_skills_list)
    
    # Display missing skills
    if missing_skills:
        st.markdown("""
        <h3 style='text-align: center; font-size: 20px;'> 
            ⚠️ You are missing these required skills for the job:
        </h3>
        """, unsafe_allow_html=True)
        
        missing_skills_list = "\n".join([f"- {skill}" for skill in missing_skills])
        st.markdown(missing_skills_list)
else:
    # If skills_matched_cv_capitalized is null or empty
    st.warning("Upload your CV to compare with the discription! ")
# # Create a selection option
# selection = st.radio(
#     "Which skills list would you like to see?",
#     ('Total List', 'Skills Matched')
# )

# # Create a button to display the selected list
# if st.button("Show Skills List"):
#     if selection == 'Total List':
#         st.write("Total Skills List:")
#         st.write(total_skills)
#     else:
#         st.write("Matched Skills List:")
#         st.write(skills_matched)


