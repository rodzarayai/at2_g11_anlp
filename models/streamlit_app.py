import streamlit as st
from job_descriptions import *
from aux_func import *
import PyPDF2

#====================================================================PAGE CONFIG
apptitle = 'CV Job Matcher App'

st.set_page_config(page_title=apptitle, 
                   page_icon="💼")


st.title("CV Job Matcher")

st.subheader("Match your CV with the last published offers in the web!")


#=================================job_desc 

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


total_skills, skills_matched, job_desc_emb = job_desc_emb(new_job_cleaned)


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
    text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", text, height=300)