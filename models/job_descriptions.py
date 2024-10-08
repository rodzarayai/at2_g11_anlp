import streamlit as st
from spacy.matcher import Matcher
import pandas as pd
import pickle
import json
from aux_func import *
from transformers import AutoTokenizer, BertModel
import os


#This allows run torch in streamlit
torch.set_num_threads(1)



def job_desc_emb(new_job_cleaned):

###################### JD  SECTION
# Get paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '../data')

    #test with custom job desc. This shouls be replaced by a csv file


    #=====================================================================Load the files

    #Open files. Depending on where we have run these notebooks we can use different namings
    with open(os.path.join(data_dir, 'job_desc_embeddings_skills.pkl'), 'rb') as f:
    #with open('../data/job_desc_embeddings_skills.pkl', 'rb') as f:
    #with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings_skills.pkl', 'rb') as f: #using in colab
        skill_embeddings = pickle.load(f)
        skill_embeddings = np.squeeze(skill_embeddings, axis=1)

    with open(os.path.join(data_dir, 'job_desc_embeddings.pkl'), 'rb') as f:
    #with open('../data/job_desc_embeddings.pkl', 'rb') as f:
    #with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings.pkl', 'rb') as f:
        job_desc_embeddings = pickle.load(f)

    # Load patterns from the JSONL file
    skills_patterns = []
    with open(os.path.join(data_dir, 'jz_skill_patterns.jsonl'), 'rb') as f:
    #with open('../data/jz_skill_patterns.jsonl', 'r') as f:
    #with open('/content/drive/MyDrive/AT2/data/jz_skill_patterns.jsonl', 'r') as f:
        for line in f:
            skills_patterns.append(json.loads(line))

    skills_path = os.path.join(data_dir, '../data/skills.json')
    skills_list = load_skills_from_json(skills_path)
    #skills_list = load_skills_from_json('/content/drive/MyDrive/AT2/data/skills.json')


    #=====================================================================PROCESSING


    #Get skills list from simpler matcher - 
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)

    # Add patterns to the matcher
    for pattern in skills_patterns:
        matcher.add(pattern['label'], [pattern['pattern']])
        
    #Get skills
    skills_matched = list(find_skills(new_job_cleaned, matcher))
    #print('2') 

    #=================================================LOAD BERT MODEL
    # Load the model and tokenizer only once
    #model_name = "bert-base-uncased"
    model_name = "bert-base-uncased"
    device = torch.device("cpu")  # Use CPU for stability
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    #st.write("Model and tokenizer loaded in a separate process.")
    #print('5') 

    #print('6') 
    job_desc_embedding = get_embeddings(new_job_cleaned, model_name, device, model)
    print("Model and tokenizer loaded from pre-saved state.")

    THRESHOLD = 0.68
    top_skills = find_top_skills(job_desc_embedding, skill_embeddings, skills_list, THRESHOLD)

    if not top_skills:
        #st.warning("No skills found that exceed the threshold.")
        skills_list, scores_list = [], []
    else:
        skills_list, scores_list = zip(*top_skills)
        skills_list = [skill.replace('-', ' ') for skill in skills_list]
    #skills_list, scores_list = zip(*top_skills)
    #format the skills without the -
    
    total_list = list(set(skills_list + skills_matched))

    return total_list, skills_matched, job_desc_embedding

#It takes so long to load. It is changed by a pre-loaded csv file with the set of skills previously calculated.
def get_skills_df(df):

###################### JD  SECTION
# Get paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '../data')

    #test with custom job desc. This shouls be replaced by a csv file

    #=====================================================================Load the files

    #Open files. Depending on where we have run these notebooks we can use different namings
    with open(os.path.join(data_dir, 'job_desc_embeddings_skills.pkl'), 'rb') as f:
    #with open('../data/job_desc_embeddings_skills.pkl', 'rb') as f:
    #with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings_skills.pkl', 'rb') as f: #using in colab
        skill_embeddings = pickle.load(f)
        skill_embeddings = np.squeeze(skill_embeddings, axis=1)

    with open(os.path.join(data_dir, 'job_desc_embeddings.pkl'), 'rb') as f:
    #with open('../data/job_desc_embeddings.pkl', 'rb') as f:
    #with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings.pkl', 'rb') as f:
        job_desc_embeddings = pickle.load(f)

    # Load patterns from the JSONL file
    skills_patterns = []
    with open(os.path.join(data_dir, 'jz_skill_patterns.jsonl'), 'rb') as f:
    #with open('../data/jz_skill_patterns.jsonl', 'r') as f:
    #with open('/content/drive/MyDrive/AT2/data/jz_skill_patterns.jsonl', 'r') as f:
        for line in f:
            skills_patterns.append(json.loads(line))

    skills_path = os.path.join(data_dir, '../data/skills.json')
    skills_list = load_skills_from_json(skills_path)
    #skills_list = load_skills_from_json('/content/drive/MyDrive/AT2/data/skills.json')


    #=====================================================================PROCESSING


    #Get skills list from simpler matcher - 
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)

    # Add patterns to the matcher
    for pattern in skills_patterns:
        matcher.add(pattern['label'], [pattern['pattern']])
        
    #Get skills
    text_preprocessor = TextPreprocessor(processing_mode='none')
    df = text_preprocessor.preprocess_dataframe(df, 'job_description')
    df['skills_matched_cleaned'] = df['job_description_processed_cleaned'].apply(find_skills, matcher=matcher)

    jobs_skills = [skill for sublist in df['skills_matched_cleaned'] for skill in sublist]

    # Count the occurrences of each skill
    skill_counts = Counter(jobs_skills)

    # Convert the Counter object to a DataFrame for plotting
    skills_df = pd.DataFrame(skill_counts.items(), columns=['skill', 'count'])

    return skills_df


