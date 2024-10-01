


#=====================================================================Load the files

with open('../data/job_desc_embeddings_skills.pkl', 'rb') as f:
#with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings_skills.pkl', 'rb') as f: #using in colab
    skill_embeddings = pickle.load(f)
    skill_embeddings = np.squeeze(skill_embeddings, axis=1)

with open('../data/job_desc_embeddings.pkl', 'rb') as f:
#with open('/content/drive/MyDrive/AT2/data/job_desc_embeddings.pkl', 'rb') as f:
    job_desc_embeddings = pickle.load(f)


# Load patterns from the JSONL file
skills_patterns = []
with open('../data/jz_skill_patterns.jsonl', 'r') as f:
#with open('/content/drive/MyDrive/AT2/data/jz_skill_patterns.jsonl', 'r') as f:
    for line in f:
        skills_patterns.append(json.loads(line))

skills_list = load_skills_from_json('../data/skills.json')
#skills_list = load_skills_from_json('/content/drive/MyDrive/AT2/data/skills.json')


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Add patterns to the matcher
for pattern in skills_patterns:
    matcher.add(pattern['label'], [pattern['pattern']])
    
    
    
text_preprocessor = TextPreprocessor(processing_mode='none')
new_job_cleaned = text_preprocessor.preprocess(new_job_desc)

skills_matched = list(find_skills(new_job_cleaned))


# Load the model and tokenizer only once
model_name = "bert-base-uncased"
model, tokenizer, device = load_model()
job_desc_embedding = get_embeddings(new_job_cleaned, model_name)

THRESHOLD = 0.65
top_skills = find_top_skills(job_desc_embedding, skill_embeddings, skills_list, THRESHOLD)

skills_list, scores_list = zip(*top_skills)
#format the skills without the -
skills_list = [skill.replace('-', ' ') for skill in skills_list]

total_list = list(set(skills_list + skills_matched))