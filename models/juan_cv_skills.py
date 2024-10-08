#NER Model for one CV

#Download relevant packages

import spacy
nlp = spacy.load('en_core_web_sm')

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
import seaborn as sns
import matplotlib.pyplot as plt
import time
import json
from collections import Counter
from collections import defaultdict
import torch
import docx2txt
import random
import PyPDF2
import os


from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModel
from subprocess import list2cmdline
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import Matcher
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


#Text extraction depending on file type

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '.join(page.extract_text() for page in reader.pages)
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return ' '.join(' '.join(row) for row in reader)

def extract_text(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext.lower() == '.docx':
        return extract_text_from_docx(file_path)
    elif ext.lower() == '.csv':
        return extract_text_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
#Data prepocessing 

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

#Loading predetermined skills from set skills list 

def load_skill_patterns(jsonl_file):
    skill_patterns = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                pattern = data.get('pattern', [])
                skill = ' '.join([item.get('LOWER', '') for item in pattern if isinstance(item, dict)])
                if skill:
                    skill_patterns.append(skill)
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line}")
    return skill_patterns

#Defining embeddings function 

@torch.no_grad()
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

#Functions to extract skills from individual CVs based on embeddings 

def extract_skills(resume_embedding, skill_embeddings, skill_patterns, threshold=0.75):
    similarities = cosine_similarity([resume_embedding], skill_embeddings)[0]
    return [skill for skill, sim in zip(skill_patterns, similarities) if sim > threshold]

def process_resume(file_path, jsonl_file):
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    skill_patterns = load_skill_patterns(jsonl_file)

    with tqdm(total=100, desc="Processing resume") as pbar:
        resume_text = extract_text(file_path)
        pbar.update(20)

        preprocessed_text = preprocess_text(resume_text)
        pbar.update(20)

        resume_embedding = get_embedding(preprocessed_text, tokenizer, model)
        pbar.update(20)

        skill_embeddings = [get_embedding(skill, tokenizer, model) for skill in skill_patterns]
        pbar.update(20)

        extracted_skills = extract_skills(resume_embedding, skill_embeddings, skill_patterns)
        pbar.update(20)

    skill_counts = Counter()
    for skill in extracted_skills:
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', preprocessed_text, re.IGNORECASE))
        if count > 0:
            skill_counts[skill] = count

    return skill_counts

#Main function 

def main():
    resume_file = "10554236.pdf"  # Replace with your resume file path
    jsonl_file = "jz_skill_patterns.jsonl"

    try:
        skill_counts = process_resume(resume_file, jsonl_file)

        print("\nTop 20 Extracted Skills:")
        for skill, count in skill_counts.most_common(20):
            print(f"  {skill} (Count: {count})")

        print(f"\nTotal unique skills extracted: {len(skill_counts)}")
    except Exception as e:
        print(f"Error processing {resume_file}: {str(e)}")

if __name__ == "__main__":
    main()
