import pickle
import pandas as pd
import numpy as np
import json

#Text processing and cleaning
import contractions # To include english contractions
import re #regex
import string #used to include punctuation during text processing
from collections import Counter #count strings in texts

#Natural Language Tool Kit NLK package
import nltk
from nltk.corpus import stopwords #Stopwords

import spacy
from spacy.matcher import Matcher
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity




def load_skills_from_json(skills_file):
    skills_list = []
    with open(skills_file, 'r') as f:
        data = json.load(f)  # Load the entire JSON file
        for skill in data.keys():  # The keys at the top level are the skill names
            skills_list.append(skill)  # Add the skill to the list
    return skills_list


#=======================================Processing class
class TextPreprocessor:
    def __init__(self, processing_mode='none', custom_punctuation=None, custom_stopwords=None, sentence_analysis=False):
        """
            Initialization considers Custom punctuation, Stop words, and Lemmatizer or Stemmer.
            Updates custom punctuation and custom stop words set with additional ones if provided.
            The processing mode to standardise variants can be choose between none, Stem and Lemma. Each mode is stored in a different
            column of the dataframe.
            Sentence analysis parameter is used to keep the punctuation symbols required for sentence analysis.

            Parameters:
            - processing_mode: String to decide whether to use 'lemma', 'stem', or 'none' for text processing.
            - custom_punctuation: Additional punctuation characters to remove from text.
            - custom_stopwords: Additional stopwords to remove from text.
            - sentence_analysis: Boolean indicating sentence analysis cleaning steps. This mode will keep the punctuation symbols.

            """

        self.punctuation = string.punctuation #Init with all punctuation characters

        if custom_punctuation:
            self.punctuation += custom_punctuation #add custom punctuation

        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords) #add custom stopwords

        # Determine which text processing mode to use
        self.processing_mode = processing_mode.lower()

        # Set the sentence analysis mode
        self.sentence_analysis = sentence_analysis

        #Set the variant standardization mode
        if self.processing_mode == 'lemma':
            self.lemmatizer = WordNetLemmatizer()
        elif self.processing_mode == 'stem':
            self.stemmer = PorterStemmer()

    # Expand contractions using the contractions library
    def expand_contractions(self, text):
        return contractions.fix(text)

    # Split hyphenated words into separate words, like phone numbers or radio fm, age, etc.
    def split_hyphenated_words(self, text):
        return re.sub(r'-', ' ', text)

    def remove_punctuation(self, text):
        return ''.join([char for char in text if char not in self.punctuation])

    def add_space_after_parenthesis(self, text):
        return re.sub(r'\)', ') ', text)

    def to_lowercase(self, text):
        return text.lower()

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        return ' '.join([word for word in words if word not in self.stop_words])

    def remove_extra_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def stem_words(self, text):
        words = word_tokenize(text)
        return ' '.join([self.stemmer.stem(word) for word in words])

    def lemmatize_words(self, text):
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

    # Order matters
    def preprocess(self, text):
        text = self.expand_contractions(text)
        text = self.split_hyphenated_words(text)
        text = self.add_space_after_parenthesis(text)

        #In case we need to analyse sentences, we will need the punctuations
        if not self.sentence_analysis:
            text = self.remove_punctuation(text)
        text = self.to_lowercase(text)
        #The stopwords are removed if the users wants to standardise variants.
        #If none is selected, the ouput will just perform previous cleaning steps
        if self.processing_mode != 'none':
            text = self.remove_stopwords(text)

        text = self.remove_extra_whitespace(text)

        #Select the processing mode for variants
        if self.processing_mode == 'lemma':
            text = self.lemmatize_words(text)
        elif self.processing_mode == 'stem':
            text = self.stem_words(text)

        return text

    #Apply preprocessing steps to daframe and create a column base on the processing mode
    def preprocess_dataframe(self, df, column_name):
        if not self.sentence_analysis:
            if self.processing_mode == 'lemma':
                df[f'{column_name}_processed_lemma'] = df[column_name].apply(self.preprocess)
            elif self.processing_mode == 'stem':
                df[f'{column_name}_processed_stem'] = df[column_name].apply(self.preprocess)
            else:  # If 'none', apply preprocessing without lemma or stem
                df[f'{column_name}_processed_cleaned'] = df[column_name].apply(self.preprocess)
        else: # Add different processed columns for sentences
            if self.processing_mode == 'lemma':
                df[f'{column_name}_processed_lemma_sent'] = df[column_name].apply(self.preprocess)
            elif self.processing_mode == 'stem':
                df[f'{column_name}_processed_stem_sent'] = df[column_name].apply(self.preprocess)
            else:  # If 'none', apply preprocessing without lemma or stem
                df[f'{column_name}_processed_cleaned_sent'] = df[column_name].apply(self.preprocess)
        return df

#=============================== GET THE EMBEDDINGS

def get_embeddings(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().to("cpu").numpy()
    return embeddings


#============================== GET THE SKILLS

# Define a function to apply the matcher and find skills in the text
def find_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()  # To store found skills
    for match_id, start, end in matches:
        skill = doc[start:end].text
        skills.add(skill)
    return skills


def find_top_skills(job_desc_embedding, skill_embeddings, skills_list, threshold=0.55):
    # Ensure that job_desc_embedding is 2D before passing to cosine_similarity
    #job_desc_embedding = np.expand_dims(job_desc_embedding, axis=0)  # Make it 2D
    similarities = cosine_similarity(job_desc_embedding, skill_embeddings).flatten()

    # Find all skills with similarity scores above the threshold
    above_threshold_indices = [i for i, score in enumerate(similarities) if score >= threshold]

    # Get the skills and scores for those above the threshold
    top_skills = [skills_list[i] for i in above_threshold_indices]
    top_scores = [similarities[i] for i in above_threshold_indices]

    # Return both the skills and their similarity scores
    return list(zip(top_skills, top_scores))

@st.cache_resource
def load_model():


    # Select device (MPS for Mac, CUDA for NVIDIA GPUs, CPU as a fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device