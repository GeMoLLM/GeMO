import json
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import contractions
import string

def read_in_texts(file_path, indices):
    texts = []
    with open(file_path) as f:
        for line in f:
            texts.append(json.loads(line))
    return [texts[i] for i in indices]

def read_in_src_texts(file_path):
    texts = []
    with open(file_path) as f:
        for line in f:
            texts.append(json.loads(line)['review_text'])
    return texts

def get_wordfreq(corpus):
    lemmatizer = WordNetLemmatizer()

    corpus = ' '.join(corpus)
    
    corpus = corpus.lower()
    
    # Expand contractions
    corpus = contractions.fix(corpus)

    # Tokenize the corpus
    tokens = word_tokenize(corpus)

    # Remove punctuation
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
    
    # Remove punctuation and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    # Lemmatize the tokens
    lemm_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Build the dictionary
    word_dict = {}
    for lemm_token, token in zip(lemm_tokens, tokens):
        if lemm_token not in word_dict:
            word_dict[lemm_token] = []
        word_dict[lemm_token].append(token)

    # Build the word frequency table
    word_freq = Counter(tokens)
    return word_freq

def calculate_counter_similarity(counter1, counter2):
    all_keys = set(counter1.keys()) | set(counter2.keys())
    dot_product = sum(counter1.get(key, 0) * counter2.get(key, 0) for key in all_keys)
    norm1 = sum(counter1.get(key, 0) ** 2 for key in all_keys) ** 0.5
    norm2 = sum(counter2.get(key, 0) ** 2 for key in all_keys) ** 0.5
    return dot_product / (norm1 * norm2)

def calculate_counter_entropy(counter):
    total = sum(counter.values())
    entropy = 0
    for key, value in counter.items():
        p = value / total
        entropy -= p * np.log2(p)
    return entropy