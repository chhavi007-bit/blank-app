import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import gspread

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

df = pd.read_csv("gpt-4.csv")
st.write(df.head())

sample_df = df.sample(n=1000, random_state=42)
print(sample_df)

import re
def clean_text(text):

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample_df['text_column'] = sample_df['data'] + ' ' + sample_df['conversation'
]
sample_df['text_column'] = sample_df['text_column'].apply(clean_text)
print(sample_df['text_column'])
sample_df['text_column'].describe()

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


sample_df['processed_text'] = sample_df['text_column'].apply(preprocess_text)
print(sample_df['processed_text'])

from collections import Counter
word_counts = Counter(" ".join(sample_df['processed_text']).split())
print(word_counts.most_common(100))
