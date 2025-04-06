import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  #remove numbers
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  #remove URLs
    text = re.sub(r"[^\w\s]", "", text)  #remove punctuation
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def add_text_features(df):
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    return df