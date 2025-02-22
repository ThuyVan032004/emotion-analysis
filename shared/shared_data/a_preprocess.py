from scipy.sparse import spmatrix
from typing import List
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from shared.constants import STOPWORDS
from .interfaces.i_preprocess import IPreprocess



class PreprocessBase(IPreprocess):
    def __init__(self):
        self.tf = TfidfVectorizer(min_df=0.0, max_df=1.0, binary=False, ngram_range=(1, 3))

    def clean_text(self, text, stopwords=STOPWORDS):
        # Lower 
        text = text.lower()
        
        # Remove stopwords
        pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        text = pattern.sub("", text)
        
        # Spacing and filters 
        text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
        text = re.sub("[^A-Za-z0-9]+", " ", text)
        text = re.sub(" +", " ", text)
        text = re.sub("http\S+", "", text)
    
        return text

    def stemmer(self, text):
        porter_stemmer = PorterStemmer()
        stemmed_text = ' '.join([porter_stemmer.stem(word) for word in text.split()])
        return stemmed_text

