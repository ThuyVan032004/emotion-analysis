from abc import abstractmethod
from scipy.sparse import spmatrix
from typing import List
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from .interfaces.i_preprocess import IPreprocess


class PreprocessBase(IPreprocess):
    def clean_text(self, text: str, stopwords: List[str]) -> str:
        lower_text = text.lower()
   
        stopwords_pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        removed_stopwords_text = stopwords_pattern.sub("", lower_text)
   
        punctuation_spaced_text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", removed_stopwords_text)
        alphanumeric_text = re.sub("[^A-Za-z0-9]+", " ", punctuation_spaced_text)
        multi_spaced_text = re.sub(" +", " ", alphanumeric_text)
        cleaned_text = re.sub("http\S+", "", multi_spaced_text)
     
        return cleaned_text

    def stemmer(self, text: str) -> str:
        porter_stemmer = PorterStemmer()
        stemmed_text = ' '.join([porter_stemmer.stem(word) for word in text.split()])
        return stemmed_text

    def tf_idf_vectorizer(self, text: (str | object)) -> spmatrix:
        tf = TfidfVectorizer(min_df=0.0, max_df=1.0, binary=False, ngram_range=(1, 3))
        tf_text = tf.fit_transform(text)
        
        return tf_text
