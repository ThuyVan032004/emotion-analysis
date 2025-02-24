from abc import ABC, abstractmethod
from typing import List
from scipy.sparse import spmatrix


class IPreprocess(ABC):
    @abstractmethod
    def clean_text(self, text, stopwords):
        pass

    @abstractmethod
    def stemmer(self, text):
        pass




