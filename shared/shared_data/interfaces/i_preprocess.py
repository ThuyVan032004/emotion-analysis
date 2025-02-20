from abc import ABC, abstractmethod
from typing import List
from scipy.sparse import spmatrix


class IPreprocess(ABC):
    @abstractmethod
    def clean_text(self, text: str, stopwords: List[str]) -> str:
        pass

    @abstractmethod
    def stemmer(self, text: str) -> str:
        pass

    @abstractmethod
    def tf_idf_vectorizer(self, text: (str | object)) -> spmatrix:
        pass



