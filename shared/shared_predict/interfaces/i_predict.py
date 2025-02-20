from abc import ABC, abstractmethod
from scipy.sparse import spmatrix


class IPredict(ABC):
    @abstractmethod
    def prediction(self, test_value: spmatrix):
        pass