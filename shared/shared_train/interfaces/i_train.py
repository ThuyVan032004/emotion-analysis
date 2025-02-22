from abc import ABC, abstractmethod
from scipy.sparse import spmatrix


class ITrain(ABC):
    @abstractmethod
    def train_model(self, train_X: spmatrix, train_y):
        pass

    @abstractmethod
    def get_params(self):
        pass