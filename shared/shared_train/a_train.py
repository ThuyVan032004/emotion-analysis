from abc import abstractmethod
from scipy.sparse import spmatrix
from .interfaces.i_train import ITrain


class TrainBase(ITrain):
    @abstractmethod
    def train_model(self, train_X: spmatrix, train_y):
        pass