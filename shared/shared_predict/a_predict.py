from abc import abstractmethod
from scipy.sparse import spmatrix
from .interfaces.i_predict import IPredict


class PredictBase(IPredict):
    @abstractmethod
    def prediction(self, test_value: spmatrix):
        pass