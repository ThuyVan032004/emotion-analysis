from abc import abstractmethod
from .interfaces.i_predict import IPredict


class PredictBase(IPredict):
    @abstractmethod
    def prediction(self, test_value):
        pass