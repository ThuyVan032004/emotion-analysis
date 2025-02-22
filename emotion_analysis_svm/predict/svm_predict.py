from scipy.sparse import spmatrix
from shared import PredictBase


class PredictSVM(PredictBase):
    def __init__(self, svm_model):
        self._model = svm_model
    
    def prediction(self, test_value: spmatrix):
        return self._model.predict(test_value)