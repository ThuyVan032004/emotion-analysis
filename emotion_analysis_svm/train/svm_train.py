from sklearn.svm import SVC
from scipy.sparse import spmatrix
from shared import TrainBase


class TrainSVM(TrainBase):
    def __init__(self):
        self._svc = SVC(kernel='linear')
        
    def train_model(self, train_X: spmatrix, train_y):
        svm_model = self._svc.fit(train_X, train_y)
        return svm_model