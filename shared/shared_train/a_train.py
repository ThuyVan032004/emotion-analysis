from abc import abstractmethod
from .interfaces.i_train import ITrain


class TrainBase(ITrain):
    @abstractmethod
    def train_model(self):
        pass