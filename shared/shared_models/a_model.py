from abc import abstractmethod
from .interfaces.i_model import IModel


class ModelBase(IModel):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass