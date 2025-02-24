from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass



