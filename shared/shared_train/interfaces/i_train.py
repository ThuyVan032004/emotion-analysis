from abc import ABC, abstractmethod


class ITrain(ABC):
    @abstractmethod
    def train_model(self):
        pass