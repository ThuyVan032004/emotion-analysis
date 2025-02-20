from abc import ABC, abstractmethod


class IPredict(ABC):
    @abstractmethod
    def prediction(self, test_value):
        pass