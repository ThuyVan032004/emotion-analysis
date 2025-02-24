from abc import ABC, abstractmethod


class IEvaluation(ABC):
    @abstractmethod
    def classification_report(self, prediction, actual_value):
        pass

    @abstractmethod
    def confusion_matrix(self, prediction, actual_value):
        pass


