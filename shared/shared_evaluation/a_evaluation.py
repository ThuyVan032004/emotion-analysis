from sklearn.metrics import classification_report, confusion_matrix
from .interfaces.i_evaluation import IEvaluation


class EvaluationBase(IEvaluation):
    def classification_report(self, prediction, actual_value):
        return classification_report(prediction, actual_value, output_dict=True)

    def confusion_matrix(self, prediction, actual_value):
        return confusion_matrix(prediction, actual_value)
        