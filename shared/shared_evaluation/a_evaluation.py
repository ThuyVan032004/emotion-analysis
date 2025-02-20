import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from .interfaces.i_evaluation import IEvaluation


class EvaluationBase(IEvaluation):
    def classification_report(self, prediction, actual_value):
        print(classification_report(prediction, actual_value))

    def confusion_matrix(self, prediction, actual_value):
        matrix = confusion_matrix(prediction, actual_value)
        
        plt.figure(figsize=(10, 3))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Prediction')
        plt.ylabel('Actual Value')
        plt.show()
        