import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import bentoml
from sklearn.model_selection import train_test_split
from emotion_analysis_svm.data import RepositorySVM, PreprocessSVM
from emotion_analysis_svm.evaluation import EvaluationSVM
from emotion_analysis_svm.train import TrainSVM
from emotion_analysis_svm.predict import PredictSVM


class TfIdfVectorizer(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def predict(self, model_input):
        return self.vectorizer.transform([model_input])


class Workloads:
    def __init__(self, repository, preprocessor, evaluator, trainer):
        self._repository = repository
        self._preprocessor = preprocessor
        self._evaluator = evaluator
        self._trainer = trainer

    def svm_workloads(self):
        with mlflow.start_run():
            data_frame = self._repository.get_all()
            data_frame.text = data_frame.text.apply(self._preprocessor.clean_text)
            data_frame.text = data_frame.text.apply(self._preprocessor.stemmer)
            
            train_df, val_df = train_test_split(data_frame, test_size=0.2,
                                                stratify=data_frame['label'], random_state=42)
            
            tf = self._preprocessor.tf
            tf_train = tf.fit_transform(train_df.text)
            tf_validation = tf.transform(val_df.text)

            svm_model = self._trainer.train_model(tf_train, train_df.label)

            self._predictor = PredictSVM(svm_model=svm_model)
            y_pred = self._predictor.prediction(tf_validation)

            params = self._trainer.get_params()
            report = self._evaluator.classification_report(y_pred, val_df.label)

            for label in report.keys():
                if isinstance(report[label], dict):
                    for metric, value in report[label].items():
                        mlflow.log_metric(f"{label}_{metric}", value)
                
            mlflow.log_params(params)

            mlflow.sklearn.log_model(
                sk_model=svm_model,
                artifact_path='svm-models',
                registered_model_name="sklearn-svm-model",
            )

            mlflow.pyfunc.log_model(
                python_model=TfIdfVectorizer(tf),
                artifact_path='tfidf-models',
                registered_model_name='tfidf-vectorizer',
            )

            client = MlflowClient()
            svm_model_name = "sklearn-svm-model"
            svm_latest_version = client.get_latest_versions(svm_model_name)[0].version

            bentoml.mlflow.import_model(
                "sklearn-svm-model",
                f"models:/sklearn-svm-model/{svm_latest_version}",
                signatures={"predict": {"batchable": True}},
            )

            # Import TF-IDF vectorizer
            tfidf_model_name = "tfidf-vectorizer"
            tfidf_latest_version = client.get_latest_versions(tfidf_model_name)[0].version
            
            bentoml.mlflow.import_model(
                "tfidf-vectorizer",
                f"models:/tfidf-vectorizer/{tfidf_latest_version}",
                signatures={"predict": {"batchable": True}},
            )


if __name__ == "__main__":
    svm_repository = RepositorySVM('datasets/text.csv')
    svm_preprocessor = PreprocessSVM()
    svm_evaluator = EvaluationSVM()
    svm_trainer = TrainSVM()

    workloads = Workloads(svm_repository, svm_preprocessor,
                          svm_evaluator, svm_trainer)
    
    workloads.svm_workloads()