import bentoml
from typing import List
from emotion_analysis_svm.data import PreprocessSVM

preprocessor = PreprocessSVM()
tf = bentoml.models.get("tfidf-vectorizer:latest")
svm = bentoml.models.get("sklearn-svm-model:latest")


def preprocess(text, runner):
    cleaned_text = preprocessor.clean_text(text)
    stemmed_text = preprocessor.stemmer(cleaned_text)
    tf_text = runner.predict(stemmed_text)
    return tf_text


@bentoml.service(
    resources={"cpu": 2},
    traffic={"timeout": 10},
)
class EmotionAnalysisSVM:
    def __init__(self):
        self.svm_model = bentoml.mlflow.load_model(svm)
        self.tf_model = bentoml.mlflow.load_model(tf)

    @bentoml.api
    def predict(self, input_text: (str | List[str])):
        try:
            if isinstance(input_text, str):
                preprocessed_text = preprocess(input_text, self.tf_model)
                result = self.svm_model.predict(preprocessed_text)
            # if isinstance(input_text, list):
            #     preprocessed_list = []
            #     for text in input_text:
            #         new_text = preprocess(text, self.tf_model)
            #         preprocessed_list.append(new_text)
                
            #     result = self.svm_model.predict(preprocessed_list)
                if result == 0:
                    result = "sad"
                if result == 1:
                    result = "joyful"
                if result == 2:
                    result = "love"
                if result == 3:
                    result = "anger"
                if result == 4:
                    result = "fear"
                if result == 5:
                    result = "suprise"
                return {"prediction": result}
        except Exception as e:
            return {"error": str(e)}


    
