import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

class Trainer:
    def __init__(self, model: Pipeline = None):
        """
        Initialization of the trainer. 
        If the model is not passed, the default configuration is created.
        """
        self.model = self._get_default_pipeline() if model is None else model
            
    def _get_default_pipeline(self) -> Pipeline:
        """Inner pipeline for creating base model."""
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    lowercase=True, stop_words="english", max_features=100_000,
                    ngram_range=(1, 3), min_df=2, sublinear_tf=True
                )),
                ("char", TfidfVectorizer(
                    analyzer="char_wb", lowercase=True, max_features=50_000,
                    ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ))
            ])),
            ("clf", LinearSVC(C=1.0, class_weight="balanced"))
        ])
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Model's training."""
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        """Prediction."""
        return self.model.predict(X)