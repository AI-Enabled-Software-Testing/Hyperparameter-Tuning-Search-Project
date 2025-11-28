from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from models.base import BaseModel
from .ParamSpace import ParamSpace


class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.estimator: KNeighborsClassifier | None = None

    def create_model(self, **params: Any) -> None:
        self.params.update(params)
        self.estimator = KNeighborsClassifier(**self.params)

    def train(self, X_train: List[np.ndarray], y_train: np.ndarray) -> KNeighborsClassifier:
        if self.estimator is None:
            self.create_model()
        estimator = self.estimator
        assert estimator is not None
        estimator.fit(X_train, y_train)
        return estimator

    def predict(self, X: List[np.ndarray]):
        if self.estimator is None:
            raise RuntimeError(
                "Estimator has not been created. Call create_model() first."
            )
        check_is_fitted(self.estimator)
        return self.estimator.predict(X)

    def predict_proba(self, X: List[np.ndarray]):
        if self.estimator is None:
            raise RuntimeError(
                "Estimator has not been created. Call create_model() first."
            )
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                "Underlying estimator does not support probability prediction."
            )
        check_is_fitted(self.estimator)
        return self.estimator.predict_proba(X)

    def evaluate(self, X_test: List[np.ndarray], y_test: np.ndarray) -> Dict[str, float]:
        if self.estimator is None:
            raise RuntimeError(
                "Estimator has not been created. Call create_model() first."
            )
        predictions = self.predict(X_test)
        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )

        proba = self.estimator.predict_proba(X_test)
        
        metrics: Dict[str, float] = {
            "accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_micro": f1_score(y_test, predictions, average="micro", zero_division=0),
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "roc_auc": roc_auc_score(y_test, proba, average="macro", multi_class="ovr"),
        }

        return metrics

    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            "n_neighbors": ParamSpace.integer(min_val=3, max_val=30, default=5),
            "weights": ParamSpace.categorical(
                choices=["uniform", "distance"], default="uniform"
            ),
            "metric": ParamSpace.categorical(choices=["minkowski", "manhattan", "euclidean", "chebyshev"], default="minkowski")
        }
