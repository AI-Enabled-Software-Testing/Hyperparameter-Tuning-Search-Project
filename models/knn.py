from typing import Any, Dict

from sklearn.metrics import classification_report
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

    def train(self, X_train, y_train) -> KNeighborsClassifier:
        if self.estimator is None:
            self.create_model()
        estimator = self.estimator
        assert estimator is not None
        estimator.fit(X_train, y_train)
        return estimator

    def predict(self, X):
        if self.estimator is None:
            raise RuntimeError(
                "Estimator has not been created. Call create_model() first."
            )
        check_is_fitted(self.estimator)
        return self.estimator.predict(X)

    def predict_proba(self, X):
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

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        predictions = self.predict(X_test)
        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        return {
            "accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }

    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            "n_neighbors": ParamSpace.integer(min_val=3, max_val=15, default=5),
            "weights": ParamSpace.categorical(
                choices=["uniform", "distance"], default="uniform"
            ),
            "metric": ParamSpace.categorical(
                choices=["minkowski", "manhattan"], default="minkowski"
            ),
        }
