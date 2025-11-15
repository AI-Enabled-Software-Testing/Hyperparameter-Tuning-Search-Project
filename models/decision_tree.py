from typing import Any, Dict

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from models.base import BaseModel
from .ParamSpace import ParamSpace


class DecisionTreeModel(BaseModel):
    """Decision Tree classifier wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.estimator: DecisionTreeClassifier | None = None

    def create_model(self, **params: Any) -> None:
        """Create the underlying sklearn estimator."""
        configuration = {**self.params, **params}
        self.estimator = DecisionTreeClassifier(**configuration)

    def train(self, X_train, y_train) -> DecisionTreeClassifier:
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

        metrics: Dict[str, float] = {
            "accuracy": report["accuracy"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }

        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_test, proba, average="weighted", multi_class="ovr"
                )

        return metrics

    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            "max_depth": ParamSpace.integer(min_val=3, max_val=20, default=10),
            "min_samples_split": ParamSpace.integer(min_val=2, max_val=20, default=5),
            "min_samples_leaf": ParamSpace.integer(min_val=1, max_val=10, default=2),
            "criterion": ParamSpace.categorical(
                choices=["gini", "entropy"], default="gini"
            )
        }
