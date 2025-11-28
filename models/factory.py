"""Factory function for creating model instances by name."""

from typing import Literal, overload
from models.cnn import CNNModel
from models.decision_tree import DecisionTreeModel
from models.knn import KNNModel

@overload
def get_model_by_name(model_name: Literal["dt"]) -> DecisionTreeModel:
    ...


@overload
def get_model_by_name(model_name: Literal["knn"]) -> KNNModel:
    ...


@overload
def get_model_by_name(model_name: Literal["cnn"]) -> CNNModel:
    ...


def get_model_by_name(model_name: Literal["dt", "knn", "cnn"]) -> KNNModel | DecisionTreeModel | CNNModel:

    models = {
        "dt": DecisionTreeModel,
        "knn": KNNModel,
        "cnn": CNNModel,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(models.keys())}"
        )

    return models[model_name]()

