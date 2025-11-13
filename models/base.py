"""Abstract interface for models used in the hyperparameter tuning framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Literal

from .ParamSpace import ParamSpace


class BaseModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    @abstractmethod
    def create_model(self, **params: Any) -> None:
        """Instantiate or reconfigure the underlying model."""
        raise NotImplementedError

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Generate predictions."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Evaluate the model and return a dictionary of metrics."""
        raise NotImplementedError

    @abstractmethod
    def get_param_space(self) -> Dict[str, ParamSpace]:
        """Return the searchable hyperparameter space."""
        raise NotImplementedError


def get_model_by_name(model_name: Literal["dt", "knn", "cnn"]) -> BaseModel:
    """Factory function to get model by name."""
    from models.decision_tree import DecisionTreeModel
    from models.knn import KNNModel
    from models.cnn import CNNModel

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
