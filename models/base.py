"""Machine learning models for hyperparameter tuning experiments.

This module contains base class for a general ML model (to be inherited)
that will be used in hyperparameter optimization experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs
        
    @abstractmethod
    def create_model(self, **params):
        """Create model instance with given parameters."""
        pass
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        """Get parameter space for hyperparameter tuning."""
        pass
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model."""
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate model and return accuracy."""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        """Perform cross-validation and return mean score."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores.mean()
    
def get_model_by_name(model_name: str) -> BaseModel:
    """Factory function to get model by name."""
    from models.decision_tree import DecisionTreeModel
    from models.mlp import MLPModel
    from models.knn import KNNModel
    from models.linear_regression import LinearRegressionModel
    models = {
        'decision_tree': DecisionTreeModel,
        'mlp': MLPModel,
        'knn': KNNModel,
        'linear_regression': LinearRegressionModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]()