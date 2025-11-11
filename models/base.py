"""Machine learning models for hyperparameter tuning experiments.

This module contains base class for a general ML model (to be inherited)
that will be used in hyperparameter optimization experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections.abc import Iterable
import warnings
from .ParamSpace import ParamSpace


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs
        
    @abstractmethod
    def create_model(self, **params):
        """Create model instance with given parameters."""
        raise NotImplementedError("Subclasses must implement create_model method.")
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, ParamSpace]:
        """Get parameter space for hyperparameter tuning."""
        raise NotImplementedError("Subclasses must implement get_param_space method.")
    
    def train(self, X_train, y_train: Iterable):
        """Fit the model."""
        if not hasattr(self.model, "fit"):
            print(f"Model {self.model} does not support training.")
            return None
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X_test: Iterable, train_with_pred: bool = False):
        """Make predictions."""
        if train_with_pred:
            if hasattr(self.model, "fit"):
                self.model.fit(X_test)
        if not hasattr(self.model, "predict"):
            print(f"Model {self.model} does not support predictions.")
            return np.array([])
        if self.model is None:
            print("Model is not trained yet.")
            return np.array([])
        return self.model.predict(X_test)

    def predict_proba(self, X_test: Iterable, train_with_pred: bool = False):
        """Predict class probabilities."""
        if train_with_pred:
            if hasattr(self.model, "fit"):
                self.model.fit(X_test)
        if not hasattr(self.model, "predict_proba"):
            print(f"Model {self.model} does not support probability predictions.")
            return np.array([])
        if self.model is None:
            print("Model is not trained yet.")
            return np.array([])
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test: Iterable, y_test: Iterable) -> dict:
        """Evaluate model and return accuracy."""
        # calculate ROC AUC
        if hasattr(self.model, "predict"):
            y_pred = self.predict(X_test)
        elif hasattr(self.model, "predict_proba"):
            y_proba = self.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)
        else:
            # Fallback metrics if no predictions possible
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "F1 (Macro)": 0.0,
                "F1 (Micro)": 0.0,
                "ROC AUC": 0.5  # Random chance
            }
        
        # Ensure we have valid predictions
        if len(y_pred) == 0 or len(y_test) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "F1 (Macro)": 0.0,
                "F1 (Micro)": 0.0,
                "ROC AUC": 0.5
            }
        
        # Calculate basic accuracy (always works)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Check for edge cases that cause UndefinedMetricWarnings
        unique_true = np.unique(y_test)
        unique_pred = np.unique(y_pred)
        n_classes_true = len(unique_true)
        n_classes_pred = len(unique_pred)
        
        # Initialize metrics with fallback values
        metrics = {
            "accuracy": accuracy,
            "precision": 0.0,
            "recall": 0.0, 
            "F1 (Macro)": 0.0,
            "F1 (Micro)": 0.0,
            "ROC AUC": 0.5
        }
        
        # Only calculate advanced metrics if we have enough classes
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings
                
                # Precision, Recall, F1 - handle missing classes gracefully
                if n_classes_true > 1 and n_classes_pred > 1:
                    metrics["precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics["recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics["F1 (Macro)"] = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    metrics["F1 (Micro)"] = f1_score(y_test, y_pred, average='micro', zero_division=0)
                elif n_classes_true == 1:
                    # Perfect classification if only one true class and predictions match
                    if unique_true[0] in unique_pred:
                        metrics["precision"] = 1.0
                        metrics["recall"] = 1.0
                        metrics["F1 (Macro)"] = 1.0
                        metrics["F1 (Micro)"] = 1.0
                
                # ROC AUC - only calculate if we have probabilities and multiple classes
                if hasattr(self.model, "predict_proba") and n_classes_true > 1:
                    try:
                        y_proba = self.model.predict_proba(X_test)
                        if y_proba.shape[1] > 1:  # Multi-class probabilities
                            metrics["ROC AUC"] = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')
                    except (ValueError, IndexError):
                        # Keep default value of 0.5 if ROC AUC calculation fails
                        pass
                
        except Exception:
            # If any metric calculation fails, keep the fallback values
            pass
            
        return metrics
    
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