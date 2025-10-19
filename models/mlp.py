from typing import Dict, Any
from models.base import BaseModel
from sklearn.neural_network import MLPClassifier

class MLPModel(BaseModel):
    """Multi-Layer Perceptron classifier wrapper."""
    
    def create_model(self, **params):
        self.model = MLPClassifier(max_iter=1000, **params)
        return self.model
    
    def get_param_space(self) -> Dict[str, Any]:
        return {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }