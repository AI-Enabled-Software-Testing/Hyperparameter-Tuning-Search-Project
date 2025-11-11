from typing import Dict, Any
from models.base import BaseModel
from sklearn.neural_network import MLPClassifier
from .ParamSpace import ParamSpace

class MLPModel(BaseModel):
    """Multi-Layer Perceptron classifier wrapper."""
    
    def create_model(self, **params):
        self.model = MLPClassifier(max_iter=1000, **params)
        return self.model
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'hidden_layer_sizes': ParamSpace.categorical(choices=[(50,), (100,), (50, 50), (100, 50), (100, 100)], default=(100,)),
            'activation': ParamSpace.categorical(choices=['relu', 'tanh', 'logistic'], default='relu'),
            'solver': ParamSpace.categorical(choices=['adam', 'sgd', 'lbfgs'], default='adam'),
            'alpha': ParamSpace.categorical(choices=list([10**i for i in range(-4, 0)]), default=0.0001),
            'learning_rate': ParamSpace.categorical(choices=['constant', 'invscaling', 'adaptive'], default='constant')
        }