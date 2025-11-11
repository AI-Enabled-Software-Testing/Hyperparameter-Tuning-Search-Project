from typing import Dict
from models.base import BaseModel
from sklearn.linear_model import LinearRegression
from .ParamSpace import ParamSpace

class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""
    
    def create_model(self, **params):
        self.model = LinearRegression(**params)
        return self.model
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'C': ParamSpace.categorical(choices=[10**i for i in range(-2, 3)], default=1.0), # Regularization strength
            'penalty': ParamSpace.categorical(choices=['l1', 'l2', 'elasticnet'], default='l2'),
            'solver': ParamSpace.categorical(choices=['liblinear', 'saga'], default='liblinear'),
            'l1_ratio': ParamSpace.categorical(choices=[i/10 for i in range(1, 10, 2)], default=0.5)  # For elasticnet
        }