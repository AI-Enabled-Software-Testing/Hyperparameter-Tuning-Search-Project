from typing import Dict
from models.base import BaseModel
from sklearn.linear_model import LinearRegression

class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""
    
    def create_model(self, **params):
        self.model = LinearRegression(**params)
        return self.model
    
    def get_param_space(self) -> Dict[str, list]:
        return {
            'C': [10**i for i in range(-2, 3)], # Regularization strength
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': list(range(0.1, 1.0, 0.2)) # For elasticnet
        }