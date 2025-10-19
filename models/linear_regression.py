from typing import Dict, Any
from models.base import BaseModel
from sklearn.linear_model import LinearRegression

class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""
    
    def create_model(self, **params):
        self.model = LinearRegression(max_iter=1000, **params)
        return self.model
    
    def get_param_space(self) -> Dict[str, Any]:
        return {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # For elasticnet
        }