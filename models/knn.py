from models.base import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict

class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier wrapper."""
    
    def create_model(self, **params):
        self.model = KNeighborsClassifier(**params)
        return self.model
    
    def get_param_space(self) -> Dict[str, list]:
        return {
            'n_neighbors': list(range(3, 16)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2],  # Manhattan and Euclidean distance
            'metric': ['minkowski', 'chebyshev', 'manhattan']
        }