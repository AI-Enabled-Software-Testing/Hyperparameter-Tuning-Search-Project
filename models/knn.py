from models.base import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict
from .ParamSpace import ParamSpace

class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier wrapper."""
    
    def create_model(self, **params):
        self.model = KNeighborsClassifier(**params)
        return self.model
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'n_neighbors': ParamSpace.integer(min_val=3, max_val=15, default=5),
            'weights': ParamSpace.categorical(choices=['uniform', 'distance'], default='uniform'),
            'algorithm': ParamSpace.categorical(choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto'),
            'p': ParamSpace.categorical(choices=[1, 2], default=2),  # Manhattan and Euclidean distance
            'metric': ParamSpace.categorical(choices=['minkowski', 'chebyshev', 'manhattan'], default='minkowski')
        }