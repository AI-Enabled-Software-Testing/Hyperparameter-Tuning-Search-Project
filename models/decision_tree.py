from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from models.base import BaseModel

class DecisionTreeModel(BaseModel):
    """Decision Tree classifier wrapper."""
    
    def create_model(self, **params):
        self.model = DecisionTreeClassifier(**params)
        return self.model
    
    def get_param_space(self) -> Dict[str, list]:
        return {
            'max_depth': list(range(3, 21)),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 11)),
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }