from sklearn.tree import DecisionTreeClassifier
from typing import Dict
from models.base import BaseModel
from .ParamSpace import ParamSpace

class DecisionTreeModel(BaseModel):
    """Decision Tree classifier wrapper."""
    
    def create_model(self, **params):
        self.model = DecisionTreeClassifier(**params)
        return self.model
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'max_depth': ParamSpace.integer(min_val=3, max_val=20, default=10),
            'min_samples_split': ParamSpace.integer(min_val=2, max_val=20, default=5),
            'min_samples_leaf': ParamSpace.integer(min_val=1, max_val=10, default=2),
            'criterion': ParamSpace.categorical(choices=['gini', 'entropy'], default='gini'),
            'splitter': ParamSpace.categorical(choices=['best', 'random'], default='best')
        }