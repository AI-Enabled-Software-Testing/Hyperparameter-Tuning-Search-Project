from dataclasses import dataclass
from typing import Union, List, Any, Dict
from enum import Enum

class ParamType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

@dataclass
class ParamSpace:
    param_type: ParamType
    min_value: Union[int, float, None] = None
    max_value: Union[int, float, None] = None
    choices: Union[List[Any], None] = None
    default: Any = None
    
    def __post_init__(self):
        """Validate parameter space configuration"""
        if self.param_type in [ParamType.INTEGER, ParamType.FLOAT]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"min_value and max_value required for {self.param_type.value}")
        elif self.param_type == ParamType.CATEGORICAL:
            if not self.choices:
                raise ValueError("choices required for categorical parameters")
    
    @classmethod
    def integer(cls, min_val: int, max_val: int, default: int = None):
        """Create an integer parameter space"""
        return cls(ParamType.INTEGER, min_val, max_val, default=default)
    
    @classmethod
    def float_range(cls, min_val: float, max_val: float, default: float = None):
        """Create a float parameter space"""
        return cls(ParamType.FLOAT, min_val, max_val, default=default)
    
    @classmethod
    def categorical(cls, choices: List[Any], default: Any = None):
        """Create a categorical parameter space"""
        return cls(ParamType.CATEGORICAL, choices=choices, default=default)
    
    @classmethod
    def boolean(cls, default: bool = None):
        """Create a boolean parameter space"""
        return cls(ParamType.BOOLEAN, choices=[True, False], default=default) 
